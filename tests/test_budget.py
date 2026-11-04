"""Numbered design regressions. Expectations are literals, not relay-derived totals."""
from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
from pathlib import Path

import pytest

from . import budget_v2_rules


def api():
    if os.environ.get("BUDGET_RULES") == "v2":
        return budget_v2_rules
    try:
        return importlib.import_module("harbor_sandbox_agent_proxy.budget")
    except ModuleNotFoundError as error:
        if error.name != "harbor_sandbox_agent_proxy.budget":
            raise
        return budget_v2_rules


def settings(**updates):
    values = dict(
        limit=1_000_000, rule="harbor-growth/1", overshoot="none",
        unknown_usage="bound", enforce=True, inject_output_limit=False,
        hsap_commit="9e12cc441bc9", output_ceiling={"m": 100_000},
        profiles={"m": dict(id="offline/verbatim", credit="eligible",
                             credit_evidence="offline M2 fixture", bound_source="provider contract",
                             bound_version="test/1", route_injected_tokens_max=0,
                             route_bound_source="offline route config", route_bound_version="1")},
    )
    values.update(updates)
    if api() is budget_v2_rules:
        return budget_v2_rules.config(**values)
    return api().TokenBudgetConfig(**values)


def budget(tmp_path, **updates):
    return api().TrialBudget(settings(**updates), tmp_path, "trusted-trial")


def body(text="hello", **updates):
    result = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 10}
    result.update(updates)
    return result


def measured(n, o=0):
    return api().Measurement(n=n, o=o, terminal=True)


@pytest.mark.asyncio
async def test_71_paid_root_reserves_every_forwarded_byte(tmp_path):
    b = budget(tmp_path)
    p = b.prepare("anthropic", "m", body("x" * 890_000, system="s" * 100_000))
    a = await b.admit(p, "first")
    b.settle(a, measured(990_000))
    p = b.prepare("anthropic", "m", body("unrelated", system="s" * 100_000))
    with pytest.raises(api().BudgetRefused):
        await b.admit(p, "second")
    assert b.held == 0
    await b.close_out()


@pytest.mark.asyncio
async def test_72_negative_branch_never_creates_a_hold(tmp_path):
    b = budget(tmp_path)
    with pytest.raises(api().BudgetRefused):
        p = b.prepare("openai", "m", body(n=-10, max_tokens=20_000))
        await b.admit(p, "negative")
    assert b.held == 0
    assert b.spent == 0
    await b.close_out()


def test_74_natural_success_keeps_grade_below_cap():
    record = dict(trial="trusted-trial", config_hash="hash", hsap_commit="sha",
                  rule="harbor-growth/1", enforce=True, limit=10_000_000,
                  spent=100_000, lifecycle="closed", complete=True, latch_cause=None,
                  bound_charged=0, partial_charged=0, refusals={}, profiles={})
    result = api().classify_ending(record, trial="trusted-trial", config_hash="hash",
                                   hsap_commit="sha", limit=10_000_000, grade=1.0)
    assert result.ending == "natural"
    assert result.accepted is True
    assert result.grade == 1.0


@pytest.mark.asyncio
async def test_79_commits_follow_admission_not_completion(tmp_path):
    totals = []
    for reverse in (False, True):
        b = budget(tmp_path / str(reverse))
        p = b.prepare("anthropic", "m", body("x" * 500))
        a = await b.admit(p, "first")
        c = await b.admit(p, "second")
        d = await b.admit(p, "third")
        order = [(a, measured(100)), (c, measured(200)), (d, measured(300))]
        for item, usage in reversed(order) if reverse else order:
            b.settle(item, usage)
        totals.append(b.spent)
        await b.close_out()
    assert totals == [400, 400]


@pytest.mark.asyncio
@pytest.mark.parametrize("sequences,counts,expected,latch_at", [
    ([1, 2], [100, 1000], [100, 1000], 2),
    ([2, 1], [1000, 100], [1000, 1100], 1),
    ([1, 1, 1], [100, 200, 300], [100, 200, 400], 3),
    ([1, 1, 1], [300, 200, 100], [300, 500, 600], 2),
])
async def test_ordered_commit_matrix(tmp_path, sequences, counts, expected, latch_at):
    # Design 79: identical admissions/measurements, including the corrected 600 row.
    results = []
    for reverse in (False, True):
        b = budget(tmp_path / str(reverse), limit=350, max_reservation=10000, enforce=False)
        attempts = []
        messages = [{"role": "user", "content": "a" * 2000}, {"role": "assistant", "content": "b" * 2000}]
        for i, size in enumerate(sequences):
            p = b.prepare("anthropic", "m", body(messages=messages[:size]))
            attempts.append(await b.admit(p, str(i)))
        pairs = list(zip(attempts, counts))
        for a, n in reversed(pairs) if reverse else pairs:
            b.settle(a, measured(n))
        await b.close_out()
        commits = [e for e in map(json.loads, b.journal_path.read_text().splitlines()) if e["kind"] == "commit"]
        assert [e["spent_after"] for e in commits] == expected
        assert next(e["admit_seq"] for e in commits if e["latch_cause_after"] == "limit") == latch_at
        results.append(b.spent)
    assert results == [expected[-1], expected[-1]]
