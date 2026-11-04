"""Offline controls for the v3 budget; live lane gates are documented separately."""
from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from .test_budget import api, body, budget, measured, settings


@pytest.mark.parametrize("number,field,value", [(73, f, v) for f in ("max_tokens", "n")
    for v in (-1, 0, True, 1.5, "10", 2**63)])
def test_73_strict_values_leave_ledger_untouched(tmp_path, number, field, value):
    b = budget(tmp_path)
    before = b.journal_path.read_bytes()
    with pytest.raises(api().BudgetRefused):
        b.prepare("openai", "m", body(**{field: value}))
    assert b.held == b.spent == 0
    assert b.journal_path.read_bytes() == before
    b.release_writer()


@pytest.mark.asyncio
@pytest.mark.parametrize("number,change", [(12, {"system": "different"}), (19, {"messages": [{"role": "user", "content": "edited"}]}),
                                            (7, {"messages": [{"role": "user", "content": "unrelated"}]})])
async def test_identity_changes_pay_whole_prompt(tmp_path, number, change):
    b = budget(tmp_path)
    first = body("x" * 500)
    a = await b.admit(b.prepare("anthropic", "m", first), "a")
    b.settle(a, measured(100, 10))
    a = await b.admit(b.prepare("anthropic", "m", {**first, **change}), "b")
    b.settle(a, measured(100, 10))
    assert b.spent == 220
    await b.close_out()


@pytest.mark.asyncio
@pytest.mark.parametrize("number", [2, 4, 5, 6, 10, 16, 17, 18, 47, 79, 81])
async def test_prefix_growth_duplicates_and_collapse(tmp_path, number):
    b = budget(tmp_path)
    request = body("x" * 1000)
    charges = []
    for i, n in enumerate((300, 200, 100)):
        a = await b.admit(b.prepare("anthropic", "m", request), str(i))
        facts = b.settle(a, measured(n))
        charges.append(facts["charge"])
    assert charges == [300, 200, 100]
    assert b.spent == 600
    assert b.record()["checkpoint_conflicts"] == 2
    assert b.record()["credit_invalidations"] == 2
    await b.close_out()


@pytest.mark.asyncio
async def test_80_pending_records_retain_no_body_and_fence_immediately(tmp_path):
    b = budget(tmp_path, unknown_usage="stop")
    p = b.prepare("anthropic", "m", body("x" * 500))
    a = await b.admit(p, "slow")
    c = await b.admit(p, "fast")
    held = b.held
    facts = b.settle(c, api().Measurement(n=10, o=None, terminal=False), sent=True)
    assert facts["state"] == "settled_pending_commit"
    assert facts["charge"] is None
    assert b.held == held
    assert not hasattr(c, "body") and not hasattr(c, "plan")
    with pytest.raises(api().BudgetRefused):
        await b.admit(p, "blocked")
    b.settle(a, measured(10))
    assert b.held == 0
    await b.close_out()


@pytest.mark.asyncio
@pytest.mark.parametrize("number", [39, 60, 75, 84])
async def test_journal_authority_over_stale_snapshot(tmp_path, number):
    b = budget(tmp_path)
    a = await b.admit(b.prepare("anthropic", "m", body("x" * 200)), "a")
    b.settle(a, measured(100, 10))
    await b.close_out()
    snapshot = b.snapshot_path
    snapshot.write_text('{"spent":0,"lifecycle":"open"}')
    record = api().read_budget(tmp_path)
    assert record["spent"] == 110
    assert record["complete"] is True
    assert record["snapshot_stale"] is True
    assert record["latch_cause"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("damage,want", [("tail", "torn_tail"), ("hash", "torn_tail"), ("settle", "duplicate_settle")])
async def test_84_corruption_is_never_promoted_by_snapshot(tmp_path, damage, want):
    b = budget(tmp_path)
    a = await b.admit(b.prepare("anthropic", "m", body()), "a")
    b.settle(a, measured(10))
    await b.close_out()
    path = b.journal_path
    lines = path.read_bytes().splitlines(keepends=True)
    if damage == "tail":
        path.write_bytes(b"".join(lines) + b'{"unfinished":')
    elif damage == "hash":
        path.write_bytes(b"".join(lines[:-1]) + lines[-1].replace(b'"h":"', b'"h":"bad'))
    else:
        event = next(json.loads(line) for line in lines if json.loads(line)["kind"] == "settle")
        events = [json.loads(line) for line in lines]
        events.insert(3, event)
        for seq, event in enumerate(events):
            event["seq"] = seq
            event.pop("h", None)
            event["h"] = api().event_hash(event)
        path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
    result = api().read_budget(tmp_path)
    assert result["complete"] is False
    assert result["incomplete_reason"] == want


@pytest.mark.asyncio
async def test_85_forced_close_owns_settlement_once(tmp_path):
    b = budget(tmp_path, close_grace_seconds=0)
    a = await b.admit(b.prepare("anthropic", "m", body()), "slow")
    reserved = b.held
    await b.close_out()
    before = b.journal_path.read_bytes()
    b.settle(a, measured(1))
    assert b.spent == reserved
    assert b.journal_path.read_bytes() == before
    assert api().read_budget(tmp_path)["forced_settled"] == [1]


@pytest.mark.asyncio
async def test_83_persistence_failure_fences_before_send(tmp_path, monkeypatch):
    b = budget(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")
    monkeypatch.setattr(b, "_append", fail)
    with pytest.raises(OSError):
        await b.admit(b.prepare("anthropic", "m", body()), "a")
    assert b.admission_fence == "persistence_failed"
    assert b.record_error is not None
    b.release_writer()


@pytest.mark.asyncio
async def test_84_exclusive_writer_and_explicit_generation(tmp_path):
    b = budget(tmp_path)
    with pytest.raises(RuntimeError):
        budget(tmp_path)
    b.release_writer()
    with pytest.raises(RuntimeError):
        budget(tmp_path)
    c = api().TrialBudget(settings(), tmp_path, "trusted-trial", new_ledger=True)
    assert c.previous_ledger_id == b.ledger_id
    assert c.journal_path != b.journal_path
    await c.close_out()


@pytest.mark.parametrize("number,updates,ending", [
    (74, {}, "natural"), (65, {"agent_error": "ordinary agent failure"}, "natural"),
    (66, {"runner_error": "runner broke"}, "harness_failure"),
    (84, {"complete": False, "lifecycle": "open"}, "accounting_censored"),
    (38, {"bound_charged": 6000}, "accounting_censored"),
    (75, {"latch_cause": "usage_unknown"}, "accounting_censored"),
])
def test_ending_matrix_retains_every_grade(number, updates, ending):
    record = dict(trial="t", config_hash="c", hsap_commit="h", limit=1_000_000,
                  rule="harbor-growth/1", enforce=True, spent=100_000,
                  complete=True, lifecycle="closed", latch_cause=None,
                  bound_charged=0, partial_charged=0, refusals={})
    agent_error = updates.pop("agent_error", None)
    runner_error = updates.pop("runner_error", None)
    record.update(updates)
    result = api().classify_ending(record, trial="t", config_hash="c", hsap_commit="h",
                                   limit=1_000_000, grade=0.7, agent_error=agent_error,
                                   runner_error=runner_error)
    assert result.ending == ending
    assert result.grade == 0.7


@pytest.mark.asyncio
@pytest.mark.parametrize("number,credit", [(81, "none"), (86, "eligible")])
async def test_86_structural_context_does_not_depend_on_numeric_credit(tmp_path, number, credit):
    cfg = settings()
    cfg.profiles["m"].credit = credit
    b = api().TrialBudget(cfg, tmp_path, "t")
    for i, text in enumerate(("a" * 1000, "b" * 200)):
        p = b.prepare("anthropic", "m", body(text, max_tokens=100))
        a = await b.admit(p, str(i))
        b.settle(a, measured(1000 if i == 0 else 200, 100 if i == 0 else 10))
    assert b.record()["carried_context"] == 1310
    assert b.record()["peak_context_per_root"] == 1100
    await b.close_out()


@pytest.mark.asyncio
async def test_88_unknown_injection_is_detected_not_invented(tmp_path):
    b = budget(tmp_path, enforce=False)
    a = await b.admit(b.prepare("anthropic", "m", body()), "a")
    b.settle(a, measured(9000))
    assert b.record()["latch_cause"] == "bound_violated"
    assert b.record()["enforce"] is False
    await b.close_out()
