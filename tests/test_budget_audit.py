"""Nonvacuous controls for the independent replay and measurement gates."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from .test_budget import api, budget, measured


def test_68_oracle_discriminates_completion_order_and_collapsing_prefix():
    from .budget_replay import replay
    rows = [{"root": "r", "messages": ["same"], "n": n, "o": 0, "admit_seq": i + 1} for i, n in enumerate((300, 200, 100))]
    assert replay(list(reversed(rows)))["charges"] == [300, 200, 100]
    rows = [{"root": "r", "messages": ["a"], "n": 100, "o": 10, "admit_seq": 1},
            {"root": "r", "messages": ["a", "b"], "n": 200, "o": 20, "admit_seq": 2}]
    assert replay(rows)["charges"] == [110, 120]
    assert replay(rows, credit="none")["charges"] == [110, 220]


def test_82_m2_rejects_empty_and_dropped_thinking():
    from .budget_replay import stability
    assert not stability([])["passed"]
    rows = [{"n": 1000 + i * 10, "extends_previous": True, "tail_tokens": 10, "thinking_present": i % 2 == 0, "thinking_dropped": i % 2 == 1} for i in range(30)]
    assert stability(rows)["passed"]
    rows[10]["n"] = 10
    assert not stability(rows)["passed"]


def test_88_m3_never_accepts_canned_usage_empty_or_missing_bound_source():
    from .budget_replay import shadow_bound
    assert not shadow_bound([])["passed"]
    row = dict(final_body_bytes=100, r_in=100, n=90, o=10, successful=True, terminal=True,
               real_route=True, profile="model/mode", bound_source="provider", bound_version="1",
               route_bound_source="gateway config", route_bound_version="1", tools_transition=True,
               thinking_transition=True)
    assert shadow_bound([row])["passed"]
    for key, value in (("real_route", False), ("n", 101), ("bound_source", ""), ("terminal", False)):
        assert not shadow_bound([{**row, key: value}])["passed"]


@pytest.mark.asyncio
async def test_1_fixture_replay_matches_independent_oracle(tmp_path):
    from .budget_replay import replay
    source = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "claude_code_relay_calls.jsonl").read_text().splitlines()]
    b = budget(tmp_path, enforce=False, inject_output_limit=True)
    rows, charges = [], []
    for i, call in enumerate(source):
        payload = copy.deepcopy(call["request"])
        plan = b.prepare("anthropic", "m", payload)
        meter = api().UsageMeter("anthropic")
        meter.observe(call["response"])
        usage = meter.measurement()
        a = await b.admit(plan, str(i))
        charges.append(b.settle(a, usage)["charge"])
        rows.append(dict(root="fixture", messages=payload["messages"], n=usage.n, o=usage.o, admit_seq=i + 1))
    expected = [10457, 2265, 695, 590, 680, 683, 680]
    assert charges == expected
    assert replay(rows)["charges"] == expected
    assert b.spent == 16050
    assert b.record()["billed_tokens"] == 89595
    await b.close_out()
