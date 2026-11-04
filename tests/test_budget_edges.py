"""Additional named regression boundaries, without live models or deployment."""
from __future__ import annotations
import asyncio
import dataclasses
import json
import pytest
from .test_budget import api, body, budget, measured


@pytest.mark.asyncio
@pytest.mark.parametrize("number", [11, 46, 48, 49, 57])
async def test_synchronous_holds_and_waiter_recheck(tmp_path, number):
    b = budget(tmp_path, limit=1000, overshoot="one_call", wait_seconds=1)
    p = b.prepare("anthropic", "m", body("x" * 600, max_tokens=100))
    a = await b.admit(p, "a")
    waiter = asyncio.create_task(b.admit(p, "b"))
    await asyncio.sleep(0)
    assert not waiter.done()
    assert b.held == p.reserved
    b.settle(a, measured(650, 100))
    c = await waiter
    assert b.held == p.reserved
    b.settle(c, measured(650, 100))
    assert b.spent == 850
    assert b.crossing_allowance == p.reserved
    await b.close_out()


@pytest.mark.asyncio
async def test_40_51_two_ledgers_and_forced_close(tmp_path):
    a, b = budget(tmp_path / "a", close_grace_seconds=0), budget(tmp_path / "b")
    p = a.prepare("anthropic", "m", body())
    await a.admit(p, "a")
    await a.close_out()
    assert a.spent == p.reserved and b.spent == 0
    assert a.ledger_id != b.ledger_id
    await b.close_out()


@pytest.mark.asyncio
@pytest.mark.parametrize("number,connect,sent,usage,expected", [
    (32, False, True, (10, 2), 12), (36, False, False, (None, None), 0),
    (31, True, True, (None, None), 0), (38, False, True, (10, None), None),
])
async def test_stage_settlement(number, tmp_path, connect, sent, usage, expected):
    b = budget(tmp_path)
    p = b.prepare("anthropic", "m", body())
    a = await b.admit(p, "a")
    facts = b.settle(a, api().Measurement(n=usage[0], o=usage[1], terminal=usage[1] is not None),
                     sent=sent, connect_failed=connect)
    assert facts["charge"] == (20 if expected is None else expected)
    await b.close_out()


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["limit", "insufficient"])
async def test_65_75_84_durable_refusal_is_the_only_budget_witness(tmp_path, cause):
    b = budget(tmp_path, limit=300, max_reservation=500, overshoot="one_call" if cause == "limit" else "none")
    p = b.prepare("anthropic", "m", body("x" * (200 if cause == "limit" else 100), max_tokens=100))
    a = await b.admit(p, "a")
    b.settle(a, measured(200, 100) if cause == "limit" else measured(100, 100))
    with pytest.raises(api().BudgetRefused) as refusal:
        await b.admit(p, "refusal")
    await b.close_out()
    b.snapshot_path.write_text('{"lifecycle":"closed","refusals":{"token_budget":999}}')
    r = api().read_budget(tmp_path)
    assert r["complete"] and r["latch_cause"] == cause
    kwargs = dict(trial="trusted-trial", config_hash=b.config_hash, hsap_commit=b.config.hsap_commit,
                  limit=300, grade=1, native_terminal=str(refusal.value))
    ending = api().classify_ending(r, **kwargs)
    assert ending.ending == "budget_exhausted" and ending.grade == 1
    r["refusal_witnesses"] = []
    assert api().classify_ending(r, **kwargs).ending == "harness_failure"


@pytest.mark.asyncio
async def test_3_86_none_profile_linear_continuation_is_not_a_reset(tmp_path):
    b = budget(tmp_path)
    b.config.profiles["m"].credit = "none"
    # Recreate plans after changing the profile; structural prefixes remain independent.
    first = {"role": "user", "content": "a" * 200}
    for i, messages in enumerate(([first], [first, {"role": "user", "content": "b" * 200}])):
        a = await b.admit(b.prepare("anthropic", "m", body(messages=messages, max_tokens=100)), str(i))
        b.settle(a, measured(100 * (i + 1), 10))
    assert b.record()["carried_context"] == 210
    assert b.record()["repeated_root_contexts"] == 0
    assert b.spent == 320
    await b.close_out()


@pytest.mark.asyncio
async def test_42_bound_violation_stops_admission_before_late_commit(tmp_path):
    b = budget(tmp_path)
    p = b.prepare("anthropic", "m", body())
    a = await b.admit(p, "slow")
    c = await b.admit(p, "unsafe")
    b.settle(c, measured(10000))
    with pytest.raises(api().BudgetRefused):
        await b.admit(p, "never sent")
    b.settle(a, measured(10))
    assert b.spent == 10000
    assert b.record()["latch_cause"] == "bound_violated"
    await b.close_out()


@pytest.mark.parametrize("number", [13, 14, 26])
def test_unsupported_inputs_and_payload_near_misses(tmp_path, number):
    b = budget(tmp_path)
    if number == 26:
        request = {"input": "hi", "previous_response_id": "known", "max_output_tokens": 10}
        dialect = "openai-responses"
    elif number == 14:
        request, dialect = body(messages=[{"role": "user", "content": "hi", "unknown": True}]), "anthropic"
    else:
        request, dialect = body(messages=[{"role": "user", "content": [{"type": "image", "source": {"url": "https://example.test"}}]}]), "anthropic"
    with pytest.raises(api().BudgetRefused):
        b.prepare(dialect, "m", request)
    b.release_writer()


@pytest.mark.asyncio
async def test_84_generation_selection_ignores_copy_timestamps(tmp_path):
    import os
    b = budget(tmp_path)
    await b.close_out()
    c = api().TrialBudget(b.config, tmp_path, "trusted-trial", new_ledger=True)
    a = await c.admit(c.prepare("anthropic", "m", body()), "joined")
    c.settle(a, measured(10))
    await c.close_out()
    os.utime(b.journal_path, ns=(999999999999999999, 1999999999999999999))
    r = api().read_budget(tmp_path)
    assert r["ledger_id"] == c.ledger_id
    assert r["attempts"]["joined:1"]["charge"] == 10


def test_65_wrong_trial_is_bound_before_incomplete_classification():
    record = dict(trial="foreign", config_hash="c", hsap_commit="h", limit=100,
                  rule="harbor-growth/1", complete=False, lifecycle="open")
    result = api().classify_ending(record, trial="expected", config_hash="c", hsap_commit="h", limit=100, grade=1)
    assert result.ending == "harness_failure"
    assert result.grade == 1


@pytest.mark.asyncio
async def test_84_fresh_process_recovers_commit_and_refusal(tmp_path):
    import subprocess
    import sys
    b = budget(tmp_path, limit=300, max_reservation=500, overshoot="one_call")
    p = b.prepare("anthropic", "m", body("x" * 200, max_tokens=100))
    a = await b.admit(p, "a")
    b.settle(a, measured(200, 100))
    with pytest.raises(api().BudgetRefused):
        await b.admit(p, "terminal")
    await b.close_out()
    b.snapshot_path.write_text('{}')
    command = [sys.executable, "-c", "import json,sys; from pathlib import Path; from harbor_sandbox_agent_proxy.budget import read_budget; print(json.dumps(read_budget(Path(sys.argv[1]))))", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    recovered = json.loads(result.stdout)
    assert recovered["complete"] and recovered["spent"] == 300
    assert recovered["refusals"] == {"token_budget": 1}


@pytest.mark.asyncio
async def test_84_unknown_settlement_stage_is_not_a_clean_record(tmp_path):
    b = budget(tmp_path)
    a = await b.admit(b.prepare("anthropic", "m", body()), "a")
    b.settle(a, measured(10))
    await b.close_out()
    events = [json.loads(line) for line in b.journal_path.read_text().splitlines()]
    for event in events:
        if event["kind"] == "settle":
            event["stage"] = "invented"
            event["h"] = api().event_hash(event)
    b.journal_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert not api().read_budget(tmp_path)["complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["null\n", "[]\n", "7\n"])
async def test_84_json_scalar_tail_is_incomplete_not_an_exception(tmp_path, value):
    b = budget(tmp_path)
    await b.close_out()
    with b.journal_path.open("a") as stream:
        stream.write(value)
    r = api().read_budget(tmp_path)
    assert not r["complete"] and r["incomplete_reason"] == "torn_tail"


@pytest.mark.asyncio
async def test_84_malformed_snapshot_is_only_a_stale_cache(tmp_path):
    b = budget(tmp_path)
    await b.close_out()
    b.snapshot_path.write_text('[]')
    r = api().read_budget(tmp_path)
    assert r["complete"] and r["snapshot_stale"]
