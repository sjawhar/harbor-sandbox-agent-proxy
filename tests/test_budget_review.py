"""Review regressions at the sender, journal and independent-gate boundaries."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from harbor_sandbox_agent_proxy import budget as mod, relay
from harbor_sandbox_agent_proxy.routing import Route
from tests.budget_replay import replay, shadow_bound, stability
from tests.test_budget import body, budget, measured, settings


async def send(b, method, request):
    sent, calls = [], []
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        calls.append(call)
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"input_tokens": 10, "output_tokens": 1}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        methods = relay.relay_methods(
            lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
            route, record=record, client=client, budget=b,
        )
        result = await methods[method](request, {"authorization": "Bearer caller"})
    return result, sent, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("output", [
    [{"type": "input_image", "image_url": "https://example.test/image.png"}],
    [{"type": "input_file", "file_id": "account-file"}],
    [{"type": "input_audio", "data": "encoded"}],
])
async def test_f1_tool_output_modalities_refused_before_send(tmp_path, kind, output):
    b = budget(tmp_path)
    before = b.journal_path.read_bytes()
    payload = {"model": "m", "input": [{"type": kind, "call_id": "c", "output": output}], "max_output_tokens": 10}
    result, sent, calls = await send(b, "generate_responses", payload)
    assert sent == []
    assert result[relay.PROVIDER_ERROR_KEY]["code"] == "unsupported_input"
    assert b.journal_path.read_bytes() == before
    await b.close_out()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("output", ["text with https://example.test", [{"type": "input_text", "text": "tool result"}]])
async def test_f1_text_tool_outputs_keep_their_payload(tmp_path, kind, output):
    b = budget(tmp_path)
    payload = {"model": "m", "input": [{"type": kind, "call_id": "c", "output": output}], "max_output_tokens": 10}
    result, sent, calls = await send(b, "generate_responses", payload)
    assert len(sent) == 1 and sent[0]["input"][0]["output"] == output
    await b.close_out()


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["anthropic", "responses", "namespace", "additional_tools", "tool_search_output", "google"])
async def test_f2_server_declarations_are_stripped_at_every_site(tmp_path, site):
    b = budget(tmp_path)
    server = {"type": "web_search", "name": "web_search", "max_uses": 3}
    if site == "anthropic":
        server["type"] = "web_search_20250305"
        method, payload = "generate_anthropic", body(tools=[server])
    elif site == "google":
        method = "generate_google"
        payload = {"model": "m", "contents": [], "tools": [{"googleMaps": {"enableWidget": True}}], "generationConfig": {"maxOutputTokens": 10}}
    else:
        method, payload = "generate_responses", {"model": "m", "input": [], "max_output_tokens": 10}
        if site in ("additional_tools", "tool_search_output"):
            item = {"type": site, "tools": [server]}
            if site == "tool_search_output":
                item.update(call_id="c", execution="client")
            payload["input"] = [item]
        else:
            payload["tools"] = [{"type": "namespace", "name": "n", "tools": [server]}] if site == "namespace" else [server]
    result, sent, calls = await send(b, method, payload)
    assert len(sent) == 1
    assert "tools" not in sent[0] or not sent[0]["tools"]
    assert calls[0].stripped
    if site in ("additional_tools", "tool_search_output"):
        assert sent[0]["input"][0]["tools"] == []
    await b.close_out()


@pytest.mark.asyncio
async def test_f3_concurrent_close_has_one_owner_and_one_result(tmp_path):
    b = budget(tmp_path, close_grace_seconds=.01)
    ready = asyncio.Event()
    async def handler():
        a = await b.admit(b.prepare("anthropic", "m", body()), "slow")
        ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            b.settle(a, mod.Measurement())
    worker = asyncio.create_task(handler())
    await ready.wait()
    results = await asyncio.gather(b.close_out(), b.close_out(), b.close_out())
    await asyncio.gather(worker, return_exceptions=True)
    kinds = [json.loads(line)["kind"] for line in b.journal_path.read_text().splitlines()]
    assert kinds.count("close_begin") == kinds.count("close") == 1
    assert results[0] == results[1] == results[2]
    assert mod.read_budget(tmp_path)["complete"]


def m2_rows():
    return [dict(n=1000 + i * 10, tail_tokens=10, extends_previous=True,
                 thinking_present=i % 2 == 0, thinking_dropped=i % 2 == 1) for i in range(30)]


@pytest.mark.parametrize("mutation", ["no_pairs", "zero_prompt", "bool_prompt", "negative_tail", "coverage_only_outside_pairs"])
def test_f4_m2_requires_measured_comparable_thinking_pairs(mutation):
    rows = m2_rows()
    assert stability(rows)["passed"]
    if mutation == "no_pairs":
        for row in rows:
            row["extends_previous"] = False
    elif mutation == "coverage_only_outside_pairs":
        for row in rows:
            if row["thinking_dropped"]:
                row["extends_previous"] = False
    else:
        rows[5]["n" if mutation != "negative_tail" else "tail_tokens"] = {"zero_prompt": 0, "bool_prompt": True, "negative_tail": -1}[mutation]
    assert not stability(rows)["passed"]


def test_f4_m3_zero_usage_cannot_pass():
    row = dict(final_body_bytes=100, r_in=100, n=0, o=0, successful=True, terminal=True,
               real_route=True, profile="model/mode", bound_source="provider", bound_version="1",
               route_bound_source="gateway", route_bound_version="1", tools_transition=True,
               thinking_transition=True)
    assert not shadow_bound([row])["passed"]
    assert shadow_bound([{**row, "n": 90, "o": 0}])["passed"]


def replay_row(i, root, expected, *, credit="eligible", dialect="anthropic", messages=None):
    return dict(admit_seq=i, model="m", dialect=dialect, profile=f"offline/{credit}", credit=credit,
                root=root, messages=messages or ["same"], n=100, o=0, committed_charge=expected)


def test_f5_replay_requires_every_committed_charge_and_frozen_profile():
    row = replay_row(1, "root", 100)
    assert replay([row])["passed"]
    for field in ("committed_charge", "model", "dialect", "profile", "credit"):
        incomplete = {k: v for k, v in row.items() if k != field}
        assert not replay([incomplete])["passed"]
    wrong = {**row, "committed_charge": 99}
    assert not replay([wrong])["passed"]


def test_f5_mixed_profiles_use_each_rows_frozen_policy():
    rows = [replay_row(1, "eligible-root", 100), replay_row(2, "none-root", 100, credit="none"),
            replay_row(3, "eligible-root", 0), replay_row(4, "none-root", 100, credit="none")]
    result = replay(rows)
    assert result["passed"] and result["charges"] == [100, 100, 0, 100]
    # The review's original positive omitted the very profile information it needed.
    missing = [{k: v for k, v in row.items() if k not in ("model", "dialect", "profile", "credit")} for row in rows]
    assert not replay(missing, credit="eligible")["passed"]
    assert not replay(missing, credit="none")["passed"]


@pytest.mark.parametrize("dialect,expected", [("anthropic", [100, 0]), ("openai", [100, 100]), ("openai-responses", [100, 100])])
def test_f5_oracle_cache_normalization_is_path_and_dialect_specific(dialect, expected):
    rows = [replay_row(i + 1, "root", expected[i], dialect=dialect,
                       messages=[{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": control}]}])
            for i, control in enumerate(({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "1h"}))]
    result = replay(rows)
    assert result["charges"] == expected and result["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("crossing", [False, True])
async def test_f6_crossing_allowance_recovered_without_snapshot(tmp_path, crossing):
    b = budget(tmp_path, limit=300 if crossing else 3000, max_reservation=2000, overshoot="one_call")
    p = b.prepare("anthropic", "m", body("x" * 400, max_tokens=100))
    a = await b.admit(p, "a")
    b.settle(a, measured(400, 100))
    await b.close_out()
    b.snapshot_path.unlink()
    record = mod.read_budget(tmp_path)
    assert record["complete"] and record["crossing_allowance"] == (p.reserved if crossing else 0)


@pytest.mark.asyncio
async def test_f8_rerolls_and_their_continuation_are_not_forks(tmp_path):
    b = budget(tmp_path)
    messages = [{"role": "user", "content": "x" * 200}]
    for i in range(3):
        a = await b.admit(b.prepare("anthropic", "m", body(messages=messages)), str(i))
        b.settle(a, measured(100, 1))
    messages.append({"role": "assistant", "content": "answer"})
    a = await b.admit(b.prepare("anthropic", "m", body(messages=messages)), "continuation")
    b.settle(a, measured(110, 1))
    result = await b.close_out()
    assert result["spent"] == 114 and result["forks"] == 0
    assert not result["carried_context_approximate"]


@pytest.mark.asyncio
async def test_f5_sender_charges_join_the_oracle_for_mixed_models(tmp_path):
    cfg = settings()
    cfg.output_ceiling["aux"] = 100000
    cfg.profiles["aux"] = cfg.profiles["m"].model_copy(update={"id": "offline/aux/none", "credit": "none"})
    b = mod.TrialBudget(cfg, tmp_path, "trusted-trial")
    rows = []
    for i, model in enumerate(("m", "aux", "m", "aux")):
        payload = body("x" * 200, model=model)
        a = await b.admit(b.prepare("anthropic", model, payload), str(i))
        b.settle(a, measured(100))
        profile = cfg.profiles[model]
        rows.append(dict(admit_seq=i + 1, model=model, dialect="anthropic", profile=profile.id,
                         credit=profile.credit, root={"model": model}, messages=payload["messages"], n=100, o=0))
    await b.close_out()
    joined = mod.read_budget(tmp_path)["attempts"]
    for i, row in enumerate(rows):
        row["committed_charge"] = joined[f"{i}:1"]["charge"]
    result = replay(rows)
    assert result["passed"] and result["charges"] == [100, 100, 0, 100]


@pytest.mark.asyncio
async def test_f5_chat_sender_join_uses_its_frozen_profile(tmp_path):
    b = budget(tmp_path)
    rows = []
    for i, control in enumerate(({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "1h"})):
        payload = body(messages=[{"role": "user", "content": [{"type": "text", "text": "x" * 200, "cache_control": control}]}])
        a = await b.admit(b.prepare("openai", "m", payload), str(i))
        facts = b.settle(a, measured(100))
        rows.append(replay_row(i + 1, {"model": "m"}, facts["charge"], dialect="openai", messages=payload["messages"]))
    result = replay(rows)
    assert result["passed"] and result["charges"] == [100, 100]
    await b.close_out()


@pytest.mark.asyncio
async def test_f3_trial_relay_close_waits_for_one_complete_outcome(tmp_path):
    from harbor_sandbox_agent_proxy.config import RelayConfig
    from harbor_sandbox_agent_proxy.outcome import TrialIdentity
    from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
    from tests.test_trial_relay import _CONNECTION, _Sink
    async def route(call):
        return Route("https://offline.test", {}, "m")
    r = TrialRelay(TrialIdentity("trusted-trial", tmp_path),
                   RelayConfig(declared_models=("m",), record="usage", token_budget=settings(close_grace_seconds=.01)),
                   route, _Sink(), _CONNECTION)
    ready = asyncio.Event()
    async def handler():
        a = await r.budget.admit(r.budget.prepare("anthropic", "m", body()), "slow")
        ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            r.budget.settle(a, mod.Measurement())
    worker = asyncio.create_task(handler())
    r._proxy_stop = asyncio.Event()
    r._proxy_task = asyncio.create_task(r._proxy_stop.wait())
    await ready.wait()
    a, b = await asyncio.gather(r.close(), r.close())
    await asyncio.gather(worker, return_exceptions=True)
    assert a is b and a.budget["complete"]
    assert mod.read_budget(tmp_path)["complete"]


@pytest.mark.asyncio
async def test_attempt_root_changes_and_iteration_work_survive_the_join(tmp_path):
    b = budget(tmp_path)
    for i, system in enumerate(("a" * 200, "b" * 200, "b" * 200)):
        a = await b.admit(b.prepare("anthropic", "m", body(system=system)), str(i))
        facts = b.settle(a, mod.Measurement(n=100, o=1, w=3, terminal=True))
        assert facts["root_changed"] is (i == 1)
        assert facts["iteration_work"] == 3
    await b.close_out()
    joined = mod.read_budget(tmp_path)["attempts"]
    assert [joined[f"{i}:1"]["root_changed"] for i in range(3)] == [False, True, False]
    assert [joined[f"{i}:1"]["iteration_work"] for i in range(3)] == [3, 3, 3]


@pytest.mark.asyncio
async def test_terminal_usage_then_stream_error_is_stage_g(tmp_path):
    from tests.test_budget_relay import invoke
    events = [
        {"type": "message_start", "message": {"id": "id", "type": "message", "role": "assistant", "model": "m", "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}, "stop_reason": None, "stop_sequence": None}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
        {"type": "error", "error": {"type": "invalid_request_error", "message": "late error"}},
    ]
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    b, result, sent, calls = await invoke(tmp_path, body(stream=True), lambda request, number: httpx.Response(200, text=text, headers={"content-type": "text/event-stream"}))
    assert calls[0].budget["attempts"][0]["stage"] == "G"
    assert b.spent == 13


@pytest.mark.asyncio
async def test_settlement_fsync_failure_is_fatal_at_wrapper_phase_end(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from harbor_sandbox_agent_proxy.agent import RelayAgent
    from harbor_sandbox_agent_proxy.config import RelayConfig
    from harbor_sandbox_agent_proxy.environment import RelayEnvironment
    from harbor_sandbox_agent_proxy.outcome import TrialIdentity
    from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
    from tests.test_trial_relay import _CONNECTION, _Sink
    async def route(call):
        return Route("https://offline.test", {}, "m")
    r = TrialRelay(TrialIdentity("trusted-trial", tmp_path),
                   RelayConfig(declared_models=("m",), record="usage", token_budget=settings(unknown_usage="stop")),
                   route, _Sink(), _CONNECTION)
    b = r.budget
    a = await b.admit(b.prepare("anthropic", "m", body()), "partial")
    real_fsync = os.fsync
    def fail_settlement(fd):
        if Path(f"/proc/self/fd/{fd}").resolve() == b.journal_path:
            last = b.journal_path.read_text().splitlines()[-1]
            if json.loads(last)["kind"] == "settle":
                raise OSError("injected settlement fsync failure")
        real_fsync(fd)
    monkeypatch.setattr(os, "fsync", fail_settlement)
    environment = object.__new__(RelayEnvironment)
    environment.relay = r
    wrapper = object.__new__(RelayAgent)
    wrapper._inner = SimpleNamespace(_extra_env={})
    async def phase(*args):
        with pytest.raises(OSError, match="settlement fsync"):
            b.settle(a, mod.Measurement(n=10))
    try:
        with pytest.raises(RuntimeError, match="could not record a model call"):
            await wrapper._relayed_phase(phase, "instruction", environment, None)
        assert isinstance(r.record_error, OSError)
        assert b.latch_cause == "usage_unknown"
        with pytest.raises(mod.BudgetRefused):
            await b.admit(b.prepare("anthropic", "m", body()), "blocked")
    finally:
        b.release_writer()


@pytest.mark.parametrize("case,exit_code", [("complete", 0), ("missing_charge", 1), ("arithmetic_only", 1)])
def test_f5_replay_cli_cannot_certify_missing_or_exploratory_evidence(tmp_path, case, exit_code):
    import subprocess
    import sys
    row = replay_row(1, "root", 100)
    if case == "missing_charge":
        row.pop("committed_charge")
    artifact = tmp_path / "replay.jsonl"
    artifact.write_text(json.dumps(row) + "\n")
    command = [sys.executable, "-m", "tests.budget_replay", "replay", str(artifact)]
    if case == "arithmetic_only":
        command.append("--arithmetic-only")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    verdict = json.loads(result.stdout)
    assert verdict["passed"] is (exit_code == 0)
    assert verdict["charges"] == [100]


@pytest.mark.asyncio
@pytest.mark.parametrize("crossing", [False, True])
async def test_f6_inconsistent_crossing_event_cannot_be_recovered_as_clean(tmp_path, crossing):
    b = budget(tmp_path, limit=300 if crossing else 3000, max_reservation=2000, overshoot="one_call")
    a = await b.admit(b.prepare("anthropic", "m", body("x" * 400, max_tokens=100)), "a")
    b.settle(a, measured(400, 100))
    await b.close_out()
    events = [json.loads(line) for line in b.journal_path.read_text().splitlines()]
    for event in events:
        if event["kind"] == "admit":
            event["crossing_allowance"] = 0 if crossing else event["R"]
            event["h"] = mod.event_hash(event)
    b.journal_path.write_bytes(b"".join(mod.serialize(event) + b"\n" for event in events))
    result = mod.read_budget(tmp_path)
    assert not result["complete"] and result["incomplete_reason"] == "crossing_allowance"


@pytest.mark.asyncio
async def test_f8_a_real_older_prefix_branch_is_still_a_fork(tmp_path):
    b = budget(tmp_path)
    first = {"role": "user", "content": "a" * 200}
    for i, messages in enumerate(([first], [first, {"role": "assistant", "content": "b" * 200}],
                                  [first, {"role": "assistant", "content": "c" * 200}])):
        a = await b.admit(b.prepare("anthropic", "m", body(messages=messages)), str(i))
        b.settle(a, measured(100 * (i + 1), 1))
    result = await b.close_out()
    assert result["spent"] == 403 and result["forks"] == 1
    assert result["carried_context_approximate"]
