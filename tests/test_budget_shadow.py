"""A disabled enforcement policy observes; it must not change baseline traffic or records."""
from __future__ import annotations

import asyncio
import copy
import datetime
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import inspect_ai.event._base as event_base
import inspect_ai.model._chat_message as chat_module
import inspect_ai._util.working as working_clock
import pytest

from harbor_sandbox_agent_proxy import budget as budget_module, relay, trial_relay
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.connection import AgentConnection
from harbor_sandbox_agent_proxy.outcome import JsonOutcomeSink, TrialIdentity
from harbor_sandbox_agent_proxy.routing import Route
from tests.test_budget import body, measured, settings

NOW = datetime.datetime(2026, 11, 4, tzinfo=datetime.UTC)
MESSAGE = {"id": "response-id", "type": "message", "role": "assistant", "model": "m",
           "content": [{"type": "text", "text": "answer"}], "stop_reason": "end_turn",
           "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 2}}


class FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture
def fixed_recording_ids(monkeypatch):
    clock = SimpleNamespace(datetime=FixedDatetime, UTC=datetime.UTC)
    monkeypatch.setattr(relay, "datetime", clock)
    monkeypatch.setattr(trial_relay, "datetime", clock)
    monkeypatch.setattr(relay, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(working_clock, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(event_base, "uuid", lambda: "event-id")
    monkeypatch.setattr(chat_module, "uuid", lambda: "message-id")
    monkeypatch.setattr(trial_relay, "uuid4", lambda: uuid.UUID(int=1))
    monkeypatch.setattr(trial_relay, "mint_caller_key", lambda: "verifier")
    monkeypatch.setattr(relay, "_retry_delay", lambda *_: 0)


def configuration(case):
    changes = dict(enforce=False, close_grace_seconds=0)
    if case == "profile":
        changes.update(output_ceiling={}, profiles={})
    if case in ("limit", "journal_latch"):
        changes.update(limit=12, max_reservation=1000000, overshoot="one_call")
    if case == "insufficient":
        changes.update(limit=1, max_reservation=1000000, overshoot="none")
    if case == "wait":
        changes.update(limit=1000, max_reservation=1000000, overshoot="none")
    if case == "reservation":
        changes["max_reservation"] = 1
    if case == "injection":
        changes["inject_output_limit"] = True
    cfg = settings(**changes)
    if case == "configuration":
        cfg.profiles["m"].outside_output_thinking = True
    if case.startswith("invalid_policy_"):
        proposal = cfg.model_dump(mode="json")
        if case == "invalid_policy_number":
            proposal["limit"] = 0
        elif case == "invalid_policy_profile":
            proposal["profiles"]["m"]["credit_evidence"] = None
        else:
            proposal["rule"] = "unknown-rule"
        return proposal
    return cfg


def requests_for(case):
    if case == "surface":
        return [body(future_envelope={"private_payload": "not-for-the-shadow-log"})]
    if case == "schema":
        return [body(tools=[{"name": "client", "input_schema": {"type": "object", "description": "s" * 65536}}])]
    if case == "number":
        return [body(max_tokens=100001)]
    if case in ("missing_output", "injection"):
        request = body()
        request.pop("max_tokens")
        return [request]
    if case == "thinking":
        return [body(max_tokens=2048, thinking={"type": "enabled", "budget_tokens": 1024})]
    if case == "wait":
        return [body("x" * 600)]
    if case in ("stream", "stream_error", "meter_feed"):
        return [body(stream=True)]
    return [body(), body()] if case in ("limit", "bound") or case.startswith(("journal_", "snapshot_")) else [body()]


def inject_budget_failure(monkeypatch, case):
    if case == "meter":
        from harbor_sandbox_agent_proxy.budget_shadow import ShadowMeter
        def broken_meter(*args, **kwargs):
            raise RuntimeError("injected shadow meter failure with private text")
        monkeypatch.setattr(ShadowMeter, "__init__", broken_meter)
    if case in ("meter_observe", "meter_feed", "meter_measurement"):
        def broken_measurement(*args, **kwargs):
            raise RuntimeError("injected observation failure with private text")
        monkeypatch.setattr(budget_module.UsageMeter, case.removeprefix("meter_"), broken_measurement)
    if case.startswith("journal_"):
        kind = case.removeprefix("journal_")
        original = budget_module.TrialBudget._append
        def append(self, event_kind, **fields):
            if not self.config.enforce and event_kind == kind:
                raise OSError("injected budget journal failure with private text")
            return original(self, event_kind, **fields)
        monkeypatch.setattr(budget_module.TrialBudget, "_append", append)
    if case.startswith("snapshot_"):
        target = {"open": "open", "admit": "admit", "settle": "commit", "close_begin": "close_begin", "close": "close"}[case.removeprefix("snapshot_")]
        original = budget_module.TrialBudget._snapshot
        def snapshot(self):
            last = json.loads(self.journal_path.read_text().splitlines()[-1])
            if not self.config.enforce and last["kind"] == target:
                raise OSError("injected budget snapshot failure with private text")
            return original(self)
        monkeypatch.setattr(budget_module.TrialBudget, "_snapshot", snapshot)


async def run_trial(root, mode, case, enabled):
    trial = TrialIdentity("shadow-identity", root)
    sink = JsonOutcomeSink(trial, protected=lambda: ("host-secret", "caller", "verifier"))
    cfg = RelayConfig(declared_models=("m",), record=mode, max_recorded_bytes=5000000,
                      token_budget=configuration(case) if enabled else None)
    async def route(call):
        return Route("https://offline.test", {"x-api-key": "host-secret", "x-bet-eval-call-id": "fixed-host-call"}, "m")
    r = trial_relay.TrialRelay(trial=trial, config=cfg, route=route, sink=sink,
        connection=AgentConnection("anthropic", ("ANTHROPIC_API_KEY",), ("ANTHROPIC_BASE_URL",)),
        agent_key="caller", verifier_key="verifier")
    observer = getattr(r, "shadow", None)
    if observer is not None:
        await asyncio.wrap_future(observer._ready)
    ledger = observer.ledger if observer is not None else r.budget
    if enabled and ledger is not None:
        if case == "closed":
            await ledger.close_out()
        elif case == "wait":
            seed = await ledger.admit(ledger.prepare("anthropic", "m", body("x" * 600)), "pending")
            seed.task = None
        elif case == "usage_unknown":
            ledger.config.unknown_usage = "stop"
            a = await ledger.admit(ledger.prepare("anthropic", "m", body()), "partial")
            ledger.settle(a, budget_module.Measurement(n=10))
        elif case == "invalid_reservation":
            object.__setattr__(ledger.config.profiles["m"], "route_injected_tokens_max", -1000000)
            if observer is not None:
                object.__setattr__(observer.config.profiles["m"], "route_injected_tokens_max", -1000000)
    sent, responses = [], []
    def respond(request):
        sent.append({"url": str(request.url), "body": request.content, "headers": list(request.headers.raw)})
        response = copy.deepcopy(MESSAGE)
        if case == "bound":
            response["usage"]["input_tokens"] = 9000
        if json.loads(request.content).get("stream") is True:
            events = [
                {"type": "message_start", "message": {**response, "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}},
                {"type": "message_stop"},
            ]
            if case == "stream_error":
                events.append({"type": "error", "error": {"type": "invalid_request_error", "message": "late error"}})
            text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=response)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        kwargs = {"shadow": observer} if observer is not None else {}
        methods = relay.relay_methods(r.current_policy, route, record=r.record, client=client,
                                      budget=r.budget, usage_only=mode == "usage", **kwargs)
        for request in requests_for(case):
            responses.append(await methods["generate_anthropic"](copy.deepcopy(request), {"x-api-key": "caller"}))
        await r.end_agent_phase()
        responses.append(await methods["generate_anthropic"](body("check"), {"x-api-key": r.verifier_key}))
    outcome = await r.close()
    await sink.close(outcome)
    files = {"outcome": (root / "sandbox-agent-proxy" / "outcome.json").read_bytes()}
    usage_path = root / "sandbox-agent-proxy" / "usage.jsonl"
    if usage_path.exists():
        files["usage"] = usage_path.read_bytes()
    return sent, responses, files, r


REFUSALS = {
    "valid": None, "surface": "unsupported_input", "profile": "unsupported_model_profile",
    "schema": "unsupported_input", "number": "invalid_request_value",
    "missing_output": "output_limit_required", "thinking": "unbounded_thinking",
    "reservation": "reservation_too_large", "invalid_reservation": "invalid_reservation",
    "closed": "budget_closed", "limit": "token_budget", "insufficient": "token_budget_insufficient",
    "wait": "token_budget_wait", "usage_unknown": "token_budget", "bound": "bound_violated",
    "injection": None, "meter": "shadow_error", "configuration": "invalid_budget_config",
    "stream": None, "stream_error": None,
    "meter_observe": "shadow_error", "meter_feed": "shadow_error", "meter_measurement": "shadow_error",
    "invalid_policy_number": "invalid_budget_config", "invalid_policy_profile": "invalid_budget_config",
    "invalid_policy_rule": "invalid_budget_config",
}
WRITES = ["journal_open", "journal_admit", "journal_settle", "journal_commit", "journal_latch",
          "journal_close_begin", "journal_close", "snapshot_open", "snapshot_admit", "snapshot_settle",
          "snapshot_close_begin", "snapshot_close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["usage", "full"])
@pytest.mark.parametrize("case", [*REFUSALS, *WRITES])
async def test_shadow_matches_no_budget_bytes_for_every_refusal_and_write_failure(tmp_path, monkeypatch, fixed_recording_ids, mode, case):
    inject_budget_failure(monkeypatch, case)
    baseline = await run_trial(tmp_path / "baseline", mode, case, False)
    observed = await run_trial(tmp_path / "shadow", mode, case, True)
    assert observed[:3] == baseline[:3]
    assert observed[3].record_error is None
    if case in WRITES or REFUSALS[case] is not None:
        path = tmp_path / "shadow" / "sandbox-agent-proxy" / "budget-shadow.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        expected = "persistence_failed" if case in WRITES else REFUSALS[case]
        assert any(e["kind"] == "would_refuse" and e["reason"] == expected for e in events)
        assert "not-for-the-shadow-log" not in path.read_text()
        assert "with private text" not in path.read_text()
        assert "host-secret" not in path.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["usage", "full"])
@pytest.mark.parametrize("enforce", [False, True])
async def test_shadow_exempts_accounting_but_enforcing_counts_it(tmp_path, mode, enforce):
    cfg_budget = settings(enforce=enforce)
    cfg_budget.profiles["m"].bound_source = "source" * 10000
    trial = TrialIdentity("recording-cap", tmp_path)
    sink = JsonOutcomeSink(trial, protected=lambda: ())
    cfg = RelayConfig(declared_models=("m",), record=mode, max_recorded_bytes=20000, token_budget=cfg_budget)
    async def route(call):
        return Route("https://offline.test", {}, "m")
    r = trial_relay.TrialRelay(trial, cfg, route, sink,
        AgentConnection("anthropic", ("ANTHROPIC_API_KEY",), ("ANTHROPIC_BASE_URL",)), agent_key="caller")
    sent = []
    def respond(request):
        sent.append(request)
        return httpx.Response(200, json=MESSAGE)
    if r.shadow is not None:
        await asyncio.wrap_future(r.shadow._ready)
    ledger = r.budget if enforce else r.shadow.ledger
    assert ledger.journal_path.stat().st_size + ledger.snapshot_path.stat().st_size > 20000
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        methods = relay.relay_methods(r.current_policy, route, record=r.record, client=client,
                                      budget=r.budget, shadow=r.shadow, usage_only=mode == "usage")
        for _ in range(2):
            result = await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
            if enforce:
                assert "recorded-byte cap" in str(result)
            else:
                assert result == MESSAGE
    assert len(sent) == (0 if enforce else 2)
    await r.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["usage", "full"])
async def test_shadow_log_failure_keeps_baseline_records_and_reports_reason(tmp_path, monkeypatch, fixed_recording_ids, caplog, mode):
    inject_budget_failure(monkeypatch, "journal_open")
    original = Path.open
    def fail_log(self, *args, **kwargs):
        if self.name == "budget-shadow.jsonl":
            raise OSError("injected shadow log failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_log)
    baseline = await run_trial(tmp_path / "baseline", mode, "journal_open", False)
    observed = await run_trial(tmp_path / "shadow", mode, "journal_open", True)
    assert observed[:3] == baseline[:3]
    assert observed[3].record_error is None
    assert any("reason=persistence_failed" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_shadow_measurement_does_not_label_a_stream_error_successful(tmp_path, fixed_recording_ids):
    await run_trial(tmp_path, "usage", "stream_error", True)
    events = [json.loads(line) for line in (tmp_path / "sandbox-agent-proxy" / "budget-shadow.jsonl").read_text().splitlines()]
    usage = next(event for event in events if event["kind"] == "usage")
    assert usage["n"] == 10 and usage["o"] == 2 and usage["terminal"]
    assert usage["successful"] is False


@pytest.mark.asyncio
async def test_shadow_call_id_stays_stable_across_preflight_and_response(tmp_path, fixed_recording_ids):
    await run_trial(tmp_path, "usage", "surface", True)
    events = [json.loads(line) for line in (tmp_path / "sandbox-agent-proxy" / "budget-shadow.jsonl").read_text().splitlines()]
    warning = next(event for event in events if event["kind"] == "would_refuse")
    usage = next(event for event in events if event["kind"] == "usage")
    assert warning["call_id"] == usage["call_id"]
    recorded = json.loads((tmp_path / "sandbox-agent-proxy" / "usage.jsonl").read_text().splitlines()[0])
    assert recorded["call_id"] == "fixed-host-call" and recorded["call_id_source"] == "header"


@pytest.mark.asyncio
async def test_enforcing_and_shadow_owners_cannot_be_combined(tmp_path):
    from harbor_sandbox_agent_proxy.budget_shadow import ShadowBudget
    enforcing = budget_module.TrialBudget(settings(), tmp_path / "enforcing", "trial")
    observer = ShadowBudget(settings(enforce=False), tmp_path / "shadow", "trial")
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        pass
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=MESSAGE))) as client:
            with pytest.raises(ValueError, match="one budget mode"):
                relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
                                    route, record=record, client=client, budget=enforcing, shadow=observer)
    finally:
        await enforcing.close_out()
        await observer.close()


@pytest.mark.parametrize("case", ["invalid_policy_number", "invalid_policy_profile", "invalid_policy_rule"])
def test_invalid_enforcing_policies_still_fail_configuration(case):
    proposal = configuration(case)
    proposal["enforce"] = True
    with pytest.raises(ValueError):
        RelayConfig(declared_models=("m",), token_budget=proposal)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["usage", "full"])
async def test_shadow_does_not_inject_chat_output_or_usage_fields(tmp_path, fixed_recording_ids, mode):
    response = {"id": "chat-response", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    async def run(root, enabled):
        identity = TrialIdentity("chat-shadow", root)
        sink = JsonOutcomeSink(identity, protected=lambda: ("host-secret",))
        cfg = RelayConfig(declared_models=("m",), record=mode,
                          token_budget=settings(enforce=False, inject_output_limit=True) if enabled else None)
        async def route(call):
            return Route("https://offline.test", {"authorization": "Bearer host-secret", "x-bet-eval-call-id": "fixed-host-call"}, "m")
        r = trial_relay.TrialRelay(identity, cfg, route, sink,
            AgentConnection("openai", ("OPENAI_API_KEY",), ("OPENAI_BASE_URL",)), agent_key="caller")
        sent = []
        def respond(request):
            sent.append((request.content, list(request.headers.raw)))
            return httpx.Response(200, json=response)
        request = {"model": "m", "messages": [{"role": "user", "content": "hello"}],
                   "stream": True, "stream_options": {"include_usage": False}, "parallel_tool_calls": False}
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            methods = relay.relay_methods(r.current_policy, route, record=r.record, client=client,
                                          budget=r.budget, shadow=r.shadow, usage_only=mode == "usage")
            result = await methods["generate_completions"](request, {"authorization": "Bearer caller"})
        outcome = await r.close()
        await sink.close(outcome)
        files = [(root / "sandbox-agent-proxy" / "outcome.json").read_bytes()]
        if mode == "usage":
            files.append((root / "sandbox-agent-proxy" / "usage.jsonl").read_bytes())
        actual = json.loads(sent[0][0])
        assert "max_tokens" not in actual and "max_completion_tokens" not in actual
        assert actual["stream_options"] == {"include_usage": False}
        assert "parallel_tool_calls" not in actual  # baseline confinement still applies
        return sent, result, files
    assert await run(tmp_path / "shadow", True) == await run(tmp_path / "baseline", False)
