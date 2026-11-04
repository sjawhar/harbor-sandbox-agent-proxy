"""Section-13 reviewer controls with async startup and the required cap assertion.

The two shadow probes await initialization before patching the ledger, and join
the inspector off-loop. The cap probe now requires refusal, not the old defect.
"""
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from harbor_sandbox_agent_proxy import relay
from harbor_sandbox_agent_proxy.budget_shadow import ShadowBudget
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.outcome import TrialIdentity
from harbor_sandbox_agent_proxy.routing import Route
from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
from tests.test_budget import body, settings
from tests.test_trial_relay import _CONNECTION, _Sink

MESSAGE = {"id": "id", "type": "message", "role": "assistant", "model": "m", "content": [],
           "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 2}}


@pytest.mark.asyncio
async def test_shadow_journal_stall_does_not_stall_forwarding_or_event_loop(tmp_path, monkeypatch):
    observer = ShadowBudget(settings(enforce=False), tmp_path, "shadow-review")
    await asyncio.wrap_future(observer._ready)
    entered, release, upstream = (threading.Event() for _ in range(3))
    heartbeat = asyncio.Event()
    observed, calls = {}, []
    loop = asyncio.get_running_loop()
    real_fsync = os.fsync
    def fsync(fd):
        if Path(f"/proc/self/fd/{fd}").resolve() == observer.ledger.journal_path:
            last = json.loads(observer.ledger.journal_path.read_text().splitlines()[-1])
            if last["kind"] == "admit":
                entered.set()
                if not release.wait(3):
                    raise OSError("reviewer bounded journal stall")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", fsync)
    def inspector():
        observed["entered"] = entered.wait(2)
        if observed["entered"]:
            loop.call_soon_threadsafe(heartbeat.set)
            time.sleep(.05)
            observed["upstream_before_release"] = upstream.is_set()
            observed["heartbeat_before_release"] = heartbeat.is_set()
        release.set()
    thread = threading.Thread(target=inspector)
    thread.start()
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        calls.append(call)
    def respond(request):
        upstream.set()
        return httpx.Response(200, json=MESSAGE)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            methods = relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
                                          route, record=record, client=client, shadow=observer)
            result = await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
        assert result == MESSAGE
        await asyncio.to_thread(thread.join)
    finally:
        release.set()
        await asyncio.to_thread(thread.join)
        await observer.close()
    print("SHADOW_JOURNAL_STALL", observed)
    assert observed["entered"]
    assert observed["upstream_before_release"], "shadow journaling blocked the real upstream send"
    assert observed["heartbeat_before_release"], "shadow journaling blocked the event loop"


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", "0", "1", [], {}])
def test_ambiguous_mode_is_rejected(value):
    raw = settings().model_dump(mode="json")
    raw["enforce"] = value
    with pytest.raises(ValidationError):
        RelayConfig(declared_models=("m",), token_budget=raw)


def test_missing_mode_and_true_invalid_policy_fail_safe():
    raw = settings().model_dump(mode="json")
    raw.pop("enforce")
    with pytest.raises(ValidationError):
        RelayConfig(declared_models=("m",), token_budget=raw)
    raw["enforce"], raw["limit"] = True, 0
    with pytest.raises(ValidationError):
        RelayConfig(declared_models=("m",), token_budget=raw)
    raw["enforce"] = False
    cfg = RelayConfig(declared_models=("m",), token_budget=raw)
    assert cfg.token_budget.enforce is False


@pytest.mark.asyncio
async def test_shadow_write_exception_keeps_baseline_return(tmp_path, monkeypatch):
    observer = ShadowBudget(settings(enforce=False), tmp_path, "shadow-review")
    await asyncio.wrap_future(observer._ready)
    def broken(*args, **kwargs):
        raise OSError("injected shadow journal failure")
    monkeypatch.setattr(observer.ledger, "_append", broken)
    sent = []
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        pass
    def respond(request):
        sent.append(request.content)
        return httpx.Response(200, json=MESSAGE)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        methods = relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
                                      route, record=record, client=client, shadow=observer)
        result = await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
    await observer.close()
    assert result == MESSAGE and len(sent) == 1 and observer.last_refusal["reason"] == "persistence_failed"


@pytest.mark.asyncio
async def test_enforcing_journal_growth_reaches_recording_cap(tmp_path):
    cap = 100000
    cfg = settings(enforce=True)
    cfg.profiles["m"].credit = "none"
    async def route(call):
        return Route("https://offline.test", {}, "m")
    r = TrialRelay(TrialIdentity("cap-review", tmp_path),
                   RelayConfig(declared_models=("m",), record="usage", max_recorded_bytes=cap, token_budget=cfg),
                   route, _Sink(), _CONNECTION, agent_key="caller")
    sent, refusals = [], []
    def respond(request):
        sent.append(request.content)
        return httpx.Response(200, json=MESSAGE)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            methods = relay.relay_methods(r.current_policy, route, record=r.record, client=client, budget=r.budget, usage_only=True)
            for i in range(40):
                result = await methods["generate_anthropic"](body(system=f"unique-root-{i}"), {"x-api-key": "caller"})
                if result != MESSAGE:
                    assert "recorded-byte cap" in str(result)
                    refusals.append(i)
        assert 0 < len(sent) < 40 and len(sent) + len(refusals) == 40
        before_retry = r.budget.recorded_bytes()
        result = await methods["generate_anthropic"](body(system="again"), {"x-api-key": "caller"})
        assert "recorded-byte cap" in str(result)
        assert r.budget.recorded_bytes() == before_retry
    finally:
        await r.close()
    ordinary = (tmp_path / "sandbox-agent-proxy" / "usage.jsonl").stat().st_size
    journal = r.budget.journal_path.stat().st_size
    snapshot = r.budget.snapshot_path.stat().st_size
    print("ENFORCING_CAP", dict(sends=len(sent), refusals=len(refusals), ordinary=ordinary, journal=journal, snapshot=snapshot, cap=cap))
    assert ordinary < cap <= journal + snapshot + ordinary
    assert r.budget.recorded_bytes() == ordinary + journal + snapshot
