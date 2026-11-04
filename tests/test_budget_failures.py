"""Concrete design 35–37 boundary controls, not repeated success-case labels."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from harbor_sandbox_agent_proxy import budget as mod, relay
from harbor_sandbox_agent_proxy.routing import Route
from tests.test_budget import body, budget, measured


def handlers(b, responder, *, recorder=None, router=None):
    sent, calls = [], []
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        calls.append(call)
    async def handle(request):
        sent.append(request)
        value = responder(request, len(sent))
        return await value if hasattr(value, "__await__") else value
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    methods = relay.relay_methods(
        lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
        router or route, record=recorder or record, client=client, budget=b,
    )
    return client, methods, sent, calls


@pytest.mark.asyncio
async def test_35_retry_does_not_reuse_terminal_usage(tmp_path, monkeypatch):
    b = budget(tmp_path)
    monkeypatch.setattr(relay, "_BASE_BACKOFF_SEC", 0)
    monkeypatch.setattr(relay, "_MAX_BACKOFF_SEC", 0)
    def respond(request, number):
        return httpx.Response(500, json={"usage": {"input_tokens": 10, "output_tokens": 2}}) if number == 1 else httpx.Response(200, json={})
    client, methods, sent, calls = handlers(b, respond)
    async with client:
        await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
    facts = calls[0].budget["attempts"]
    assert len(sent) == len(facts) == 2
    assert facts[0]["charge"] == 12
    assert facts[1]["n"] is None and facts[1]["o"] is None
    assert facts[1]["charge_kind"] == "bound"
    assert b.spent == 12 + len(sent[1].content) + 10
    await b.close_out()


@pytest.mark.asyncio
async def test_35_backoff_cancellation_never_admits_another_attempt(tmp_path, monkeypatch):
    b = budget(tmp_path)
    backoff = asyncio.Event()
    async def pause(seconds):
        backoff.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(relay.anyio, "sleep", pause)
    client, methods, sent, calls = handlers(b, lambda request, number: httpx.Response(500, json={"usage": {"input_tokens": 10, "output_tokens": 2}}))
    async with client:
        task = asyncio.create_task(methods["generate_anthropic"](body(), {"x-api-key": "caller"}))
        await backoff.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(sent) == 1 and b.spent == 12 and b.held == 0
    kinds = [json.loads(line)["kind"] for line in b.journal_path.read_text().splitlines()]
    assert kinds.count("admit") == kinds.count("settle") == kinds.count("commit") == 1
    await b.close_out()


@pytest.mark.asyncio
async def test_36_cancellation_before_route_completion_has_no_hold(tmp_path):
    b = budget(tmp_path)
    routing = asyncio.Event()
    async def route(call):
        routing.set()
        await asyncio.Event().wait()
    client, methods, sent, calls = handlers(b, lambda request, number: httpx.Response(200, json={}), router=route)
    async with client:
        task = asyncio.create_task(methods["generate_anthropic"](body(), {"x-api-key": "caller"}))
        await routing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert sent == [] and b.spent == b.held == 0
    assert [json.loads(line)["kind"] for line in b.journal_path.read_text().splitlines()] == ["open"]
    await b.close_out()


@pytest.mark.asyncio
async def test_36_cancelled_waiter_is_removed_without_an_admission(tmp_path):
    b = budget(tmp_path, limit=1000, overshoot="one_call")
    p = b.prepare("anthropic", "m", body("x" * 800))
    first = await b.admit(p, "first")
    waiting = asyncio.create_task(b.admit(p, "waiting"))
    await asyncio.sleep(0)
    assert not waiting.done()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not b._waiters and b.held == p.reserved
    b.settle(first, measured(100, 1))
    await b.close_out()
    assert len(mod.read_budget(tmp_path)["attempts"]) == 1


@pytest.mark.asyncio
async def test_36_cancellation_after_send_settles_bound_once(tmp_path):
    b = budget(tmp_path)
    sending = asyncio.Event()
    async def respond(request, number):
        sending.set()
        await asyncio.Event().wait()
    client, methods, sent, calls = handlers(b, respond)
    async with client:
        task = asyncio.create_task(methods["generate_anthropic"](body(), {"x-api-key": "caller"}))
        await sending.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(sent) == 1 and b.held == 0
    assert b.spent == len(sent[0].content) + 10
    await b.close_out()
    record = mod.read_budget(tmp_path)
    assert record["complete"] and len(record["attempts"]) == 1
    assert next(iter(record["attempts"].values()))["stage"] == "F"


@pytest.mark.asyncio
async def test_37_recorder_failure_cannot_lose_committed_spend(tmp_path):
    b = budget(tmp_path)
    async def fail(call):
        raise OSError("injected recorder failure")
    client, methods, sent, calls = handlers(b, lambda request, number: httpx.Response(200, json={"usage": {"input_tokens": 10, "output_tokens": 2}}), recorder=fail)
    async with client:
        with pytest.raises(OSError, match="recorder failure"):
            await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
    assert b.spent == 12 and b.held == 0
    assert mod.read_budget(tmp_path)["spent"] == 12
    await b.close_out()
