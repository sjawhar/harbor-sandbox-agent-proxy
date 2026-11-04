"""Storage must not be on the shadow sender's delivery/completion dependency."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

from harbor_sandbox_agent_proxy import relay
from harbor_sandbox_agent_proxy.budget_shadow import ShadowBudget
from harbor_sandbox_agent_proxy.routing import Route
from tests.test_budget import body, settings
from tests.test_budget_shadow import MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["open", "admit", "settle", "snapshot", "diagnostic"])
async def test_shadow_blocked_storage_never_delays_delivery_response_or_heartbeat(tmp_path, monkeypatch, stage):
    # Adapted from the independent section-13 fsync probe. No clocks are frozen.
    entered, release, upstream, returned = (threading.Event() for _ in range(4))
    heartbeat = asyncio.Event()
    observed = {}
    loop = asyncio.get_running_loop()
    journal = tmp_path / "budget.jsonl"
    real_fsync, real_open = os.fsync, Path.open

    def stall():
        entered.set()
        if not release.wait(3):
            raise OSError("bounded test storage stall")

    def fsync(fd):
        path = Path(f"/proc/self/fd/{fd}").resolve()
        if path == journal and stage in ("open", "admit", "settle"):
            last = json.loads(journal.read_text().splitlines()[-1])
            if last["kind"] == stage:
                stall()
        if path == tmp_path / "budget.json.tmp" and stage == "snapshot":
            stall()
        return real_fsync(fd)

    def open_file(path, *args, **kwargs):
        if path.name == "budget-shadow.jsonl" and stage == "diagnostic":
            stall()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(Path, "open", open_file)

    def inspector():
        observed["entered"] = entered.wait(2)
        if observed["entered"]:
            loop.call_soon_threadsafe(heartbeat.set)
            # Wait in the independent thread, never in the forwarder's loop.
            time.sleep(0.1)
            observed["delivery"] = upstream.is_set()
            observed["return"] = returned.is_set()
            observed["heartbeat"] = heartbeat.is_set()
        release.set()

    thread = threading.Thread(target=inspector)
    thread.start()
    observer = ShadowBudget(settings(enforce=False), tmp_path, "storage-timing")
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        pass
    def respond(request):
        upstream.set()
        return httpx.Response(200, json=MESSAGE)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            methods = relay.relay_methods(
                lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
                route, record=record, client=client, shadow=observer,
            )
            result = await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
            returned.set()
            assert result == MESSAGE
            await asyncio.to_thread(thread.join)
    finally:
        release.set()
        await asyncio.to_thread(thread.join)
        await observer.close()
    assert observed == {"entered": True, "delivery": True, "return": True, "heartbeat": True}


@pytest.mark.asyncio
async def test_shadow_overflow_drops_observations_not_requests_and_reports_loss(tmp_path, monkeypatch):
    from harbor_sandbox_agent_proxy import budget_shadow
    monkeypatch.setattr(budget_shadow, "_QUEUE_CAPACITY", 2)
    observer = ShadowBudget(settings(enforce=False), tmp_path, "overflow")
    await asyncio.wrap_future(observer._ready)
    entered, release = threading.Event(), threading.Event()
    original = Path.open
    def open_file(path, *args, **kwargs):
        if path == observer.path:
            entered.set()
            if not release.wait(3):
                raise OSError("bounded queue test stall")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    observer.log("marker", "blocked")
    assert await asyncio.to_thread(entered.wait, 2)
    sent = []
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        pass
    def respond(request):
        sent.append(request.content)
        return httpx.Response(200, json=MESSAGE)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            methods = relay.relay_methods(
                lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
                route, record=record, client=client, shadow=observer,
            )
            for _ in range(3):
                assert await methods["generate_anthropic"](body(), {"x-api-key": "caller"}) == MESSAGE
        assert len(sent) == 3
        assert observer.dropped_observations == 4
        assert observer.observation_status()["complete"] is False
    finally:
        release.set()
        await observer.close()
    events = [json.loads(line) for line in observer.path.read_text().splitlines()]
    status = events[-1]
    assert status["kind"] == "observation_status"
    assert status["dropped_observations"] == 4 and status["complete"] is False
    assert observer.ledger.spent == 12


@pytest.mark.asyncio
async def test_shadow_stuck_writer_cannot_hold_trial_shutdown_open(tmp_path, monkeypatch):
    from harbor_sandbox_agent_proxy import budget_shadow
    monkeypatch.setattr(budget_shadow, "_CLOSE_TIMEOUT", .05)
    observer = ShadowBudget(settings(enforce=False), tmp_path, "bounded-close")
    await asyncio.wrap_future(observer._ready)
    entered, release = threading.Event(), threading.Event()
    original = Path.open
    def open_file(path, *args, **kwargs):
        if path == observer.path:
            entered.set()
            if not release.wait(3):
                raise OSError("bounded shutdown test stall")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    observer.log("marker", "blocked")
    assert await asyncio.to_thread(entered.wait, 2)
    try:
        await asyncio.wait_for(observer.close(), .5)
        assert not release.is_set() and observer._writer.is_alive()
        assert observer.observation_status()["complete"] is False
        assert observer.observation_status()["close_timed_out"] is True
    finally:
        release.set()
        await asyncio.wait_for(asyncio.wrap_future(observer._done), 2)
    assert observer.ledger._lock.closed
    status = json.loads(observer.path.read_text().splitlines()[-1])
    assert status["complete"] is False and status["close_timed_out"] is True


@pytest.mark.asyncio
async def test_shadow_writer_preserves_admission_order_across_out_of_order_responses(tmp_path):
    observer = ShadowBudget(settings(enforce=False), tmp_path, "ordered")
    sent, recorded = asyncio.Event(), []
    release = asyncio.Event()
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        recorded.append(call)
    async def respond(request):
        text = json.loads(request.content)["messages"][0]["content"]
        if text == "first":
            sent.set()
            await release.wait()
        return httpx.Response(200, json=MESSAGE)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            methods = relay.relay_methods(
                lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
                route, record=record, client=client, shadow=observer,
            )
            first = asyncio.create_task(methods["generate_anthropic"](body("first"), {"x-api-key": "caller"}))
            await asyncio.wait_for(sent.wait(), 1)
            assert await methods["generate_anthropic"](body("second"), {"x-api-key": "caller"}) == MESSAGE
            assert not first.done() and len(recorded) == 1
            release.set()
            assert await first == MESSAGE
    finally:
        release.set()
        await observer.close()
    events = [json.loads(line) for line in observer.ledger.journal_path.read_text().splitlines()]
    assert [event["admit_seq"] for event in events if event["kind"] == "admit"] == [1, 2]
    assert [event["admit_seq"] for event in events if event["kind"] == "settle"] == [2, 1]
    assert [event["admit_seq"] for event in events if event["kind"] == "commit"] == [1, 2]
    assert observer.observation_status()["complete"] is True
