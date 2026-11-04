"""Delta-review regressions for cancellation-safe close and measured empty prefixes."""
from __future__ import annotations

import asyncio
import json

import pytest

from harbor_sandbox_agent_proxy import budget as mod
from tests.budget_replay import replay
from tests.test_budget import body, budget, measured, settings


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["budget", "relay"])
@pytest.mark.parametrize("mode", ["cancel_first", "cancel_all_then_retry", "forced_after_first_cancel"])
async def test_d1_caller_cancellation_cannot_own_shutdown_lifetime(tmp_path, layer, mode):
    proxy_stop = proxy_task = None
    if layer == "relay":
        from harbor_sandbox_agent_proxy.config import RelayConfig
        from harbor_sandbox_agent_proxy.outcome import TrialIdentity
        from harbor_sandbox_agent_proxy.routing import Route
        from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
        from tests.test_trial_relay import _CONNECTION, _Sink
        async def route(call):
            return Route("https://offline.test", {}, "m")
        r = TrialRelay(TrialIdentity("trusted-trial", tmp_path),
                       RelayConfig(declared_models=("m",), record="usage", token_budget=settings(close_grace_seconds=.02 if mode.startswith("forced") else 1)),
                       route, _Sink(), _CONNECTION)
        b, close = r.budget, r.close
        proxy_stop = asyncio.Event()
        proxy_task = asyncio.create_task(proxy_stop.wait())
        r._proxy_stop, r._proxy_task = proxy_stop, proxy_task
    else:
        b = budget(tmp_path, close_grace_seconds=.02 if mode.startswith("forced") else 1)
        close = b.close_out
    ready, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    append = b._append
    def observe(kind, **fields):
        append(kind, **fields)
        if kind == "close_begin":
            closing.set()
    b._append = observe
    async def handler():
        a = await b.admit(b.prepare("anthropic", "m", body()), "handler")
        ready.set()
        try:
            await release.wait()
        finally:
            b.settle(a, measured(10, 1))
    worker = asyncio.create_task(handler())
    first = other = None
    try:
        await ready.wait()
        first = asyncio.create_task(close())
        await closing.wait()
        other = asyncio.create_task(close())
        await asyncio.sleep(0)
        first.cancel()
        if mode == "cancel_all_then_retry":
            other.cancel()
        cancelled = await asyncio.gather(first, return_exceptions=True)
        assert isinstance(cancelled[0], asyncio.CancelledError)
        if mode != "forced_after_first_cancel":
            release.set()
        if mode == "cancel_all_then_retry":
            await asyncio.gather(other, return_exceptions=True)
            result = await close()
        else:
            result = await other
        handler_result = (await asyncio.gather(worker, return_exceptions=True))[0]
        if mode == "forced_after_first_cancel":
            assert isinstance(handler_result, asyncio.CancelledError)
        else:
            assert handler_result is None
        record = result.budget if layer == "relay" else result
        assert record["complete"] and b._lock.closed and b.lifecycle == "closed"
        assert await close() is result
        assert mod.read_budget(tmp_path)["complete"]
        kinds = [json.loads(line)["kind"] for line in b.journal_path.read_text().splitlines()]
        assert kinds.count("close_begin") == kinds.count("close") == 1
        if proxy_stop is not None:
            assert proxy_stop.is_set() and proxy_task.done()
    finally:
        release.set()
        await asyncio.gather(worker, return_exceptions=True)
        for task in (first, other):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if proxy_stop is not None:
            proxy_stop.set()
            await proxy_task
        b.release_writer()


@pytest.mark.asyncio
async def test_d1_real_close_failure_is_shared_without_restarting_journal(tmp_path):
    b = budget(tmp_path)
    append = b._append
    failure = OSError("injected close write failure")
    def fail(kind, **fields):
        if kind == "close":
            raise failure
        append(kind, **fields)
    b._append = fail
    results = await asyncio.gather(b.close_out(), b.close_out(), return_exceptions=True)
    assert results[0] is results[1] is failure
    with pytest.raises(OSError) as later:
        await b.close_out()
    assert later.value is failure and b._lock.closed
    kinds = [json.loads(line)["kind"] for line in b.journal_path.read_text().splitlines()]
    assert kinds.count("close_begin") == 1 and "close" not in kinds
    assert not mod.read_budget(tmp_path)["complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("credit,expected", [("eligible", [101, 1, 11]), ("none", [101, 101, 111])])
async def test_d2_root_only_reroll_and_first_message_join_the_oracle(tmp_path, credit, expected):
    cfg = settings()
    cfg.profiles["m"].credit = credit
    b = mod.TrialBudget(cfg, tmp_path, "trusted-trial")
    rows, charges = [], []
    first_message = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "question"}]}
    for i, messages in enumerate(([], [], [first_message])):
        request = {"model": "m", "instructions": "system " + "x" * 200, "input": messages, "max_output_tokens": 10}
        plan = b.prepare("openai-responses", "m", request)
        # Admission still reserves the entire final body, independent of prefix credit.
        assert plan.r_in == len(plan.wire) and plan.r_out == 10
        a = await b.admit(plan, str(i))
        n = 110 if messages else 100
        facts = b.settle(a, measured(n, 1))
        charges.append(facts["charge"])
        rows.append(dict(admit_seq=i + 1, model="m", dialect="openai-responses", profile=cfg.profiles["m"].id,
                         credit=credit, root={"model": "m", "instructions": request["instructions"]},
                         messages=messages, n=n, o=1, committed_charge=facts["charge"]))
    await b.close_out()
    assert charges == expected
    result = replay(rows)
    assert result["passed"] and result["charges"] == expected
    assert b.record()["forks"] == 0


@pytest.mark.asyncio
async def test_d2_empty_prefix_credit_never_crosses_roots(tmp_path):
    b = budget(tmp_path)
    charges = []
    for i, instructions in enumerate(("a" * 200, "b" * 200)):
        request = {"model": "m", "instructions": instructions, "input": [], "max_output_tokens": 10}
        a = await b.admit(b.prepare("openai-responses", "m", request), str(i))
        charges.append(b.settle(a, measured(100, 1))["charge"])
    assert charges == [101, 101]
    await b.close_out()


@pytest.mark.asyncio
async def test_reviewer_d1_budget_owner_probe_with_forced_cancellation_cleanup(tmp_path):
    b = budget(tmp_path, close_grace_seconds=.1)
    admitted, release = asyncio.Event(), asyncio.Event()
    async def handler():
        a = await b.admit(b.prepare("anthropic", "m", body()), "handler")
        admitted.set()
        try:
            await release.wait()
        finally:
            b.settle(a, measured(10, 1))
    worker = asyncio.create_task(handler())
    await admitted.wait()
    owner = asyncio.create_task(b.close_out())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(b.close_out())
    await asyncio.sleep(0)
    owner.cancel()
    owner_result, waiter_result = await asyncio.gather(owner, waiter, return_exceptions=True)
    release.set()
    worker_result = (await asyncio.gather(worker, return_exceptions=True))[0]
    assert worker_result is None or isinstance(worker_result, asyncio.CancelledError)
    retry = await b.close_out()
    recovered = mod.read_budget(tmp_path)
    assert isinstance(owner_result, asyncio.CancelledError)
    assert isinstance(waiter_result, dict) and waiter_result["complete"]
    assert retry is waiter_result and recovered["complete"]
    assert b._lock.closed and b.lifecycle == "closed"


@pytest.mark.asyncio
async def test_reviewer_d1_trial_owner_probe_with_retained_stop_event(tmp_path):
    from harbor_sandbox_agent_proxy.config import RelayConfig
    from harbor_sandbox_agent_proxy.outcome import TrialIdentity
    from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
    from harbor_sandbox_agent_proxy.routing import Route
    from tests.test_trial_relay import _CONNECTION, _Sink
    async def route(call):
        return Route("https://offline.test", {}, "m")
    r = TrialRelay(TrialIdentity("trusted-trial", tmp_path),
                   RelayConfig(declared_models=("m",), record="usage", token_budget=settings(close_grace_seconds=.1)),
                   route, _Sink(), _CONNECTION)
    admitted, release = asyncio.Event(), asyncio.Event()
    async def handler():
        a = await r.budget.admit(r.budget.prepare("anthropic", "m", body()), "handler")
        admitted.set()
        await release.wait()
        r.budget.settle(a, measured(10, 1))
    worker = asyncio.create_task(handler())
    proxy_stop = asyncio.Event()
    proxy_task = asyncio.create_task(proxy_stop.wait())
    r._proxy_stop, r._proxy_task = proxy_stop, proxy_task
    await admitted.wait()
    owner = asyncio.create_task(r.close())
    await asyncio.sleep(0)
    other = asyncio.create_task(r.close())
    await asyncio.sleep(0)
    owner.cancel()
    results = await asyncio.gather(owner, other, return_exceptions=True)
    release.set()
    worker_result = (await asyncio.gather(worker, return_exceptions=True))[0]
    assert worker_result is None or isinstance(worker_result, asyncio.CancelledError)
    retry = await r.close()
    assert isinstance(results[0], asyncio.CancelledError)
    assert not isinstance(results[1], BaseException) and results[1].budget["complete"]
    assert retry is results[1]
    assert proxy_stop.is_set() and proxy_task.done()
    assert r.budget._lock.closed and r.budget.lifecycle == "closed"
    assert mod.read_budget(tmp_path)["complete"]
