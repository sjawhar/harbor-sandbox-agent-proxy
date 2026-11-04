"""Budgeted real forwarder with an offline HTTP boundary, including retry legs."""
from __future__ import annotations
import asyncio
import copy
import json
import httpx
import pytest
from harbor_sandbox_agent_proxy import relay
from harbor_sandbox_agent_proxy.routing import RequestDialect, Route
from .test_budget import api, body, budget


async def invoke(tmp_path, payload, responder, **config):
    b = budget(tmp_path, **config)
    calls, sent = [], []
    async def record(call):
        calls.append(call)
    async def route(call):
        return Route("https://upstream.test", {"x-api-key": "host-secret", "x-bet-eval-call-id": "host-call"}, "m")
    async def handle(request):
        sent.append(request)
        response = responder(request, len(sent))
        return await response if hasattr(response, "__await__") else response
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        methods = relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))), route, record=record, client=client, budget=b)
        result = await methods["generate_anthropic"](payload, {"x-api-key": "caller"})
    await b.close_out()
    return b, result, sent, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("number", [8, 9, 35, 37, 41, 50, 59])
async def test_forwarder_publishes_raw_budget_facts(tmp_path, number):
    b, result, sent, calls = await invoke(tmp_path, body(), lambda r, n: httpx.Response(200, json={"id": "id", "type": "message", "role": "assistant", "model": "m", "content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2}}))
    assert b.spent == 12
    assert calls[0].budget["attempts"][0]["charge"] == 12
    assert calls[0].budget["call_id"] == "host-call"
    assert calls[0].budget["attempts"][0]["forwarded_bytes"] == len(sent[0].content)
    assert b"host-secret" not in b.journal_path.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("number,status,want", [(31, 400, 0), (31, 402, None), (31, 405, None), (31, 408, None), (31, 501, None), (31, 507, None), (33, 400, 12)])
async def test_attempt_status_never_erases_usage(tmp_path, monkeypatch, number, status, want):
    monkeypatch.setattr(relay, "_BASE_BACKOFF_SEC", 0)
    monkeypatch.setattr(relay, "_MAX_BACKOFF_SEC", 0)
    usage = {"usage": {"input_tokens": 10, "output_tokens": 2}} if number == 33 else {"error": {"message": "offline"}}
    b, result, sent, calls = await invoke(tmp_path, body(), lambda r, n: httpx.Response(status, json=usage))
    if want is None:
        assert b.spent == sum(e["R"] for e in map(json.loads, b.journal_path.read_text().splitlines()) if e["kind"] == "admit")
    else:
        assert b.spent == want


@pytest.mark.asyncio
async def test_43_77_rejects_before_confinement_removes_file(tmp_path):
    payload = body(messages=[{"role": "user", "content": [{"type": "document", "source": {"type": "file", "file_id": "resource"}}]}])
    b, result, sent, calls = await invoke(tmp_path, payload, lambda r, n: httpx.Response(200, json={}))
    assert sent == []
    assert b.spent == 0
    assert "unsupported_input" in str(result)


@pytest.mark.asyncio
async def test_55_verifier_ignores_agent_latch_but_is_metered(tmp_path):
    b = budget(tmp_path, limit=200, overshoot="one_call")
    calls = []
    async def record(call): calls.append(call)
    async def route(call): return Route("https://upstream.test", {}, "m")
    phase = ["agent"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"usage": {"input_tokens": 100, "output_tokens": 100}}))) as client:
        methods = relay.relay_methods(lambda: (phase[0], relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))), route, record=record, client=client, budget=b)
        await methods["generate_anthropic"](body(max_tokens=100), {"x-api-key": "caller"})
        phase[0] = "verifier"
        result = await methods["generate_anthropic"](body(max_tokens=100), {"x-api-key": "caller"})
    assert "usage" in result
    assert b.spent == 200
    assert calls[-1].budget["phase"] == "verifier"
    await b.close_out()


@pytest.mark.asyncio
async def test_50_52_budget_latch_precedes_routing_and_has_fixed_error(tmp_path):
    b = budget(tmp_path)
    b.latch_cause = "limit"
    b.spent = b.config.limit
    routes = []
    async def route(call):
        routes.append(call)
        return Route("https://upstream.test", {}, "m")
    async def record(call):
        pass
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as client:
        methods = relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))), route, record=record, client=client, budget=b)
        result = await methods["generate_anthropic"](body(), {"x-api-key": "caller"})
    assert routes == []
    payload = result[relay.PROVIDER_ERROR_KEY]
    assert payload["status"] == 400
    assert payload["code"] == "token_budget"
    assert payload["message"] == "generate_anthropic: token budget of 1,000,000 spent (1,000,000 tokens counted, rule harbor-growth/1)"
    b.release_writer()


@pytest.mark.asyncio
async def test_24_chat_stream_usage_normalization_is_forwarded(tmp_path):
    b = budget(tmp_path)
    sent = []
    async def route(call): return Route("https://upstream.test", {}, "m")
    async def record(call): pass
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 2}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        methods = relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))), route, record=record, client=client, budget=b)
        await methods["generate_completions"](body(stream=True, stream_options={"include_usage": False}), {"authorization": "Bearer caller"})
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert b.spent == 12
    await b.close_out()


@pytest.mark.asyncio
async def test_59_usage_cap_counts_persisted_records_not_resend_bodies(tmp_path):
    b = budget(tmp_path)
    sends = []
    async def route(call): return Route("https://upstream.test", {}, "m")
    async def record(call):
        with (tmp_path / "usage.jsonl").open("a") as stream:
            stream.write(json.dumps(call.budget) + "\n")
    def handle(request):
        sends.append(len(request.content))
        return httpx.Response(200, json={"usage": {"input_tokens": 10, "output_tokens": 2}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        methods = relay.relay_methods(lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}), max_recorded_bytes=20000)), route, record=record, client=client, budget=b, usage_only=True)
        for i in range(2):
            await methods["generate_anthropic"](body("x" * 100000), {"x-api-key": "caller"})
    assert len(sends) == 2
    assert b.recorded_bytes() < 20000
    await b.close_out()
