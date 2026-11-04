"""Native HTTP/SSE controls against the design-pinned Inspect source.

Set HSAP_NATIVE_PROXY_SOURCE to the captured Inspect 0.3.285 proxy.py. The source
is hash-pinned rather than vendored, and no installed dependency is modified.
This is loopback HTTP, not a sandbox queue, deployed CLI or live model gate.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest

from harbor_sandbox_agent_proxy import budget as mod, relay
from harbor_sandbox_agent_proxy.routing import Route
from tests.test_budget import body, budget, measured

NATIVE_SHA256 = "7c1ea740e18e3c3c7e8599afcf2f4a5e8c51e96460f7f3f0c3f06dcdeee3a449"


@pytest.fixture
def native_proxy():
    source = os.environ.get("HSAP_NATIVE_PROXY_SOURCE")
    if source is None:
        pytest.skip("set HSAP_NATIVE_PROXY_SOURCE to run the pinned native renderer controls")
    path = Path(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == NATIVE_SHA256
    spec = importlib.util.spec_from_file_location("hsap_native_inspect_0_3_285", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_f7_closed_code_survives_native_http_and_sse(tmp_path, native_proxy, streaming):
    b = budget(tmp_path)
    a = await b.admit(b.prepare("anthropic", "m", body("x" * 990000)), "initial")
    b.settle(a, measured(990000))
    sent = []
    async def route(call):
        return Route("https://offline.test", {}, "m")
    async def record(call):
        pass
    def upstream(request):
        sent.append(request)
        return httpx.Response(200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        methods = relay.relay_methods(
            lambda: ("agent", relay.RelayPolicy(caller_key="caller", allowed_models=frozenset({"m"}))),
            route, record=record, client=client, budget=b,
        )
        async def bridge(method, **params):
            return await methods[method](params["json_data"], params.get("headers"))
        server = await native_proxy.model_proxy_server(0, bridge)
        server.server = await asyncio.start_server(server._handle_client, "127.0.0.1", 0)
        port = server.server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
                response = await http.post("/v1/messages", json=body("x" * 20000, stream=streaming), headers={"x-api-key": "caller"})
            if streaming:
                assert response.status_code == 200
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                error = next(event["error"] for event in events if event.get("type") == "error")
            else:
                assert response.status_code == 400
                error = response.json()["error"]
            assert error["type"] == "invalid_request_error"
            assert " [hsap-budget-code:token_budget_insufficient]" in error["message"]
            assert mod.native_refusal_code(error["message"]) == "token_budget_insufficient"
            assert error.get("code") is None  # hsap's encoding, not an Inspect error.code claim
            terminal = error["message"]
        finally:
            await server.stop()
    await b.close_out()
    assert sent == []
    recovered = mod.read_budget(tmp_path)
    ending = mod.classify_ending(recovered, trial=b.trial, config_hash=b.config_hash,
                                 hsap_commit=b.config.hsap_commit, limit=b.config.limit,
                                 native_terminal=terminal, grade=1)
    assert ending.ending == "budget_exhausted" and ending.grade == 1


@pytest.mark.parametrize("message", [None, "unrelated", "x [hsap-budget-code:invented]", "x [hsap-budget-code:token_budget] trailing"])
def test_f7_decoder_accepts_only_the_closed_terminal_encoding(message):
    assert mod.native_refusal_code(message) is None
