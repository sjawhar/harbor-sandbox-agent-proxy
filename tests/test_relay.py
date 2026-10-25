from __future__ import annotations

import json

import httpx

from harbor_sandbox_agent_proxy.collaborators import DefaultUpstream
from harbor_sandbox_agent_proxy.relay import RelayCall, RelayPolicy, relay_methods


async def test_forwarder_resolves_upstream_for_each_request() -> None:
    host_environment = {"ANTHROPIC_API_KEY": "first-host-key"}
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"content": []})

    calls: list[RelayCall] = []

    async def record(call: RelayCall) -> None:
        calls.append(call)

    policy = RelayPolicy(caller_key="sandbox-key", allowed_models=frozenset({"claude"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        methods = relay_methods(
            lambda: ("agent", policy),
            DefaultUpstream(host_environment),
            record=record,
            client=client,
        )
        await methods["generate_anthropic"](
            {"model": "claude", "messages": []}, {"x-api-key": "sandbox-key"}
        )
        host_environment["ANTHROPIC_API_KEY"] = "renewed-host-key"
        await methods["generate_anthropic"](
            {"model": "claude", "messages": []}, {"x-api-key": "sandbox-key"}
        )

    assert [request.headers["x-api-key"] for request in seen] == [
        "first-host-key",
        "renewed-host-key",
    ]
    assert all("x-api-key" not in call.headers for call in calls)


async def test_default_host_key_stays_out_of_sandbox_environment_and_json_outcome(
    tmp_path,
) -> None:
    from harbor_sandbox_agent_proxy.outcome import JsonOutcomeSink, TrialIdentity
    from harbor_sandbox_agent_proxy.trial_relay import TrialRelay

    host_key = "host-secret-not-for-sandbox"
    identity = TrialIdentity(name="trial-1", directory=tmp_path)
    sink = JsonOutcomeSink(identity, protected_values=frozenset({host_key}))
    relay = TrialRelay(
        trial=identity,
        upstream=DefaultUpstream({"ANTHROPIC_API_KEY": host_key}),
        sink=sink,
        declared_models=frozenset({"claude"}),
    )

    sandbox_environment = relay.phase_env()
    await relay.close()
    outcome = (tmp_path / "sandbox-agent-proxy" / "outcome.json").read_text(
        encoding="utf-8"
    )

    assert host_key not in sandbox_environment.values()
    assert host_key not in outcome


async def test_client_tool_search_stays_in_a_bridged_only_responses_request() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"output": []})

    calls: list[RelayCall] = []

    async def record(call: RelayCall) -> None:
        calls.append(call)

    policy = RelayPolicy(
        caller_key="sandbox-key",
        allowed_models=frozenset({"gpt"}),
        tools="bridged-only",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        methods = relay_methods(
            lambda: ("agent", policy),
            DefaultUpstream({"OPENAI_API_KEY": "host-key"}),
            record=record,
            client=client,
        )
        await methods["generate_responses"](
            {
                "model": "gpt",
                "input": "use the tool",
                "tools": [{"type": "tool_search", "execution": "client"}],
            },
            {"authorization": "Bearer sandbox-key"},
        )

    assert json.loads(seen[0].content)["tools"] == [
        {"type": "tool_search", "execution": "client"}
    ]
    assert calls[0].stripped == ()


async def test_bridged_only_keeps_a_bridged_responses_namespace() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"output": []})

    calls: list[RelayCall] = []

    async def record(call: RelayCall) -> None:
        calls.append(call)

    policy = RelayPolicy(
        caller_key="sandbox-key",
        allowed_models=frozenset({"gpt"}),
        tools="bridged-only",
        bridged={"e2e-tools": frozenset({"echo"})},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        methods = relay_methods(
            lambda: ("agent", policy),
            DefaultUpstream({"OPENAI_API_KEY": "host-key"}),
            record=record,
            client=client,
        )
        await methods["generate_responses"](
            {
                "model": "gpt",
                "input": "use the tool",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__e2e_tools",
                        "tools": [
                            {"type": "function", "name": "echo"},
                            {"type": "function", "name": "not-served"},
                        ],
                    },
                    {
                        "type": "namespace",
                        "name": "mcp__unbridged",
                        "tools": [{"type": "function", "name": "shell"}],
                    },
                ],
            },
            {"authorization": "Bearer sandbox-key"},
        )

    assert json.loads(seen[0].content)["tools"] == [
        {
            "type": "namespace",
            "name": "mcp__e2e_tools",
            "tools": [{"type": "function", "name": "echo"}],
        }
    ]
    assert calls[0].stripped == (
        "tool:mcp__e2e_tools/not-served",
        "tool:mcp__unbridged/shell",
        "namespace:mcp__unbridged",
    )
