from __future__ import annotations

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
    outcome = (tmp_path / "sandbox-agent-proxy" / "outcome.json").read_text(encoding="utf-8")

    assert host_key not in sandbox_environment.values()
    assert host_key not in outcome
