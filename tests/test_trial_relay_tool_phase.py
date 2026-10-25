"""A bridged task tool is served only in the agent's turn."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import types
from collections.abc import AsyncGenerator, Generator, Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx
import inspect_ai.agent
import inspect_ai.tool
import pydantic
import pytest
from harbor.environments.base import BaseEnvironment

from harbor_sandbox_agent_proxy import trial_relay
from harbor_sandbox_agent_proxy._inspect_compat import (
    SandboxAgentBridge,
    call_tool,
    list_tools,
)
from harbor_sandbox_agent_proxy.collaborators import (
    BridgedMcpServer,
    RequestDialect,
    UpstreamTarget,
)
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.outcome import (
    EventSink,
    RelayOutcome,
    TrialEvent,
    TrialIdentity,
)
from harbor_sandbox_agent_proxy.proxy import _register_bridged_tool_specs
from harbor_sandbox_agent_proxy.relay import RelayCall, relay_methods
from harbor_sandbox_agent_proxy.trial_relay import TrialRelay, phase_guarded_tool

_SERVER = "task-tools"
_TOOL = "ping"
_ARGUMENTS: dict[str, pydantic.JsonValue] = {"text": "hello"}
_REFUSAL_DATA: dict[str, pydantic.JsonValue] = {
    "server": _SERVER,
    "tool": _TOOL,
    "arguments": _ARGUMENTS,
    "reason": "outside the agent's turn",
}


def _settings(**overrides: object) -> RelayConfig:
    return RelayConfig.model_validate(
        {
            "declared_models": ["anthropic/claude-test"],
            "max_request_bytes": 1024,
            "max_calls": 10,
            **overrides,
        }
    )


@inspect_ai.tool.tool
def ping(ran: list[str]) -> inspect_ai.tool.Tool:
    async def execute(text: str) -> str:
        """Answer a ping from the task's own server, echoing ``text``.

        Args:
            text: The text to echo back.
        """
        ran.append(text)
        return f"pong: {text}"

    return execute


class _Sink(EventSink):
    def __init__(self) -> None:
        self.events: list[TrialEvent] = []
        self.outcome: RelayOutcome | None = None

    async def record(self, event: TrialEvent) -> None:
        self.events.append(event)

    async def close(self, outcome: RelayOutcome) -> None:
        self.outcome = outcome


class _Upstream:
    async def resolve(self, *, dialect: RequestDialect, model: str) -> UpstreamTarget:
        del dialect, model
        return UpstreamTarget(
            "https://upstream.test/v1/messages", {"x-api-key": "host-key"}
        )


def _bridge(specs: Sequence[inspect_ai.agent.BridgedToolsSpec]) -> SandboxAgentBridge:
    bridge = SandboxAgentBridge(
        state=inspect_ai.agent.AgentState(messages=[]),
        filter=None,
        retry_refusals=None,
        compaction=None,
        port=13131,
        model=None,
    )
    _register_bridged_tool_specs(bridge, specs, 13131)
    return bridge


def _grant(
    bridge: SandboxAgentBridge, arguments: dict[str, pydantic.JsonValue]
) -> None:
    served = inspect_ai.tool.ToolDef(bridge.bridged_tools[_SERVER][_TOOL])
    declared = f"mcp__{_SERVER}__{_TOOL}"
    bridge.register_tool_execution_grants(
        [
            inspect_ai.tool.ToolCall(
                id=f"call-{uuid4().hex}", function=declared, arguments=dict(arguments)
            )
        ],
        [
            inspect_ai.tool.ToolInfo(
                name=declared,
                description=served.description,
                parameters=served.parameters,
            )
        ],
    )


@dataclasses.dataclass
class _Harness:
    relay: TrialRelay
    bridge: SandboxAgentBridge
    sink_events: list[TrialEvent]
    ran: list[str]
    failures: list[Exception]
    require_proposal: bool

    async def call(self) -> pydantic.JsonValue:
        if self.require_proposal:
            _grant(self.bridge, _ARGUMENTS)
        return await call_tool(self.bridge)(_SERVER, _TOOL, dict(_ARGUMENTS))


@contextlib.asynccontextmanager
async def _opened(
    monkeypatch: pytest.MonkeyPatch, *, require_proposal: bool
) -> AsyncGenerator[_Harness]:
    ran: list[str] = []
    sink = _Sink()
    captured: list[Sequence[inspect_ai.agent.BridgedToolsSpec]] = []

    class ToolSource:
        async def tools(self) -> list[inspect_ai.tool.Tool]:
            return [ping(ran)]

    def server(*, name: str, command: str, args: list[str]) -> object:
        del name, command, args
        return object()

    def tools(server: object) -> ToolSource:
        del server
        return ToolSource()

    @contextlib.asynccontextmanager
    async def connection(source: object) -> AsyncGenerator[None]:
        del source
        yield

    def adapter(environment: object, **kwargs: object) -> object:
        del kwargs
        return environment

    @contextlib.contextmanager
    def bind(sandbox: object) -> Generator[object]:
        yield sandbox

    @contextlib.asynccontextmanager
    async def proxy(
        sandbox: object,
        *,
        bridged_tools: Sequence[inspect_ai.agent.BridgedToolsSpec],
        **kwargs: object,
    ) -> AsyncGenerator[object]:
        del sandbox, kwargs
        captured.append(bridged_tools)
        yield types.SimpleNamespace(port=13131)

    async def answered(sandbox: object, *, port: int) -> None:
        del sandbox, port

    monkeypatch.setattr(inspect_ai.tool, "mcp_server_sandbox", server)
    monkeypatch.setattr(inspect_ai.tool, "mcp_tools", tools)
    monkeypatch.setattr(inspect_ai.tool, "mcp_connection", connection)
    monkeypatch.setattr(trial_relay, "HarborSandboxEnvironment", adapter)
    monkeypatch.setattr(trial_relay, "bind_sandbox_environment", bind)
    monkeypatch.setattr(trial_relay, "sandbox_model_proxy", proxy)
    monkeypatch.setattr(trial_relay, "_wait_for_proxy", answered)
    relay_instance = TrialRelay(
        trial=TrialIdentity(name="example__1", directory=Path("/tmp/example__1")),
        config=_settings(),
        upstream=_Upstream(),
        sink=sink,
        environment=cast(BaseEnvironment, object()),
        bridged_servers=(
            BridgedMcpServer(
                name=_SERVER,
                command="python3",
                args=("/opt/task_tools.py",),
                require_proposal=require_proposal,
            ),
        ),
    )
    await relay_instance.open_agent_phase()
    try:
        (specs,) = captured
        bridge = _bridge(specs)
        failures: list[Exception] = []
        monkeypatch.setattr(bridge, "request_fail", failures.append)
        yield _Harness(
            relay=relay_instance,
            bridge=bridge,
            sink_events=sink.events,
            ran=ran,
            failures=failures,
            require_proposal=require_proposal,
        )
    finally:
        if sink.outcome is None:
            _ = await relay_instance.close()


def _refusals(harness: _Harness) -> list[TrialEvent]:
    assert harness.sink_events == harness.relay.events
    refusals: list[TrialEvent] = []
    for event in harness.relay.events:
        if event.kind != "refused-tool-call":
            continue
        assert event.phase == "verifier"
        assert event.data == _REFUSAL_DATA
        refusals.append(event)
    return refusals


async def test_a_claude_code_trials_granted_call_is_refused_outside_the_agents_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _opened(monkeypatch, require_proposal=True) as harness:
        _grant(harness.bridge, _ARGUMENTS)
        await harness.relay.end_agent_phase()

        with pytest.raises(inspect_ai.tool.ToolError) as refused:
            _ = await call_tool(harness.bridge)(_SERVER, _TOOL, dict(_ARGUMENTS))

        assert f"{_SERVER}/{_TOOL}" in refused.value.message
        assert "relay phase: verifier" in refused.value.message
        assert not harness.bridge.consume_tool_execution_grant(
            _SERVER, _TOOL, dict(_ARGUMENTS)
        )
        assert harness.failures == []
        assert harness.ran == []
        (event,) = _refusals(harness)
        outcome = await harness.relay.close()

    assert [recorded for recorded in outcome.events if recorded == event] == [event]
    assert outcome.counts == {
        "agent_calls": 0,
        "verifier_calls": 0,
        "failed_calls": 0,
        "refused_calls": 0,
    }


async def test_a_codex_trials_bare_call_is_served_in_the_agents_turn_and_refused_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _opened(monkeypatch, require_proposal=False) as harness:
        call = call_tool(harness.bridge)

        assert await call(_SERVER, _TOOL, dict(_ARGUMENTS)) == "pong: hello"
        assert harness.ran == ["hello"]
        assert harness.relay.events == harness.sink_events
        assert not _refusals(harness)

        await harness.relay.end_agent_phase()
        with pytest.raises(inspect_ai.tool.ToolError, match="relay phase: verifier"):
            _ = await call(_SERVER, _TOOL, dict(_ARGUMENTS))

    assert harness.ran == ["hello"]
    assert harness.failures == []
    assert len(_refusals(harness)) == 1


@pytest.mark.parametrize(
    "require_proposal", [True, False], ids=["claude-code", "codex"]
)
async def test_a_later_steps_agent_turn_is_served_between_two_refusals(
    monkeypatch: pytest.MonkeyPatch, require_proposal: bool
) -> None:
    async with _opened(monkeypatch, require_proposal=require_proposal) as harness:
        await harness.relay.end_agent_phase()
        with pytest.raises(inspect_ai.tool.ToolError):
            _ = await harness.call()

        await harness.relay.resume_agent_phase()
        assert await harness.call() == "pong: hello"

        await harness.relay.end_agent_phase()
        with pytest.raises(inspect_ai.tool.ToolError):
            _ = await harness.call()

    assert harness.ran == ["hello"]
    assert harness.failures == []
    refusals = _refusals(harness)
    assert [event.phase for event in refusals] == ["verifier", "verifier"]


@pytest.mark.parametrize(
    "require_proposal", [True, False], ids=["claude-code", "codex"]
)
async def test_the_judges_model_calls_are_answered_while_its_tool_calls_are_refused(
    monkeypatch: pytest.MonkeyPatch, require_proposal: bool
) -> None:
    sent: list[dict[str, Any]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"content": []})

    async def record(_call: RelayCall) -> None:
        return None

    async with (
        _opened(monkeypatch, require_proposal=require_proposal) as harness,
        httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client,
    ):
        await harness.relay.end_agent_phase()
        generate = relay_methods(
            harness.relay.current_policy,
            harness.relay.upstream,
            record=record,
            client=client,
        )["generate_anthropic"]

        answer = await generate(
            {"model": "claude-test", "messages": []},
            {"x-api-key": harness.relay.verifier_key},
        )
        with pytest.raises(inspect_ai.tool.ToolError):
            _ = await harness.call()

    assert answer == {"content": []}
    assert [body["model"] for body in sent] == ["claude-test"]
    assert harness.ran == []
    assert harness.failures == []
    assert len(_refusals(harness)) == 1


async def _never_refused(event: TrialEvent) -> None:
    raise AssertionError(f"the agent phase refused a tool call: {event!r}")


async def test_the_guard_changes_nothing_the_bridge_serves_for_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = ping([])
    guarded = phase_guarded_tool(
        inner, server=_SERVER, phase=lambda: "agent", refused=_never_refused
    )
    expected = inspect_ai.tool.ToolDef(inner)
    actual = inspect_ai.tool.ToolDef(guarded)

    assert (
        actual.name,
        actual.description,
        actual.parameters.model_dump(),
        actual.parallel,
        actual.max_output,
        actual.options,
    ) == (
        expected.name,
        expected.description,
        expected.parameters.model_dump(),
        expected.parallel,
        expected.max_output,
        expected.options,
    )
    assert actual.viewer is expected.viewer
    assert actual.model_input is expected.model_input

    unwrapped = _bridge(
        [inspect_ai.agent.BridgedToolsSpec(name=_SERVER, tools=[inner])]
    )
    async with _opened(monkeypatch, require_proposal=True) as harness:
        served = await list_tools(harness.bridge)(_SERVER)

    assert served == await list_tools(unwrapped)(_SERVER)
    assert isinstance(served, list)
    (listed,) = served
    assert isinstance(listed, dict)
    assert listed["name"] == _TOOL
    assert listed["description"] == expected.description
