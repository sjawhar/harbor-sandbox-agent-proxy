"""Contracts for the RelayEnvironment host-subclass seam."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from pathlib import Path, PurePosixPath
from typing import ClassVar, cast

import harbor.environments.base as harbor_environment_base
import harbor.models.agent.context as agent_context
import pytest

from harbor_sandbox_agent_proxy import agent as relay_agent
from harbor_sandbox_agent_proxy.environment import (
    RelayEnvironment,
    RunnerSecretExposureError,
)
from harbor_sandbox_agent_proxy.outcome import EventSink, JsonOutcomeSink, RelayOutcome
from harbor_sandbox_agent_proxy.routing import (
    ModelCall,
    RequestDialect,
    Route,
    RouteCall,
    default_protected_values,
    match_declared,
)
from harbor_sandbox_agent_proxy.task_metadata import BridgedMcpServer
from tests.test_relay_agent import InstalledAgent
from tests.test_relay_environment import _empty_relay_outcome, _environment, _inner


@dataclasses.dataclass
class RecordingSink(EventSink):
    order: list[str]
    records: list[object] = dataclasses.field(default_factory=list)
    outcomes: list[RelayOutcome] = dataclasses.field(default_factory=list)

    async def record(self, event: object) -> None:
        self.records.append(event)

    async def close(self, outcome: RelayOutcome) -> None:
        self.order.append("sink.close")
        self.outcomes.append(outcome)


class Host(RelayEnvironment):
    route_calls: ClassVar[list[ModelCall]] = []
    secret_values: ClassVar[frozenset[str]] = frozenset()
    event_sink: ClassVar[RecordingSink]
    servers: ClassVar[tuple[BridgedMcpServer, ...]] = ()

    @classmethod
    def reset(cls, *, secret: str = "host-secret") -> None:
        cls.route_calls = []
        cls.secret_values = frozenset({secret})
        cls.event_sink = RecordingSink([])
        cls.servers = (BridgedMcpServer(name="host-tools", command="host-tools"),)

    async def route(self, call: ModelCall) -> Route:
        type(self).route_calls.append(call)
        return Route(
            base_url="https://host.example/anthropic",
            headers={"x-api-key": "host-key"},
            model=match_declared(call),
        )

    def protected_values(self) -> Collection[str]:
        return type(self).secret_values

    async def open_event_sink(self, _trial: object) -> EventSink:
        return type(self).event_sink

    async def bridged_tool_servers(self) -> Sequence[BridgedMcpServer]:
        return type(self).servers


@dataclasses.dataclass
class ClosingRelay:
    order: list[str]
    outcome: RelayOutcome
    close_error: BaseException | None = None
    caller_keys: frozenset[str] = frozenset({"caller-key"})
    verifier_conversation: object | None = None

    def phase_env(self) -> dict[str, str]:
        return {}

    async def close(self) -> RelayOutcome:
        self.order.append("relay.close")
        if self.close_error is not None:
            raise self.close_error
        return self.outcome


class CapturingTrialRelay:
    latest: ClassVar[CapturingTrialRelay | None] = None

    def __init__(self, **kwargs: object) -> None:
        self.route = cast(RouteCall, kwargs["route"])
        self.bridged_servers = cast(
            Sequence[BridgedMcpServer], kwargs["bridged_servers"]
        )
        self.agent_key = cast(str, kwargs["agent_key"])
        self.record_error: Exception | None = None
        CapturingTrialRelay.latest = self

    def mcp_server_configs(self) -> list[object]:
        return []

    def phase_env(self) -> dict[str, str]:
        return {"ANTHROPIC_API_KEY": self.agent_key}

    async def open_agent_phase(self) -> None:
        return None

    async def resume_agent_phase(self) -> None:
        return None

    async def end_agent_phase(self) -> None:
        return None


def _wrapped_installed_agent() -> dict[str, str]:
    return {"import_path": f"{InstalledAgent.__module__}:{InstalledAgent.__name__}"}


async def test_relay_agent_uses_the_environment_host_for_route_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Host.reset()
    monkeypatch.setattr(relay_agent, "TrialRelay", CapturingTrialRelay)
    environment = _environment(
        tmp_path,
        environment_class=Host,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    wrapper = relay_agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent=_wrapped_installed_agent(),
        relay={"declared_models": ["anthropic/claude-test"]},
    )

    await wrapper.run("work", environment, agent_context.AgentContext())

    captured = CapturingTrialRelay.latest
    assert captured is not None
    call = ModelCall(
        RequestDialect.MESSAGES,
        "claude-test",
        ("anthropic/claude-test",),
    )
    assert (await captured.route(call)).base_url == "https://host.example/anthropic"
    assert Host.route_calls == [call]
    assert captured.bridged_servers == Host.servers


async def test_relay_agent_refuses_an_environment_that_only_imitates_the_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(relay_agent, "TrialRelay", CapturingTrialRelay)
    _ = _environment(
        tmp_path,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    wrapper = relay_agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent=_wrapped_installed_agent(),
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    inner = _inner()
    setattr(inner, "attach_relay", lambda _relay: None)

    with pytest.raises(RuntimeError, match="use both wrapper classes"):
        _ = await wrapper._open_relay(
            cast(harbor_environment_base.BaseEnvironment, inner)
        )


async def test_default_host_methods_use_the_plain_harbor_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "default-host-key")
    (tmp_path / "task.toml").write_text(
        """
[metadata.sandbox_agent_proxy]
bridged_mcp_servers = [{ name = "task-tools", command = "task-tools" }]
""".strip(),
        encoding="utf-8",
    )
    environment = _environment(
        tmp_path,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    call = ModelCall(
        RequestDialect.MESSAGES,
        "claude-test",
        ("anthropic/claude-test",),
    )

    assert (await environment.route(call)).headers == {"x-api-key": "default-host-key"}
    assert environment.protected_values() == default_protected_values()
    assert isinstance(
        await environment.open_event_sink(environment.trial), JsonOutcomeSink
    )
    assert await environment.bridged_tool_servers() == (
        BridgedMcpServer(name="task-tools", command="task-tools"),
    )


async def test_host_protected_values_refuse_exec_environment(tmp_path: Path) -> None:
    Host.reset()
    environment = _environment(
        tmp_path,
        environment_class=Host,
        relay={"declared_models": ["anthropic/claude-test"]},
    )

    with pytest.raises(RunnerSecretExposureError, match="HOST_SECRET"):
        _ = await environment.exec("unsafe", env={"HOST_SECRET": "host-secret"})


async def test_host_sink_closes_after_the_inner_environment_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Host.reset()
    environment = _environment(
        tmp_path,
        environment_class=Host,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    order = Host.event_sink.order
    inner = _inner()
    original_stop = inner.stop

    async def stop(delete: bool) -> None:
        order.append("stop")
        await original_stop(delete)

    monkeypatch.setattr(inner, "stop", stop)
    relay = ClosingRelay(order, _empty_relay_outcome())
    environment.attach_relay(relay)

    await environment.stop(delete=False)

    assert order == ["stop", "relay.close", "sink.close"]
    assert Host.event_sink.outcomes == [relay.outcome]
    assert environment.caller_keys == {"caller-key"}


async def test_stop_closes_the_relay_and_sink_when_the_inner_stop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Host.reset()
    environment = _environment(
        tmp_path,
        environment_class=Host,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    order = Host.event_sink.order
    inner = _inner()

    async def stop(_delete: bool) -> None:
        order.append("stop")
        raise RuntimeError("inner stop failed")

    monkeypatch.setattr(inner, "stop", stop)
    environment.attach_relay(ClosingRelay(order, _empty_relay_outcome()))

    with pytest.raises(RuntimeError, match="inner stop failed"):
        await environment.stop(delete=False)

    assert order == ["stop", "relay.close", "sink.close"]


async def test_stop_stops_the_inner_environment_before_a_relay_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Host.reset()
    environment = _environment(
        tmp_path,
        environment_class=Host,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    order = Host.event_sink.order
    inner = _inner()
    original_stop = inner.stop

    async def stop(delete: bool) -> None:
        order.append("stop")
        await original_stop(delete)

    monkeypatch.setattr(inner, "stop", stop)
    environment.attach_relay(
        ClosingRelay(
            order,
            _empty_relay_outcome(),
            close_error=RuntimeError("relay close failed"),
        )
    )

    with pytest.raises(RuntimeError, match="relay close failed"):
        await environment.stop(delete=False)

    assert order == ["stop", "relay.close"]


async def test_start_does_not_exec_after_starting_the_sandbox(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        relay={"declared_models": ["anthropic/claude-test"]},
    )

    await environment.start(force_build=False)

    assert [
        call.name for call in _inner().calls if call.name != "_validate_definition"
    ] == [
        "_startup_env",
        "start",
    ]


async def test_configured_conversation_path_is_written_beneath_tests(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        relay={
            "declared_models": ["anthropic/claude-test"],
            "conversation_path": "/tests/hawk-relay/conversation.json",
        },
    )
    environment.attach_relay(
        ClosingRelay(
            [],
            _empty_relay_outcome(),
            verifier_conversation={"messages": []},
        )
    )

    await environment._write_conversation(PurePosixPath("/tests"))

    assert "/tests/hawk-relay/conversation.json" in _inner().uploaded_contents


async def test_runner_loop_outcome_closes_the_host_sink_at_stop(
    tmp_path: Path,
) -> None:
    Host.reset()
    environment = _environment(
        tmp_path,
        environment_class=Host,
        relay={"declared_models": ["anthropic/claude-test"]},
    )
    outcome = _empty_relay_outcome()

    environment.record_runner_loop_outcome(outcome)
    await environment.stop(delete=False)

    assert Host.event_sink.outcomes == [outcome]
