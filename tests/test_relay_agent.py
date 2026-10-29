"""Contracts for Harbor's wrapper around installed and runner-loop agents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import operator
import types
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, override
from uuid import uuid4

import harbor.agents.base as agent_base
import harbor.agents.capabilities as agent_capabilities
import harbor.agents.installed.base as installed_agent_base
import harbor.agents.model_connection as model_connection
import harbor.agents.options as agent_options
import harbor.environments.base as environment_base
import harbor.models.agent.context as agent_context
import harbor.models.task.config as task_config
import pydantic
import pytest

from harbor_sandbox_agent_proxy import agent, trial_relay
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.environment import RelayEnvironment
from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
from tests.test_relay_environment import _environment as _relay_environment
from tests.test_relay_environment import _inner

RELAY: dict[str, object] = {
    "polling_interval": 0.25,
    "allowed_betas": ["claude-code-20250219"],
    "declared_models": ["anthropic/claude-test"],
}
RESUMABLE = agent_capabilities.AgentCapabilities(resume=True, load_atif_trajectory=True)


class InstalledAgent(installed_agent_base.BaseInstalledAgent):
    latest: ClassVar[InstalledAgent | None] = None
    capabilities: ClassVar[agent_capabilities.AgentCapabilities] = RESUMABLE
    MODEL_CONNECTION: ClassVar[model_connection.ModelConnectionSpec] = (
        model_connection.ModelConnectionSpec(default_provider="anthropic")
    )
    seen_mcp_servers: list[task_config.MCPServerConfig]
    calls: list[str]
    setup_timeout_sec: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.setup_timeout_sec = kwargs.get("override_setup_timeout_sec")
        super().__init__(*args, **kwargs)
        self.seen_mcp_servers = []
        self.calls = []
        InstalledAgent.latest = self

    @staticmethod
    @override
    def name() -> str:
        return "installed-test-agent"

    @override
    async def install(self, environment: environment_base.BaseEnvironment) -> None:
        del environment

    @override
    async def run(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context
        self.calls.append("run")
        self.seen_mcp_servers = list(self.mcp_servers)

    @override
    async def resume(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context
        self.calls.append("resume")

    @override
    async def load(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context
        self.calls.append("load")


class SetupExecAgent(InstalledAgent):
    setup_calls: ClassVar[int] = 0

    @override
    async def setup(self, environment: environment_base.BaseEnvironment) -> None:
        type(self).setup_calls += 1
        _ = await environment.exec("setup")


class LoopOptions(agent_options.AgentOptions):
    api_base: str | None = None


class LoopAgent(agent_base.BaseAgent):
    latest: ClassVar[LoopAgent | None] = None
    options_model: ClassVar[type[Any] | None] = LoopOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        LoopAgent.latest = self

    @staticmethod
    @override
    def name() -> str:
        return "loop-test-agent"

    @override
    def version(self) -> str | None:
        return "test"

    @override
    async def setup(self, environment: environment_base.BaseEnvironment) -> None:
        del environment

    @override
    async def run(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context
        self._write_trajectory("runner trajectory")

    @override
    async def resume(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context
        self._write_trajectory("resumed trajectory")

    def _write_trajectory(self, text: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(text, encoding="utf-8")


class FailingLoopAgent(LoopAgent):
    @override
    async def run(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context
        self._write_trajectory("partial trajectory")
        raise RuntimeError("the agent's loop failed")


class LoopWithoutApiBase(agent_base.BaseAgent):
    @staticmethod
    @override
    def name() -> str:
        return "loop-without-api-base"

    @override
    def version(self) -> str | None:
        return "test"

    @override
    async def setup(self, environment: environment_base.BaseEnvironment) -> None:
        del environment

    @override
    async def run(
        self,
        instruction: str,
        environment: environment_base.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        del instruction, environment, context


class PlainAgent(LoopWithoutApiBase):
    latest: ClassVar[PlainAgent | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        PlainAgent.latest = self


class FakeTrialRelay:
    latest: ClassVar[FakeTrialRelay | None] = None
    next_record_error: ClassVar[Exception | None] = None
    created: ClassVar[int] = 0

    def __init__(self, **kwargs: object) -> None:
        self.agent_key = str(kwargs["agent_key"])
        self.verifier_key = "verifier-caller-key"
        self.record_error = FakeTrialRelay.next_record_error
        self.kwargs = kwargs
        self.opened = 0
        self.resumed = 0
        self.ended = 0
        self.phase = "agent"
        FakeTrialRelay.latest = self
        FakeTrialRelay.created += 1

    def mcp_server_configs(self) -> list[task_config.MCPServerConfig]:
        return [
            task_config.MCPServerConfig(
                name="task-tools",
                transport="streamable-http",
                url="http://localhost:13131/mcp/task-tools",
            )
        ]

    def phase_env(self) -> dict[str, str]:
        key = self.agent_key if self.phase == "agent" else self.verifier_key
        return {
            "ANTHROPIC_API_KEY": key,
            "ANTHROPIC_BASE_URL": "http://localhost:13131",
        }

    async def open_agent_phase(self) -> None:
        self.opened += 1

    async def resume_agent_phase(self) -> None:
        self.phase = "agent"
        self.resumed += 1

    async def end_agent_phase(self) -> None:
        self.phase = "verifier"
        self.ended += 1


class FakeRelayEnvironment(RelayEnvironment):
    async def bridged_tool_servers(self) -> tuple[()]:
        return ()


@pytest.fixture(autouse=True)
def relay_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "TrialRelay", FakeTrialRelay)
    FakeTrialRelay.next_record_error = None
    FakeTrialRelay.created = 0
    SetupExecAgent.setup_calls = 0


def _installed_agent_config(
    *, model_name: str | None = "anthropic/claude-test"
) -> dict[str, object]:
    return {"import_path": f"{__name__}:InstalledAgent", "model_name": model_name}


def _relay_agent(
    tmp_path: Path, wrapped: dict[str, object], **kwargs: Any
) -> agent.RelayAgent:
    return agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent=wrapped,
        relay=RELAY,
        extra_env={"UNMANAGED": "kept", "ANTHROPIC_API_KEY": "untrusted-job-key"},
        **kwargs,
    )


def _environment(tmp_path: Path) -> RelayEnvironment:
    return _relay_environment(
        tmp_path,
        environment_class=FakeRelayEnvironment,
        relay=RELAY,
    )


async def test_installed_agent_setup_receives_declared_proxy_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_key = "host-key-never-enters-the-sandbox"
    monkeypatch.setenv("ANTHROPIC_API_KEY", host_key)
    wrapper = _relay_agent(
        tmp_path,
        {
            "import_path": f"{__name__}:SetupExecAgent",
            "model_name": "anthropic/claude-test",
        },
    )
    environment = _environment(tmp_path)

    assert environment.relay is None
    with environment.scoped_exec_env(wrapper.extra_env):
        await wrapper.setup(environment)

    (setup_call,) = [call for call in _inner().calls if call.name == "exec"]
    setup_env = setup_call.kwargs["env"]
    assert isinstance(setup_env, dict)
    assert setup_env == {
        "PERSISTENT": "kept",
        "UNMANAGED": "kept",
        "ANTHROPIC_API_KEY": wrapper._agent_key,
        "ANTHROPIC_BASE_URL": "http://localhost:13131",
    }
    assert host_key not in setup_env.values()
    assert environment.relay is None


async def test_installed_agent_gets_proxy_environment_and_bridged_servers_only_for_run(
    tmp_path: Path,
) -> None:
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    assert InstalledAgent.latest is not None
    inner = InstalledAgent.latest
    assert inner.extra_env == {
        "UNMANAGED": "kept",
        "ANTHROPIC_API_KEY": wrapper._agent_key,
        "ANTHROPIC_BASE_URL": "http://localhost:13131",
    }
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in inner.extra_env
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in inner.extra_env
    assert inner.mcp_servers == []
    environment = _environment(tmp_path)
    await wrapper.run("do work", environment, agent_context.AgentContext())
    assert environment.relay is FakeTrialRelay.latest
    assert FakeTrialRelay.latest is not None and FakeTrialRelay.latest.opened
    assert FakeTrialRelay.latest.ended
    assert FakeTrialRelay.latest.agent_key == inner.extra_env["ANTHROPIC_API_KEY"]
    assert FakeTrialRelay.latest.kwargs["bridged_servers"] == ()
    assert [(server.name, server.url) for server in inner.seen_mcp_servers] == [
        ("task-tools", "http://localhost:13131/mcp/task-tools")
    ]


async def test_installed_agent_fails_the_trial_when_recording_failed(
    tmp_path: Path,
) -> None:
    FakeTrialRelay.next_record_error = ValueError("event sink unavailable")
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    with pytest.raises(RuntimeError, match="could not record a model call"):
        await wrapper.run(
            "do work", _environment(tmp_path), agent_context.AgentContext()
        )
    assert FakeTrialRelay.latest is not None and FakeTrialRelay.latest.ended


def test_installed_agent_without_model_has_no_small_fast_alias(tmp_path: Path) -> None:
    _ = _relay_agent(tmp_path, _installed_agent_config(model_name=None))
    assert InstalledAgent.latest is not None
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in InstalledAgent.latest.extra_env


async def test_runner_loop_agent_preserves_host_trajectory_on_the_environment(
    tmp_path: Path,
) -> None:
    wrapper = _relay_agent(
        tmp_path,
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    )
    environment = _environment(tmp_path)
    await wrapper.run("do work", environment, agent_context.AgentContext())
    assert LoopAgent.latest is not None
    assert isinstance(LoopAgent.latest.options, LoopOptions)
    assert LoopAgent.latest.options.api_base is None
    saved = tmp_path / "sandbox-agent-proxy" / "trajectory.json"
    assert saved.read_text(encoding="utf-8") == "runner trajectory"
    await environment.stop(delete=False)
    outcome = environment.outcome
    assert outcome is not None
    assert outcome.mode == "runner-loop"
    assert outcome.runner_atif == saved
    assert outcome.runner_atif_sha256 is not None


async def test_the_jobs_relay_settings_reach_the_trial_relay_typed(
    tmp_path: Path,
) -> None:
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    await wrapper.run("do work", _environment(tmp_path), agent_context.AgentContext())
    assert FakeTrialRelay.latest is not None
    settings = FakeTrialRelay.latest.kwargs["config"]
    assert isinstance(settings, RelayConfig)
    assert settings.polling_interval == 0.25
    assert settings.allowed_betas == ("claude-code-20250219",)
    assert settings.declared_models == ("anthropic/claude-test",)


@pytest.mark.parametrize(
    ("relay", "field"),
    [
        pytest.param({}, "declared_models", id="no-declared-models"),
        pytest.param({**RELAY, "polling_interval": 0}, "polling_interval", id="bad"),
        pytest.param({**RELAY, "gateway": "x"}, "gateway", id="unknown-key"),
    ],
)
def test_a_wrapped_agent_refuses_relay_settings_it_cannot_parse(
    tmp_path: Path, relay: dict[str, object], field: str
) -> None:
    with pytest.raises(pydantic.ValidationError, match=field):
        _ = agent.RelayAgent(
            logs_dir=tmp_path / "logs", agent=_installed_agent_config(), relay=relay
        )


def test_a_runner_loop_agent_gets_the_jobs_agent_env(tmp_path: Path) -> None:
    _ = _relay_agent(
        tmp_path,
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    )
    assert LoopAgent.latest is not None
    assert LoopAgent.latest.extra_env == {
        "UNMANAGED": "kept",
        "ANTHROPIC_API_KEY": "untrusted-job-key",
    }


def test_the_setup_timeout_override_reaches_the_inner_agent(tmp_path: Path) -> None:
    _ = _relay_agent(
        tmp_path, _installed_agent_config(), override_setup_timeout_sec=900.0
    )
    assert InstalledAgent.latest is not None
    assert InstalledAgent.latest.setup_timeout_sec == 900.0


def test_a_wrapped_agent_declares_its_inner_agents_capabilities(tmp_path: Path) -> None:
    installed = _relay_agent(tmp_path, _installed_agent_config())
    loop = _relay_agent(
        tmp_path,
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    )
    assert installed.capabilities == RESUMABLE
    assert loop.capabilities == agent_capabilities.AgentCapabilities()
    assert agent.RelayAgent.capabilities == agent_capabilities.AgentCapabilities()


@pytest.mark.parametrize("method", ["resume", "load"])
async def test_resume_and_load_run_through_the_relay(
    tmp_path: Path, method: str
) -> None:
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    environment = _environment(tmp_path)
    await getattr(wrapper, method)(
        "do work",
        environment,
        agent_context.AgentContext(),
    )
    relay = FakeTrialRelay.latest
    assert relay is not None and environment.relay is relay
    assert (relay.opened, relay.ended) == (1, 1)
    assert InstalledAgent.latest is not None
    assert InstalledAgent.latest.calls == [method]


async def test_each_later_agent_phase_reenters_the_one_relay(tmp_path: Path) -> None:
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    environment = _environment(tmp_path)
    await wrapper.run("one", environment, agent_context.AgentContext())
    await wrapper.resume("two", environment, agent_context.AgentContext())
    await wrapper.run("three", environment, agent_context.AgentContext())
    relay = FakeTrialRelay.latest
    assert relay is not None and FakeTrialRelay.created == 1
    assert (relay.opened, relay.resumed, relay.ended) == (1, 2, 3)
    assert InstalledAgent.latest is not None
    assert InstalledAgent.latest.calls == ["run", "resume", "run"]
    assert len(InstalledAgent.latest.mcp_servers) == 1


async def test_a_timeout_while_the_relay_waits_for_its_proxy_leaves_the_verifier_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relays: list[TrialRelay] = []

    class RecordedTrialRelay(TrialRelay):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            relays.append(self)

    def adapter(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    @contextlib.contextmanager
    def bind(environment: object) -> Generator[object]:
        yield environment

    @contextlib.asynccontextmanager
    async def proxy(sandbox: object, **kwargs: object) -> AsyncGenerator[object]:
        del sandbox, kwargs
        yield types.SimpleNamespace(port=13131)

    probing = asyncio.Event()

    async def answers_after_a_while(sandbox: object, *, port: int) -> None:
        del sandbox, port
        probing.set()
        await asyncio.sleep(0.05)

    monkeypatch.setattr(agent, "TrialRelay", RecordedTrialRelay)
    monkeypatch.setattr(trial_relay, "HarborSandboxEnvironment", adapter)
    monkeypatch.setattr(trial_relay, "bind_sandbox_environment", bind)
    monkeypatch.setattr(trial_relay, "sandbox_model_proxy", proxy)
    monkeypatch.setattr(trial_relay, "_wait_for_proxy", answers_after_a_while)
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    wrapper.context_id = uuid4()
    phase = asyncio.create_task(
        wrapper.run("do work", _environment(tmp_path), agent_context.AgentContext())
    )
    _ = await probing.wait()

    _ = phase.cancel()
    with pytest.raises(asyncio.CancelledError):
        await phase

    (relay,) = relays
    assert InstalledAgent.latest is not None and InstalledAgent.latest.calls == []
    assert relay.phase == "verifier"
    assert relay.phase_env()["ANTHROPIC_API_KEY"] == relay.verifier_key
    assert relay.agent_key not in relay.phase_env().values()
    outcome = await relay.close()
    assert outcome.refusal_reason == (
        "the agent made no answered model call through the relay"
    )


async def test_a_runner_loop_resume_keeps_the_host_trajectory(tmp_path: Path) -> None:
    wrapper = _relay_agent(
        tmp_path,
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    )
    environment = _environment(tmp_path)
    await wrapper.resume("do work", environment, agent_context.AgentContext())
    saved = tmp_path / "sandbox-agent-proxy" / "trajectory.json"
    assert saved.read_text(encoding="utf-8") == "resumed trajectory"
    await environment.stop(delete=False)
    assert environment.outcome is not None
    assert environment.outcome.runner_atif == saved


def test_preflight_checks_the_inner_agents_options() -> None:
    loop = {
        "import_path": f"{__name__}:LoopAgent",
        "model_name": "anthropic/claude-test",
    }
    agent.RelayAgent.preflight({"agent": loop, "relay": RELAY}, {})
    with pytest.raises(ValueError, match="Unknown option 'bogus'"):
        agent.RelayAgent.preflight(
            {"agent": {**loop, "kwargs": {"bogus": 1}}, "relay": RELAY}, {}
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"relay": RELAY}, "agent", id="no-agent"),
        pytest.param(
            {"agent": {"import_path": "no_such_module:Agent"}, "relay": RELAY},
            "no_such_module",
            id="unimportable",
        ),
        pytest.param(
            {"agent": {"import_path": f"{__name__}:LoopAgent"}, "relay": {}},
            "declared_models",
            id="no-declared-models",
        ),
    ],
)
def test_preflight_refuses_a_wrapper_it_cannot_build(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        agent.RelayAgent.preflight(kwargs, {})


@pytest.mark.parametrize(
    "wrapped",
    [
        _installed_agent_config(),
        {
            "import_path": f"{__name__}:LoopAgent",
            "model_name": "anthropic/claude-test",
        },
    ],
)
def test_the_inner_agent_gets_the_jobs_agent_env_as_harbor_resolved_it(
    tmp_path: Path, wrapped: dict[str, object]
) -> None:
    _ = agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent=wrapped,
        relay=RELAY,
        extra_env={"TEMPLATED": "${UNSET_TEMPLATE}"},
    )
    import_path = wrapped["import_path"]
    assert isinstance(import_path, str)
    inner = (
        InstalledAgent.latest
        if import_path.endswith("InstalledAgent")
        else LoopAgent.latest
    )
    assert inner is not None
    assert inner.extra_env["TEMPLATED"] == "${UNSET_TEMPLATE}"


async def test_a_failed_runner_loop_phase_keeps_its_trajectory(tmp_path: Path) -> None:
    wrapper = _relay_agent(
        tmp_path,
        {
            "import_path": f"{__name__}:FailingLoopAgent",
            "model_name": "anthropic/claude-test",
        },
    )
    environment = _environment(tmp_path)
    with pytest.raises(RuntimeError, match="the agent's loop failed"):
        await wrapper.run("do work", environment, agent_context.AgentContext())
    saved = tmp_path / "sandbox-agent-proxy" / "trajectory.json"
    assert saved.read_text(encoding="utf-8") == "partial trajectory"
    await environment.stop(delete=False)
    assert environment.outcome is not None
    assert environment.outcome.runner_atif == saved
    assert environment.outcome.runner_atif_sha256 is not None


TRIAL_LOGGER = logging.getLogger("harbor-sandbox-agent-proxy-test-trial")


@pytest.mark.parametrize(
    ("argument", "own", "trials", "read", "expected"),
    [
        pytest.param(
            "skills_dir",
            "/agent/skills",
            "/trial/skills",
            operator.attrgetter("skills_dir"),
            "/trial/skills",
            id="skills_dir",
        ),
        pytest.param(
            "load_trajectory",
            "/agent/trajectory.json",
            "/trial/trajectory.json",
            operator.attrgetter("load_trajectory"),
            Path("/trial/trajectory.json"),
            id="load_trajectory",
        ),
        pytest.param(
            "environment_logs_dir",
            "/agent/logs",
            PurePosixPath("/trial/logs"),
            operator.attrgetter("environment_logs_dir"),
            PurePosixPath("/trial/logs"),
            id="environment_logs_dir",
        ),
        pytest.param(
            "mcp_servers",
            [{"name": "agent-tools", "transport": "stdio", "command": "tools"}],
            [
                task_config.MCPServerConfig(
                    name="task-tools", transport="streamable-http", url="http://t/"
                )
            ],
            operator.attrgetter("mcp_servers"),
            [
                task_config.MCPServerConfig(
                    name="task-tools", transport="streamable-http", url="http://t/"
                )
            ],
            id="mcp_servers",
        ),
        pytest.param(
            "logger",
            "the agent's own logger",
            TRIAL_LOGGER,
            operator.attrgetter("logger.name"),
            f"{TRIAL_LOGGER.name}.{agent_base.__name__}",
            id="logger",
        ),
    ],
)
def test_the_trials_arguments_win_over_the_inner_agents_own_kwargs(
    tmp_path: Path,
    argument: str,
    own: object,
    trials: object,
    read: Callable[[agent_base.BaseAgent], object],
    expected: object,
) -> None:
    from_trial: dict[str, Any] = {argument: trials}
    _ = agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent={"import_path": f"{__name__}:PlainAgent", "kwargs": {argument: own}},
        relay=RELAY,
        **from_trial,
    )
    assert PlainAgent.latest is not None
    assert read(PlainAgent.latest) == expected


@pytest.mark.parametrize(
    ("argument", "own", "read", "expected"),
    [
        pytest.param(
            "skills_dir",
            "/agent/skills",
            operator.attrgetter("skills_dir"),
            "/agent/skills",
            id="skills_dir",
        ),
        pytest.param(
            "load_trajectory",
            "/agent/trajectory.json",
            operator.attrgetter("load_trajectory"),
            Path("/agent/trajectory.json"),
            id="load_trajectory",
        ),
        pytest.param(
            "mcp_servers",
            [{"name": "agent-tools", "transport": "stdio", "command": "tools"}],
            operator.attrgetter("mcp_servers"),
            [{"name": "agent-tools", "transport": "stdio", "command": "tools"}],
            id="mcp_servers",
        ),
    ],
)
def test_the_inner_agents_own_kwarg_stands_when_the_trial_passes_none(
    tmp_path: Path,
    argument: str,
    own: object,
    read: Callable[[agent_base.BaseAgent], object],
    expected: object,
) -> None:
    _ = agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent={"import_path": f"{__name__}:PlainAgent", "kwargs": {argument: own}},
        relay=RELAY,
    )
    assert PlainAgent.latest is not None
    assert read(PlainAgent.latest) == expected
