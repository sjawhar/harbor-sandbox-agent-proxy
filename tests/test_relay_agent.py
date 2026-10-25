"""Contracts for Harbor's wrapper around installed and runner-loop agents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar, cast, override

import harbor.agents.base as agent_base
import harbor.agents.capabilities as agent_capabilities
import harbor.agents.installed.base as installed_agent_base
import harbor.agents.options as agent_options
import harbor.environments.base as environment_base
import harbor.models.agent.context as agent_context
import harbor.models.task.config as task_config
import harbor.models.trial.paths as trial_paths
import pytest

from harbor_sandbox_agent_proxy import agent
from harbor_sandbox_agent_proxy.collaborators import RequestDialect, UpstreamTarget
from harbor_sandbox_agent_proxy.config import RelayConfig

RELAY: dict[str, object] = {
    "polling_interval": 0.25,
    "allowed_betas": ["claude-code-20250219"],
    "declared_models": ["anthropic/claude-test"],
}
RESUMABLE = agent_capabilities.AgentCapabilities(resume=True, load_atif_trajectory=True)


class _Upstream:
    async def resolve(self, *, dialect: RequestDialect, model: str) -> UpstreamTarget:
        del dialect, model
        return UpstreamTarget(
            "https://upstream.test/v1/messages", {"x-api-key": "host-key"}
        )

    def api_base(self, *, model: str) -> str:
        assert model == "anthropic/claude-test"
        return "https://upstream.test/anthropic"


class _Secrets:
    def protected_values(self) -> frozenset[str]:
        return frozenset({"host-key"})


class InstalledAgent(installed_agent_base.BaseInstalledAgent):
    latest: ClassVar[InstalledAgent | None] = None
    capabilities: ClassVar[agent_capabilities.AgentCapabilities] = RESUMABLE
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
        raise RuntimeError("the agent loop failed")


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


class FakeRelayEnvironment:
    trial_paths: trial_paths.TrialPaths
    attached: FakeTrialRelay | None

    def __init__(self, trial_dir: Path) -> None:
        self.trial_paths = trial_paths.TrialPaths(trial_dir)
        self.attached = None

    def attach_relay(self, relay: FakeTrialRelay) -> None:
        self.attached = relay


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


@pytest.fixture(autouse=True)
def relay_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "TrialRelay", FakeTrialRelay)
    monkeypatch.setattr(agent, "RUNNER_LOOP_OUTCOMES", {})
    FakeTrialRelay.next_record_error = None
    FakeTrialRelay.created = 0


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
        upstream=_Upstream(),
        secrets=_Secrets(),
        extra_env={"UNMANAGED": "kept", "ANTHROPIC_API_KEY": "untrusted-job-key"},
        **kwargs,
    )


def _environment(tmp_path: Path) -> environment_base.BaseEnvironment:
    return cast(
        environment_base.BaseEnvironment,
        cast(object, FakeRelayEnvironment(tmp_path / "trial")),
    )


async def test_installed_agent_gets_proxy_environment_and_bridged_servers_only_for_run(
    tmp_path: Path,
) -> None:
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    assert InstalledAgent.latest is not None
    inner = InstalledAgent.latest
    assert inner.extra_env["UNMANAGED"] == "kept"
    assert inner.extra_env["ANTHROPIC_API_KEY"] != "untrusted-job-key"
    assert inner.extra_env["ANTHROPIC_BASE_URL"] == "http://localhost:13131"
    assert inner.extra_env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-test"
    environment = FakeRelayEnvironment(tmp_path / "trial")
    await wrapper.run(
        "do work",
        cast(environment_base.BaseEnvironment, cast(object, environment)),
        agent_context.AgentContext(),
    )
    assert environment.attached is FakeTrialRelay.latest
    assert FakeTrialRelay.latest is not None and FakeTrialRelay.latest.opened
    assert FakeTrialRelay.latest.ended
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


async def test_runner_loop_agent_gets_gateway_api_base_and_preserves_host_trajectory(
    tmp_path: Path,
) -> None:
    wrapper = _relay_agent(
        tmp_path,
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    )
    await wrapper.run("do work", _environment(tmp_path), agent_context.AgentContext())
    assert LoopAgent.latest is not None
    assert isinstance(LoopAgent.latest.options, LoopOptions)
    assert LoopAgent.latest.options.api_base == "https://upstream.test/anthropic"
    saved = tmp_path / "sandbox-agent-proxy" / "trajectory.json"
    assert saved.read_text(encoding="utf-8") == "runner trajectory"
    outcome = agent.RUNNER_LOOP_OUTCOMES["trial"]
    assert outcome.mode == "runner-loop"
    assert outcome.runner_atif == saved
    assert outcome.runner_atif_sha256 is not None


def test_runner_loop_agent_without_api_base_is_refused_when_it_names_a_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="anthropic/claude-test"):
        _relay_agent(
            tmp_path,
            {
                "import_path": f"{__name__}:LoopWithoutApiBase",
                "model_name": "anthropic/claude-test",
            },
        )


def test_runner_loop_agent_without_model_does_not_need_an_api_base(
    tmp_path: Path,
) -> None:
    wrapper = _relay_agent(tmp_path, {"import_path": f"{__name__}:LoopWithoutApiBase"})
    assert wrapper.version() == "test"


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
    "relay", [{}, {**RELAY, "polling_interval": 0}, {**RELAY, "gateway": "x"}]
)
def test_a_wrapped_agent_refuses_relay_settings_it_cannot_parse(
    tmp_path: Path, relay: dict[str, object]
) -> None:
    with pytest.raises(Exception):
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


def test_a_wrapped_agents_caller_key_is_registered_for_redaction(
    tmp_path: Path,
) -> None:
    _ = _relay_agent(tmp_path, _installed_agent_config())
    assert InstalledAgent.latest is not None
    key = InstalledAgent.latest.extra_env["ANTHROPIC_API_KEY"]
    assert key != "untrusted-job-key"


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
    environment = FakeRelayEnvironment(tmp_path / "trial")
    await getattr(wrapper, method)(
        "do work",
        cast(environment_base.BaseEnvironment, cast(object, environment)),
        agent_context.AgentContext(),
    )
    relay = FakeTrialRelay.latest
    assert relay is not None and environment.attached is relay
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
    class WaitingRelay(FakeTrialRelay):
        async def open_agent_phase(self) -> None:
            await asyncio.Event().wait()

    monkeypatch.setattr(agent, "TrialRelay", WaitingRelay)
    wrapper = _relay_agent(tmp_path, _installed_agent_config())
    phase = asyncio.create_task(
        wrapper.run("do work", _environment(tmp_path), agent_context.AgentContext())
    )
    await asyncio.sleep(0)
    _ = phase.cancel()
    with pytest.raises(asyncio.CancelledError):
        await phase
    assert WaitingRelay.latest is not None
    assert WaitingRelay.latest.phase == "verifier"


async def test_a_runner_loop_resume_keeps_the_host_trajectory(tmp_path: Path) -> None:
    wrapper = _relay_agent(
        tmp_path,
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    )
    await wrapper.resume(
        "do work", _environment(tmp_path), agent_context.AgentContext()
    )
    saved = tmp_path / "sandbox-agent-proxy" / "trajectory.json"
    assert saved.read_text(encoding="utf-8") == "resumed trajectory"
    assert agent.RUNNER_LOOP_OUTCOMES["trial"].runner_atif == saved


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
    "kwargs",
    [
        {"relay": RELAY},
        {"agent": {"import_path": "no_such_module:Agent"}, "relay": RELAY},
        {"agent": {"import_path": f"{__name__}:LoopAgent"}, "relay": {}},
    ],
)
def test_preflight_refuses_a_wrapper_it_cannot_build(kwargs: dict[str, object]) -> None:
    with pytest.raises(Exception):
        agent.RelayAgent.preflight(kwargs, {})


def test_a_model_free_runner_loop_agent_needs_no_llm_proxy(tmp_path: Path) -> None:
    loop: dict[str, object] = {"import_path": f"{__name__}:LoopWithoutApiBase"}
    agent.RelayAgent.preflight({"agent": loop, "relay": RELAY}, {})
    wrapper = _relay_agent(tmp_path, loop)
    assert wrapper.version() == "test"


@pytest.mark.parametrize(
    "wrapped",
    [
        _installed_agent_config(model_name=None),
        {"import_path": f"{__name__}:LoopAgent", "model_name": "anthropic/claude-test"},
    ],
)
def test_preflight_refuses_an_agent_that_calls_a_model_with_no_llm_proxy(
    wrapped: dict[str, object],
) -> None:
    agent.RelayAgent.preflight({"agent": wrapped, "relay": RELAY}, {})


def test_preflight_refuses_a_runner_loop_agent_hawk_cannot_route() -> None:
    class NoApiBaseUpstream:
        async def resolve(
            self, *, dialect: RequestDialect, model: str
        ) -> UpstreamTarget:
            del dialect, model
            return UpstreamTarget("https://upstream.test", {})

    config = agent.harbor_trial_config.AgentConfig.model_validate(
        {"import_path": f"{__name__}:LoopAgent", "model_name": "claude-test"}
    )
    with pytest.raises(ValueError, match="api_base"):
        _ = agent.runner_loop_api_base(LoopAgent, config, NoApiBaseUpstream())


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
        upstream=_Upstream(),
        secrets=_Secrets(),
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
    with pytest.raises(RuntimeError, match="agent loop failed"):
        await wrapper.run(
            "do work", _environment(tmp_path), agent_context.AgentContext()
        )
    saved = tmp_path / "sandbox-agent-proxy" / "trajectory.json"
    assert saved.read_text(encoding="utf-8") == "partial trajectory"
    assert agent.RUNNER_LOOP_OUTCOMES["trial"].runner_atif == saved


@pytest.mark.parametrize(
    "argument, own, trials",
    [
        ("skills_dir", "/agent/skills", "/trial/skills"),
        ("load_trajectory", "/agent/trajectory.json", "/trial/trajectory.json"),
    ],
)
def test_the_trials_arguments_win_over_the_inner_agents_own_kwargs(
    tmp_path: Path, argument: str, own: str, trials: str
) -> None:
    settings = {
        "logs_dir": tmp_path / "logs",
        "agent": {
            "import_path": f"{__name__}:PlainAgent",
            "kwargs": {argument: own},
        },
        "relay": RELAY,
        "upstream": _Upstream(),
        "secrets": _Secrets(),
    }
    if argument == "skills_dir":
        _ = agent.RelayAgent(**settings, skills_dir=trials)
    else:
        _ = agent.RelayAgent(**settings, load_trajectory=trials)
    assert PlainAgent.latest is not None
    assert getattr(PlainAgent.latest, argument) == (
        Path(trials) if argument == "load_trajectory" else trials
    )


@pytest.mark.parametrize(
    "argument, own",
    [("skills_dir", "/agent/skills"), ("load_trajectory", "/agent/trajectory.json")],
)
def test_the_inner_agents_own_kwarg_stands_when_the_trial_passes_none(
    tmp_path: Path, argument: str, own: str
) -> None:
    _ = agent.RelayAgent(
        logs_dir=tmp_path / "logs",
        agent={"import_path": f"{__name__}:PlainAgent", "kwargs": {argument: own}},
        relay=RELAY,
        upstream=_Upstream(),
        secrets=_Secrets(),
    )
    assert PlainAgent.latest is not None
    assert getattr(PlainAgent.latest, argument) == (
        Path(own) if argument == "load_trajectory" else own
    )
