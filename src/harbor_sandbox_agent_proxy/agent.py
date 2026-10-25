"""Harbor's wrapper agent for sandbox and runner-loop model callers."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, override
from uuid import UUID

import anyio
import harbor.agents.base as harbor_agent_base
import harbor.agents.factory as harbor_agent_factory
import harbor.agents.installed.base as harbor_installed_agent_base
import harbor.environments.base as harbor_environment_base
import harbor.models.agent.context as harbor_agent_context
import harbor.models.task.config as harbor_task_config
import harbor.models.trial.config as harbor_trial_config
import harbor.models.trial.result as harbor_trial_result
import pydantic

from .collaborators import (
    BridgedToolServerProvider,
    DefaultBridgedToolServerProvider,
    DefaultSecretSource,
    DefaultUpstream,
    SecretSource,
    Upstream,
)
from .config import RelayConfig
from .outcome import EventSinkFactory, JsonOutcomeSink, RelayOutcome, TrialIdentity
from .registry import resolve_collaborator
from .relay import mint_caller_key
from .trial_relay import TrialRelay

RUNNER_LOOP_OUTCOMES: dict[str, RelayOutcome] = {}
"""Runner-owned trajectories from agents whose model loop stays in the host process."""

type _AgentPhase = Callable[
    [str, harbor_environment_base.BaseEnvironment, harbor_agent_context.AgentContext],
    Awaitable[None],
]


class RelayAgent(harbor_agent_base.BaseAgent):
    """Wrap a Harbor agent and install the appropriate relay behavior around it."""

    options_model: ClassVar[type[Any] | None] = None
    _config: RelayConfig
    _inner_config: harbor_trial_config.AgentConfig
    _upstream: Upstream
    _secrets: SecretSource
    _sink_factory: EventSinkFactory | None
    _bridged_tool_server_provider: BridgedToolServerProvider
    _agent_key: str
    _installed: bool
    _inner: harbor_agent_base.BaseAgent
    _relay: TrialRelay | None

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: Any | None = None,
        mcp_servers: list[harbor_task_config.MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *,
        extra_env: dict[str, str] | None = None,
        load_trajectory: str | Path | None = None,
        environment_logs_dir: PurePosixPath | None = None,
        agent: object,
        relay: object,
        upstream: object | None = None,
        secrets: object | None = None,
        event_sink_factory: object | None = None,
        bridged_tool_server_provider: object | None = None,
        override_setup_timeout_sec: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
            load_trajectory=load_trajectory,
            environment_logs_dir=environment_logs_dir,
            **kwargs,
        )
        self._config = RelayConfig.model_validate(relay)
        self._inner_config, inner_class = _inner_agent(agent)
        self._upstream = _upstream(upstream)
        self._secrets = _secrets(secrets, self._upstream)
        self._sink_factory = _sink_factory(event_sink_factory)
        self._bridged_tool_server_provider = _bridged_tool_server_provider(
            bridged_tool_server_provider
        )
        vars(self)["capabilities"] = inner_class.capabilities
        self._installed = issubclass(
            inner_class, harbor_installed_agent_base.BaseInstalledAgent
        )
        self._agent_key = mint_caller_key()
        inner_env = {**self.extra_env}
        inner_kwargs: dict[str, Any] = dict(self._inner_config.kwargs)
        if self._installed:
            inner_env |= _agent_proxy_environment(
                key=self._agent_key,
                port=self._config.port,
                model_name=self._inner_config.model_name,
            )
        elif self._inner_config.model_name is not None:
            inner_kwargs["api_base"] = runner_loop_api_base(
                inner_class, self._inner_config, self._upstream
            )
        if override_setup_timeout_sec is not None:
            inner_kwargs["override_setup_timeout_sec"] = override_setup_timeout_sec
        framework: dict[str, Any] = {
            "logs_dir": logs_dir,
            "model_name": self._inner_config.model_name,
            "logger": logger,
            "extra_env": inner_env,
        }
        framework |= {
            key: value
            for key, value in (
                ("mcp_servers", None if mcp_servers is None else list(mcp_servers)),
                ("skills_dir", skills_dir),
                ("load_trajectory", load_trajectory),
                ("environment_logs_dir", environment_logs_dir),
            )
            if value is not None
        }
        self._inner = inner_class(**{**inner_kwargs, **framework})
        self._relay = None
        self._session_id: str | None = None
        self._context_id: UUID | None = None

    @staticmethod
    @override
    def name() -> str:
        return "sandbox-agent-proxy"

    @classmethod
    @override
    def preflight(
        cls,
        kwargs: dict[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Check that Harbor can construct the wrapped agent before any trial starts."""
        options = dict(kwargs or {})
        inner_config, inner_class = _inner_agent(options.get("agent"))
        _ = RelayConfig.model_validate(options.get("relay"))
        if _calls_a_model(inner_class, inner_config):
            if not issubclass(
                inner_class, harbor_installed_agent_base.BaseInstalledAgent
            ):
                runner_loop_api_base(
                    inner_class,
                    inner_config,
                    _upstream(options.get("upstream")),
                )
        inner_class.preflight(kwargs=inner_config.kwargs, env=env)

    @override
    def version(self) -> str | None:
        return self._inner.version()

    @override
    def to_agent_info(self) -> harbor_trial_result.AgentInfo:
        return self._inner.to_agent_info()

    @property
    @override
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value
        if hasattr(self, "_inner"):
            self._inner.session_id = value

    @property
    @override
    def context_id(self) -> UUID | None:
        return self._context_id

    @context_id.setter
    def context_id(self, value: UUID | None) -> None:
        self._context_id = value
        if hasattr(self, "_inner"):
            self._inner.context_id = value

    @override
    async def setup(self, environment: harbor_environment_base.BaseEnvironment) -> None:
        await self._inner.setup(environment)

    @override
    async def run(
        self,
        instruction: str,
        environment: harbor_environment_base.BaseEnvironment,
        context: harbor_agent_context.AgentContext,
    ) -> None:
        await self._agent_phase(self._inner.run, instruction, environment, context)

    @override
    async def resume(
        self,
        instruction: str,
        environment: harbor_environment_base.BaseEnvironment,
        context: harbor_agent_context.AgentContext,
    ) -> None:
        await self._agent_phase(self._inner.resume, instruction, environment, context)

    @override
    async def load(
        self,
        instruction: str,
        environment: harbor_environment_base.BaseEnvironment,
        context: harbor_agent_context.AgentContext,
    ) -> None:
        await self._agent_phase(self._inner.load, instruction, environment, context)

    @override
    def populate_context_post_run(
        self, context: harbor_agent_context.AgentContext
    ) -> None:
        self._inner.populate_context_post_run(context)

    async def _agent_phase(
        self,
        phase: _AgentPhase,
        instruction: str,
        environment: harbor_environment_base.BaseEnvironment,
        context: harbor_agent_context.AgentContext,
    ) -> None:
        if self._installed:
            await self._relayed_phase(phase, instruction, environment, context)
            return
        try:
            await phase(instruction, environment, context)
        finally:
            with anyio.CancelScope(shield=True):
                await self._save_runner_trajectory(environment)

    async def _relayed_phase(
        self,
        phase: _AgentPhase,
        instruction: str,
        environment: harbor_environment_base.BaseEnvironment,
        context: harbor_agent_context.AgentContext,
    ) -> None:
        relay = self._relay
        if relay is None:
            relay = await self._open_relay(environment)
        else:
            await relay.resume_agent_phase()
            self._inner._extra_env.update(relay.phase_env())
        inner_error: BaseException | None = None
        try:
            await phase(instruction, environment, context)
        except BaseException as error:
            inner_error = error
            raise
        finally:
            with anyio.CancelScope(shield=True):
                await relay.end_agent_phase()
                if relay.record_error is not None and inner_error is None:
                    raise RuntimeError(
                        f"the relay could not record a model call: {relay.record_error!r}"
                    ) from relay.record_error

    async def _open_relay(
        self, environment: harbor_environment_base.BaseEnvironment
    ) -> TrialRelay:
        """Start the per-trial relay on the first installed-agent phase."""
        attach = getattr(environment, "attach_relay", None)
        trial_paths = getattr(environment, "trial_paths", None)
        if attach is None or trial_paths is None:
            raise RuntimeError(
                "the relay environment wrapper is not in place; use both wrapper classes"
            )
        identity = TrialIdentity(
            name=trial_paths.trial_dir.name, directory=trial_paths.trial_dir
        )
        sink = (
            JsonOutcomeSink(identity, protected_values=self._secrets.protected_values())
            if self._sink_factory is None
            else await self._sink_factory.open(identity)
        )
        relay = TrialRelay(
            trial=identity,
            config=self._config,
            upstream=self._upstream,
            sink=sink,
            bridged_servers=await self._bridged_tool_server_provider.servers_for(
                identity.name
            ),
            secrets=self._secrets,
            environment=environment,
            agent_model=self._inner.model_name,
            agent_key=self._agent_key,
        )
        attach(relay)
        try:
            await relay.open_agent_phase()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await relay.end_agent_phase()
            raise
        self._inner.mcp_servers.extend(relay.mcp_server_configs())
        self._inner._extra_env.update(relay.phase_env())
        self._relay = relay
        return relay

    async def _save_runner_trajectory(
        self, environment: harbor_environment_base.BaseEnvironment
    ) -> None:
        trial_name = environment.trial_paths.trial_dir.name
        source = self.logs_dir / "trajectory.json"
        target = self.logs_dir.parent / "sandbox-agent-proxy" / "trajectory.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copyfile(source, target)
            runner_atif: Path | None = target
            runner_atif_sha256: str | None = _sha256(target)
        else:
            runner_atif = None
            runner_atif_sha256 = None
        RUNNER_LOOP_OUTCOMES[trial_name] = RelayOutcome(
            mode="runner-loop",
            calls=[],
            events=[],
            conversation=None,
            answered_agent_calls=0,
            counts={
                "agent_calls": 0,
                "verifier_calls": 0,
                "failed_calls": 0,
                "refused_calls": 0,
            },
            dropped_betas=frozenset(),
            refusal_reason=None,
            proxy_exit=None,
            runner_atif=runner_atif,
            runner_atif_sha256=runner_atif_sha256,
            record_error=None,
        )


def _upstream(value: object | None) -> Upstream:
    candidate = (
        DefaultUpstream()
        if value is None
        else resolve_collaborator(value, label="upstream")
    )
    if not isinstance(candidate, Upstream):
        raise TypeError("upstream must implement resolve")
    return candidate


def _secrets(value: object | None, upstream: Upstream) -> SecretSource:
    candidate = (
        DefaultSecretSource(upstream.environment)
        if value is None and isinstance(upstream, DefaultUpstream)
        else value
    )
    if candidate is None:
        raise ValueError("a non-default upstream requires a secrets collaborator")
    resolved = resolve_collaborator(candidate, label="secrets")
    if not isinstance(resolved, SecretSource):
        raise TypeError("secrets must implement protected_values")
    return resolved


def _sink_factory(value: object | None) -> EventSinkFactory | None:
    if value is None:
        return None
    candidate = resolve_collaborator(value, label="event_sink_factory")
    if not isinstance(candidate, EventSinkFactory):
        raise TypeError("event_sink_factory must implement open")
    return candidate


def _bridged_tool_server_provider(value: object | None) -> BridgedToolServerProvider:
    candidate = (
        DefaultBridgedToolServerProvider()
        if value is None
        else resolve_collaborator(value, label="bridged_tool_server_provider")
    )
    if not isinstance(candidate, BridgedToolServerProvider):
        raise TypeError("bridged_tool_server_provider must implement servers_for")
    return candidate


def _calls_a_model(
    agent_class: type[harbor_agent_base.BaseAgent],
    config: harbor_trial_config.AgentConfig,
) -> bool:
    return (
        issubclass(agent_class, harbor_installed_agent_base.BaseInstalledAgent)
        or config.model_name is not None
    )


_OBJECT_MAPPING: Final[pydantic.TypeAdapter[dict[str, object]]] = pydantic.TypeAdapter(
    dict[str, object]
)


def _inner_agent(
    agent: object,
) -> tuple[harbor_trial_config.AgentConfig, type[harbor_agent_base.BaseAgent]]:
    try:
        mapping = _OBJECT_MAPPING.validate_python(agent)
    except pydantic.ValidationError as error:
        raise ValueError(
            "the relay agent needs the wrapped agent config in its 'agent' kwargs"
        ) from error
    config = harbor_trial_config.AgentConfig.model_validate(mapping)
    try:
        agent_class = harbor_agent_factory.AgentFactory.get_agent_class_from_config(
            config
        )
    except TypeError as error:
        raise ValueError(str(error)) from error
    return config, agent_class


def runner_loop_api_base(
    agent_class: type[harbor_agent_base.BaseAgent],
    config: harbor_trial_config.AgentConfig,
    upstream: Upstream,
) -> str:
    """Resolve the host-side API base for a runner-loop agent's model calls."""
    model_name = config.model_name
    if model_name is None:
        raise ValueError(f"{agent_class.name()} names no model for its API base")
    options = agent_class.options_model
    if options is None or "api_base" not in options.model_fields:
        raise ValueError(
            f"{agent_class.name()} names model {model_name} but has no api_base option"
        )
    resolver = getattr(upstream, "api_base", None)
    if not callable(resolver):
        raise ValueError(
            f"{agent_class.name()} needs an upstream collaborator with api_base(model=...)"
        )
    base = resolver(model=model_name)
    if not isinstance(base, str) or not base:
        raise ValueError(
            f"{agent_class.name()} received no API base for model {model_name}"
        )
    return base


def _agent_proxy_environment(
    *, key: str, port: int, model_name: str | None
) -> dict[str, str]:
    environment = {
        "ANTHROPIC_API_KEY": key,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_BASE_URL": f"http://localhost:{port}",
        "OPENAI_API_KEY": key,
        "OPENAI_BASE_URL": f"http://localhost:{port}/v1",
        "OPENAI_API_BASE": f"http://localhost:{port}/v1",
        "GEMINI_API_KEY": key,
        "GOOGLE_API_KEY": key,
        "GOOGLE_GENERATIVE_AI_API_KEY": key,
        "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{port}",
    }
    if model_name is not None:
        model = model_name.partition("/")[2] or model_name
        environment["ANTHROPIC_SMALL_FAST_MODEL"] = model
        environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    return environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
