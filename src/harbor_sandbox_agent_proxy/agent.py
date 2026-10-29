"""Harbor's wrapper agent for sandbox and runner-loop model callers."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast, override
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

from .config import RelayConfig
from .connection import AgentConnection, agent_connection
from .environment import RelayEnvironment
from .outcome import RelayOutcome
from .relay import mint_caller_key
from .trial_relay import TrialRelay

type _AgentPhase = Callable[
    [str, harbor_environment_base.BaseEnvironment, harbor_agent_context.AgentContext],
    Awaitable[None],
]


class RelayAgent(harbor_agent_base.BaseAgent):
    """Wrap a Harbor agent and install the appropriate relay behavior around it."""

    _config: RelayConfig
    _inner_config: harbor_trial_config.AgentConfig
    _connection: AgentConnection | None
    _agent_key: str
    _installed: bool
    _inner: harbor_agent_base.BaseAgent

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
        vars(self)["capabilities"] = inner_class.capabilities
        self._installed = issubclass(
            inner_class, harbor_installed_agent_base.BaseInstalledAgent
        )
        self._connection = None
        self._agent_key = mint_caller_key()
        inner_env = {**self.extra_env}
        inner_kwargs: dict[str, Any] = dict(self._inner_config.kwargs)
        if self._installed:
            self._connection = agent_connection(
                inner_class, self._inner_config.model_name
            )
            inner_env |= self._connection.proxy_env(
                key=self._agent_key, port=self._config.port
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
        if issubclass(inner_class, harbor_installed_agent_base.BaseInstalledAgent):
            _ = agent_connection(inner_class, inner_config.model_name)
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
        environment = _relay_environment(environment)
        if self._connection is None:
            await self._inner.setup(environment)
            return
        environment.caller_keys = environment.caller_keys | {self._agent_key}
        with environment.scoped_agent_setup_env(
            self._connection.proxy_env(key=self._agent_key, port=self._config.port)
        ):
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
        environment = _relay_environment(environment)
        if self._installed:
            environment.caller_keys = environment.caller_keys | {self._agent_key}
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
        environment = _relay_environment(environment)
        relay = environment.relay
        if relay is None:
            relay = await self._open_relay(environment)
        else:
            relay = cast(TrialRelay, relay)
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
        environment = _relay_environment(environment)
        if self._connection is None:
            raise RuntimeError("only installed agents open a sandbox relay")
        sink = await environment.open_sink()
        servers = await environment.bridged_tool_servers()
        relay = TrialRelay(
            trial=environment.trial,
            config=self._config,
            route=environment.route,
            sink=sink,
            connection=self._connection,
            bridged_servers=servers,
            protected=environment.protected_values,
            environment=environment,
            agent_key=self._agent_key,
        )
        environment.attach_relay(relay)
        try:
            await relay.open_agent_phase()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await relay.end_agent_phase()
            raise
        self._inner.mcp_servers.extend(relay.mcp_server_configs())
        self._inner._extra_env.update(relay.phase_env())
        return relay

    async def _save_runner_trajectory(
        self, environment: harbor_environment_base.BaseEnvironment
    ) -> None:
        environment = _relay_environment(environment)
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
        environment.record_runner_loop_outcome(
            RelayOutcome(
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
        )


def _relay_environment(
    environment: harbor_environment_base.BaseEnvironment,
) -> RelayEnvironment:
    if not isinstance(environment, RelayEnvironment):
        raise RuntimeError(
            "the relay environment wrapper is not in place; use both wrapper classes"
        )
    return environment


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
