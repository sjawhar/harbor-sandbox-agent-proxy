"""Harbor agent wrapper that constructs a per-trial relay from injected collaborators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

from .collaborators import DefaultSecretSource, DefaultUpstream, SecretSource, Upstream
from .config import RelayConfig
from .outcome import JsonOutcomeSink, TrialIdentity
from .registry import resolve_collaborator
from .trial_relay import TrialRelay


class RelayAgent(BaseAgent):
    """Run an inner Harbor agent with a host-mediated sandbox model endpoint."""
    _config: RelayConfig
    _inner_config: AgentConfig
    _upstream: Upstream
    _secrets: SecretSource
    _inner: BaseAgent
    _relay: TrialRelay | None


    def __init__(self, logs_dir: Path, *, agent: object, relay: object, upstream: object | None = None, secrets: object | None = None, **kwargs: Any) -> None:
        super().__init__(logs_dir, **kwargs)
        self._config = RelayConfig.model_validate(relay)
        self._inner_config = AgentConfig.model_validate(agent)
        self._upstream = _upstream(upstream)
        self._secrets = _secrets(secrets, self._upstream)
        self._inner = AgentFactory.create_agent_from_config(
            self._inner_config,
            logs_dir=logs_dir,
            model_name=self._inner_config.model_name,
            extra_env=self.extra_env,
        )
        self._relay: TrialRelay | None = None

    @staticmethod
    @override
    def name() -> str:
        return "sandbox-agent-proxy"

    @override
    def version(self) -> str | None:
        return self._inner.version()

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        await self._inner.setup(environment)

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self._run_phase(self._inner.run, instruction, environment, context)

    @override
    async def resume(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self._run_phase(self._inner.resume, instruction, environment, context)

    @override
    async def load(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self._run_phase(self._inner.load, instruction, environment, context)

    async def _run_phase(self, method: Any, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        relay = self._relay
        if relay is None:
            identity = TrialIdentity(
                name=environment.trial_paths.trial_dir.name,
                directory=environment.trial_paths.trial_dir,
            )
            relay = TrialRelay(
                trial=identity,
                upstream=self._upstream,
                sink=JsonOutcomeSink(identity, protected_values=self._secrets.protected_values()),
                declared_models=frozenset(self._config.declared_models),
                secrets=self._secrets,
                port=self._config.port,
            )
            attach = getattr(environment, "attach_relay", None)
            if attach is None:
                raise RuntimeError("RelayAgent requires RelayEnvironment")
            attach(relay)
            self._relay = relay
            await relay.open_agent_phase()
            self._inner._extra_env.update(relay.phase_env())
        else:
            await relay.resume_agent_phase()
            self._inner._extra_env.update(relay.phase_env())
        try:
            await method(instruction, environment, context)
        finally:
            await relay.end_agent_phase()


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
