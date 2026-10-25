"""Harbor environment wrapper that removes host credential values from sandbox commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.import_path import import_class

from .collaborators import DefaultSecretSource, SecretSource
from .registry import resolve_collaborator
from .trial_relay import TrialRelay


class RelayEnvironment(BaseEnvironment):
    _inner: BaseEnvironment
    _secrets: SecretSource
    relay: TrialRelay | None

    """Delegate lifecycle operations while preventing host secrets entering the sandbox."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        environment: dict[str, object],
        relay: object,
        secrets: object | None = None,
        **kwargs: Any,
    ) -> None:
        del relay
        raw_path = environment.get("import_path")
        if not isinstance(raw_path, str):
            raise ValueError("environment requires an import_path")
        raw_kwargs = environment.get("kwargs", {})
        if not isinstance(raw_kwargs, dict):
            raise ValueError("environment kwargs must be an object")
        inner_class = import_class(raw_path, base=BaseEnvironment, label="environment")
        self._inner = inner_class(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **raw_kwargs,
            **kwargs,
        )
        source = (
            DefaultSecretSource({})
            if secrets is None
            else resolve_collaborator(secrets, label="secrets")
        )
        if not isinstance(source, SecretSource):
            raise TypeError("secrets must implement protected_values")
        self._secrets = source
        self.relay = None
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            **kwargs,
        )

    @staticmethod
    @override
    def type() -> str:
        return "sandbox-agent-proxy"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return self._inner.capabilities

    @property
    @override
    def os(self):
        return self._inner.os

    def attach_relay(self, relay: TrialRelay) -> None:
        self.relay = relay
        self._secrets = _RelaySecrets(relay)


    def _validate_definition(self) -> None:
        self._inner._validate_definition()

    async def start(self, force_build: bool) -> None:
        await self._inner.start(force_build)

    async def stop(self, delete: bool) -> None:
        relay, self.relay = self.relay, None
        if relay is not None:
            await relay.close()
        await self._inner.stop(delete)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._inner.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._inner.upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._inner.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._inner.download_dir(source_dir, target_dir)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self._inner.default_user = self.default_user
        return await self._inner.exec(
            command,
            cwd=cwd,
            env=self._confined(env),
            timeout_sec=timeout_sec,
            user=user,
        )

    def _confined(self, env: dict[str, str] | None) -> dict[str, str] | None:
        merged = self._merge_env(env) or {}
        protected = self._secrets.protected_values()
        confined = {name: value for name, value in merged.items() if not any(secret and secret in value for secret in protected)}
        if self.relay is not None:
            for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GEMINI_API_KEY"):
                confined.pop(name, None)
            confined.update(self.relay.phase_env())
        return confined or None


class _RelaySecrets:
    def __init__(self, relay: TrialRelay) -> None:
        self._relay = relay

    def protected_values(self):
        return self._relay.protected_values()
