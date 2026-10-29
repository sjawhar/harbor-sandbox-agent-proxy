"""Regressions for credential confinement across Harbor environment paths."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, override

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from harbor_sandbox_agent_proxy.environment import (
    RelayEnvironment,
    RunnerSecretExposureError,
)


class RecordingEnvironment(BaseEnvironment):
    """Inner environment that exposes the configuration and sandbox env it sees."""

    latest: ClassVar[RecordingEnvironment | None] = None

    def __init__(
        self,
        *,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        **kwargs: object,
    ) -> None:
        raw_persistent_env = kwargs.get("persistent_env")
        persistent_env = (
            {
                name: value
                for name, value in raw_persistent_env.items()
                if isinstance(name, str) and isinstance(value, str)
            }
            if isinstance(raw_persistent_env, dict)
            else None
        )
        self.received_task_env = dict(task_env_config.env)
        self.received_persistent_env = persistent_env
        self.seen_startup: dict[str, str] | None = None
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            persistent_env=persistent_env,
        )
        type(self).latest = self

    @staticmethod
    @override
    def type() -> str:
        return "recording"

    @override
    def _validate_definition(self) -> None:
        pass

    @override
    async def start(self, force_build: bool) -> None:
        del force_build
        self.seen_startup = self._startup_env()

    @override
    async def stop(self, delete: bool) -> None:
        del delete

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        del source_path, target_path

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        del source_dir, target_dir

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        del source_path, target_path

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        del source_dir, target_dir

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        del command, cwd, env, timeout_sec, user
        return ExecResult(return_code=0)


def _environment(
    tmp_path: Path,
    *,
    task_env: dict[str, str] | None = None,
    persistent_env: dict[str, str] | None = None,
) -> RelayEnvironment:
    kwargs: dict[str, object] = {}
    if persistent_env is not None:
        kwargs["persistent_env"] = persistent_env
    return RelayEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="test",
        session_id="test",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(env=task_env or {}),
        environment={"import_path": f"{__name__}:RecordingEnvironment"},
        relay={"declared_models": ["test-model"]},
        **kwargs,
    )


async def test_template_task_env_is_resolved_then_refused_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_key = "host-key-not-for-sandbox"
    monkeypatch.setenv("OPENAI_API_KEY", host_key)
    environment = _environment(
        tmp_path, task_env={"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    )

    assert RecordingEnvironment.latest is not None
    assert RecordingEnvironment.latest.received_task_env == {}

    with pytest.raises(RunnerSecretExposureError, match="OPENAI_API_KEY") as error:
        await environment.start(False)

    assert host_key not in str(error.value)


async def test_persistent_env_is_resolved_then_refused_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_key = "host-key-not-for-sandbox"
    monkeypatch.setenv("OPENAI_API_KEY", host_key)
    environment = _environment(
        tmp_path,
        persistent_env={"JOB_CONFIG_KEY": "${OPENAI_API_KEY}"},
    )

    assert RecordingEnvironment.latest is not None
    assert RecordingEnvironment.latest.received_persistent_env == {}

    with pytest.raises(RunnerSecretExposureError, match="JOB_CONFIG_KEY") as error:
        await environment.start(False)

    assert host_key not in str(error.value)
