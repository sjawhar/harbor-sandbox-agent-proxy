"""Regression for host credentials expanded in a provider's startup environment."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from harbor_sandbox_agent_proxy.environment import RelayEnvironment


class StartupRecordingEnvironment(BaseEnvironment):
    """Provider-shaped environment that records the startup environment it receives."""

    latest: ClassVar[StartupRecordingEnvironment | None] = None

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
        del kwargs
        self.seen_startup: dict[str, str] | None = None
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
        )
        type(self).latest = self

    @staticmethod
    @override
    def type() -> str:
        return "startup-recording"

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


async def test_inner_startup_environment_removes_templated_host_key(
    tmp_path: Path, monkeypatch
) -> None:
    host_key = "host-key-not-for-sandbox"
    monkeypatch.setenv("OPENAI_API_KEY", host_key)
    environment = RelayEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="test",
        session_id="test",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(env={"OPENAI_API_KEY": "${OPENAI_API_KEY}"}),
        environment={"import_path": f"{__name__}:StartupRecordingEnvironment"},
        relay={"declared_models": ["test-model"]},
    )

    await environment.start(False)

    assert StartupRecordingEnvironment.latest is not None
    assert StartupRecordingEnvironment.latest.seen_startup == {}
