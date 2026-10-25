"""Security regressions for the verifier tests-tree isolation boundary."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar, override

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig, TaskOS
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from harbor_sandbox_agent_proxy.environment import RelayEnvironment

_STAGE_GLOB = ".sandbox-agent-proxy-tests.*"
_MIRROR_GLOB = ".sandbox-agent-proxy-tests-mirror.*"


@dataclass(frozen=True)
class _Upload:
    target: Path
    stage: Path | None
    stage_mode: int | None
    stage_uid: int | None
    tests_dir_existed: bool


class LocalShellEnvironment(BaseEnvironment):
    """An injected Harbor environment that executes the wrapper's shell protocol locally."""

    latest: ClassVar[LocalShellEnvironment | None] = None

    def __init__(
        self,
        *,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        root: str,
        on_upload: Callable[[Path], None] | None = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.root = Path(root)
        self.on_upload = on_upload
        self.uploads: list[_Upload] = []
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
        return "local-shell"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(dynamic_network_policy=True)

    @property
    @override
    def os(self) -> TaskOS:
        return TaskOS.LINUX

    @override
    def _validate_definition(self) -> None:
        pass

    @override
    async def start(self, force_build: bool) -> None:
        del force_build

    @override
    async def stop(self, delete: bool) -> None:
        del delete

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = self._before_upload(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(source_path, target)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        target = self._before_upload(target_dir)
        if not target.parent.is_dir():
            raise RuntimeError(f"no directory {target.parent} to copy into")
        _ = shutil.copytree(source_dir, target, symlinks=True, dirs_exist_ok=True)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        _ = shutil.copy2(source_path, target_path)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        _ = shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        del timeout_sec, user
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            command,
            cwd=cwd,
            env=os.environ | (env or {}),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode is not None
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )

    def _before_upload(self, target_path: str) -> Path:
        target = Path(target_path)
        stage: Path | None = None
        if target.is_relative_to(self.root):
            top = self.root / target.relative_to(self.root).parts[0]
            stage = None if top == self.root / "tests" else top
        info = stage.stat() if stage is not None and stage.exists() else None
        self.uploads.append(
            _Upload(
                target=target,
                stage=stage,
                stage_mode=None if info is None else stat.S_IMODE(info.st_mode),
                stage_uid=None if info is None else info.st_uid,
                tests_dir_existed=os.path.lexists(self.root / "tests"),
            )
        )
        if self.on_upload is not None:
            self.on_upload(target)
        return target


def _host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_upload: Callable[[Path], None] | None = None,
) -> tuple[RelayEnvironment, Path]:
    root = tmp_path / "root"
    root.mkdir()

    def for_os(_: TaskOS) -> EnvironmentPaths:
        return EnvironmentPaths(tests_dir=PurePosixPath(root, "tests"))

    monkeypatch.setattr(EnvironmentPaths, "for_os", staticmethod(for_os))
    paths = TrialPaths(tmp_path / "trial")
    environment = RelayEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="test",
        session_id="test-session",
        trial_paths=paths,
        task_env_config=EnvironmentConfig(),
        environment={
            "import_path": f"{__name__}:LocalShellEnvironment",
            "kwargs": {"root": str(root), "on_upload": on_upload},
        },
        relay={"declared_models": ["test-model"]},
    )
    environment.default_user = os.geteuid()
    return environment, root


def _inner() -> LocalShellEnvironment:
    inner = LocalShellEnvironment.latest
    assert inner is not None
    return inner


def _tests_source(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    source = tmp_path / name
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(text, encoding="utf-8")
    return source


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_locked(directory: Path, owner: int) -> None:
    for path in (directory, *directory.rglob("*")):
        info = path.lstat()
        assert (path, info.st_uid) == (path, owner)
        if not stat.S_ISLNK(info.st_mode):
            assert (path, oct(info.st_mode & 0o077)) == (path, oct(0))


def _stages(root: Path) -> list[Path]:
    return sorted(root.glob(_STAGE_GLOB))


def _mirrors(root: Path) -> list[Path]:
    return sorted(root.glob(_MIRROR_GLOB))


async def test_the_tests_tree_is_assembled_root_only_and_moved_in_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verifier tree must not be visible before it is locked and renamed."""
    tests = tmp_path / "root" / "tests"

    def staged_only(target: Path) -> None:
        assert target != tests and not target.is_relative_to(tests), (
            f"{target} was uploaded straight into the tests directory, where the "
            + "agent's user can read it until the tree is locked"
        )

    environment, root = _host_environment(tmp_path, monkeypatch, on_upload=staged_only)
    files = {"test.sh": "#!/bin/sh\n", "judges/prompt.md": "Grade it.\n"}

    await environment.upload_dir(_tests_source(tmp_path, "tests", files), str(tests))

    for upload in _inner().uploads:
        assert upload.stage is not None, f"{upload.target} is not in a stage"
        assert upload.stage.parent == root
        assert upload.stage.name.startswith(".sandbox-agent-proxy-tests.")
        assert upload.stage_mode == 0o700
        assert upload.stage_uid == os.geteuid()
        assert not upload.tests_dir_existed
    assert _tree(tests) == files
    _assert_locked(tests, os.geteuid())
    assert _stages(root) == []


async def test_an_untrusted_tests_directory_is_replaced_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent-owned files must not be inherited into the verifier tree."""
    environment, root = _host_environment(tmp_path, monkeypatch)
    tests = root / "tests"
    tests.mkdir()
    _ = (tests / "conftest.py").write_text("planted\n", encoding="utf-8")

    await environment.upload_dir(
        _tests_source(tmp_path, "tests", {"test.sh": "new\n"}), str(tests)
    )

    assert _tree(tests) == {"test.sh": "new\n"}
    _assert_locked(tests, os.geteuid())
    assert _stages(root) == []


async def test_a_second_upload_builds_on_the_root_only_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second verifier upload keeps trusted first-upload files without reading /tests."""
    environment, root = _host_environment(tmp_path, monkeypatch)
    tests = root / "tests"
    await environment.upload_dir(
        _tests_source(tmp_path, "shared", {"test.sh": "shared\n", "helper.py": "x\n"}),
        str(tests),
    )
    await environment.upload_dir(
        _tests_source(tmp_path, "step", {"test.sh": "step\n", "rubric.md": "R\n"}),
        str(tests),
    )

    assert _tree(tests) == {
        "helper.py": "x\n",
        "rubric.md": "R\n",
        "test.sh": "step\n",
    }
    (mirror,) = _mirrors(root)
    assert _tree(mirror / "tests") == _tree(tests)
    assert stat.S_IMODE(mirror.stat().st_mode) == 0o700
    assert _stages(root) == []


async def test_emptying_tests_starts_the_mirror_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later verifier step must not inherit a prior step's tests."""
    environment, root = _host_environment(tmp_path, monkeypatch)
    tests = root / "tests"
    await environment.upload_dir(
        _tests_source(tmp_path, "first", {"test.sh": "first\n", "helper.py": "x\n"}),
        str(tests),
    )

    _ = await environment.empty_dirs([PurePosixPath(tests)], chmod=False)
    await environment.upload_dir(
        _tests_source(tmp_path, "second", {"rubric.md": "R\n"}), str(tests)
    )

    assert _tree(tests) == {"rubric.md": "R\n"}
    (mirror,) = _mirrors(root)
    assert _tree(mirror / "tests") == _tree(tests)


async def test_a_tampered_mirror_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper must not reuse a mirror after its root-only boundary is widened."""
    environment, root = _host_environment(tmp_path, monkeypatch)
    tests = root / "tests"
    await environment.upload_dir(
        _tests_source(tmp_path, "first", {"test.sh": "first\n"}), str(tests)
    )
    (mirror,) = _mirrors(root)
    mirror.chmod(0o755)

    with pytest.raises(RuntimeError, match=r"not the directory only root can enter"):
        await environment.upload_dir(
            _tests_source(tmp_path, "second", {"test.sh": "second\n"}), str(tests)
        )

    assert _tree(tests) == {"test.sh": "first\n"}
    assert _stages(root) == []


async def test_a_symbolic_link_at_tests_is_refused_and_never_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task-controlled symlink must not redirect the verifier upload."""
    environment, root = _host_environment(tmp_path, monkeypatch)
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    _ = (elsewhere / "planted.txt").write_text("planted\n", encoding="utf-8")
    tests = root / "tests"
    tests.symlink_to(elsewhere)

    with pytest.raises(RuntimeError, match=r"symbolic link"):
        await environment.upload_dir(
            _tests_source(tmp_path, "tests", {"test.sh": "#!/bin/sh\n"}), str(tests)
        )

    assert tests.is_symlink() and tests.readlink() == elsewhere
    assert _tree(elsewhere) == {"planted.txt": "planted\n"}
    assert _inner().uploads == []
    assert _stages(root) == []


async def test_a_tests_directory_that_appears_while_staging_is_refused_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent agent write must win the race by failing verification safely."""
    tests = tmp_path / "root" / "tests"

    def plant(target: Path) -> None:
        del target
        if not os.path.lexists(tests):
            tests.mkdir()
            _ = (tests / "planted.txt").write_text("planted\n", encoding="utf-8")

    environment, root = _host_environment(tmp_path, monkeypatch, on_upload=plant)

    with pytest.raises(RuntimeError, match=r"appeared"):
        await environment.upload_dir(
            _tests_source(tmp_path, "tests", {"test.sh": "#!/bin/sh\n"}), str(tests)
        )

    assert _tree(tests) == {"planted.txt": "planted\n"}
    assert _stages(root) == []


async def test_a_writable_tests_parent_is_refused_before_any_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No stage is private if an untrusted user can rename it from its parent."""
    environment, root = _host_environment(tmp_path, monkeypatch)
    root.chmod(0o777)

    with pytest.raises(RuntimeError, match=r"with mode 777; users other than root"):
        await environment.upload_dir(
            _tests_source(tmp_path, "tests", {"test.sh": "#!/bin/sh\n"}),
            str(root / "tests"),
        )

    assert _inner().uploads == []
    assert sorted(root.iterdir()) == []
