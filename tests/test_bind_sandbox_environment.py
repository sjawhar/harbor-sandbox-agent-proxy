"""Tests for ``bind_sandbox_environment``."""

from __future__ import annotations

from typing import Any, final, override

import pytest
from inspect_ai.util import (
    ExecResult,
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    sandbox,
    sandbox_with,
)

from harbor_sandbox_agent_proxy.sandbox_binding import bind_sandbox_environment


@final
class FakeSandboxEnvironment(SandboxEnvironment):
    """A minimal ``SandboxEnvironment`` a caller might construct itself."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.files: dict[str, str | bytes] = {}

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        if cmd[-3:-1] == ["test", "-r"]:
            exists = cmd[-1] in self.files
            return ExecResult(
                success=exists, returncode=0 if exists else 1, stdout="", stderr=""
            )
        return ExecResult(success=True, returncode=0, stdout=self.label, stderr="")

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        self.files[file] = contents

    @override
    async def read_file(self, file: str, text: bool = True) -> Any:
        if file not in self.files:
            raise FileNotFoundError(file)
        return self.files[file]

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        raise NotImplementedError

    @classmethod
    @override
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        pass


def test_sandbox_raises_outside_any_binding() -> None:
    # red: with no binding in effect, sandbox() has nothing to resolve
    with pytest.raises(ProcessLookupError):
        sandbox()


def test_bind_sandbox_environment_resolves_via_sandbox() -> None:
    environment = FakeSandboxEnvironment("bound")
    with bind_sandbox_environment(environment) as bound:
        # the yielded value is usable directly (through as_type, same as any proxy)
        assert bound.as_type(FakeSandboxEnvironment).label == "bound"
        # and it is what sandbox() with no name, "default", or the sole name resolves
        assert sandbox().as_type(FakeSandboxEnvironment).label == "bound"
        assert sandbox("default").as_type(FakeSandboxEnvironment).label == "bound"

    # the binding does not outlive the `with` block
    with pytest.raises(ProcessLookupError):
        sandbox()


def test_bind_sandbox_environment_custom_name_is_also_the_default() -> None:
    environment = FakeSandboxEnvironment("harbor")
    with bind_sandbox_environment(environment, name="harbor-trial"):
        # resolvable by its bound name
        assert sandbox("harbor-trial").as_type(FakeSandboxEnvironment).label == "harbor"
        # and, as the block's only sandbox, by sandbox() with no argument too
        assert sandbox().as_type(FakeSandboxEnvironment).label == "harbor"
        # sandbox() with a sole bound environment resolves any name to it (existing
        # sandbox() behaviour -- not specific to bind_sandbox_environment); name
        # mismatches only raise once more than one environment is bound.
        assert (
            sandbox("anything-else").as_type(FakeSandboxEnvironment).label == "harbor"
        )


def test_bind_sandbox_environment_records_events_when_active() -> None:
    """Commands run through the binding are wrapped for event recording.

    ``bind_sandbox_environment`` must wrap with ``SandboxEnvironmentProxy``, the same
    wrapper ``init_sandbox_environments_sample`` applies to a task's own sandboxes, so
    that sandbox commands inside an active sample transcript produce the same
    ``SandboxEvent``s a task-declared sandbox would. ``no_events()`` is the proxy's own
    documented escape hatch; a plain (unwrapped) environment has no such method.
    """
    environment = FakeSandboxEnvironment("events")
    with bind_sandbox_environment(environment) as bound:
        # only a SandboxEnvironmentProxy exposes no_events(); a bare environment does not
        assert hasattr(bound, "no_events")
        assert not hasattr(environment, "no_events")


async def test_bind_sandbox_environment_sequential_rebinding_is_clean() -> None:
    """Two non-nested bindings in sequence must not leak state between them."""
    first = FakeSandboxEnvironment("first")
    with bind_sandbox_environment(first):
        assert sandbox().as_type(FakeSandboxEnvironment).label == "first"
        await sandbox().write_file("f.txt", "data")

    second = FakeSandboxEnvironment("second")
    with bind_sandbox_environment(second):
        assert sandbox().as_type(FakeSandboxEnvironment).label == "second"
        # the second binding's file state is independent of the first's
        assert "f.txt" not in second.files


async def test_sandbox_with_resolves_the_bound_environment() -> None:
    environment = FakeSandboxEnvironment("with")
    with bind_sandbox_environment(environment):
        await sandbox().write_file("marker.txt", "present")
        found = await sandbox_with("marker.txt")
        assert found is not None
        assert found.as_type(FakeSandboxEnvironment).label == "with"
        assert await sandbox_with("missing.txt") is None
