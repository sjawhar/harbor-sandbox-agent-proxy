"""The Inspect sandbox the relay hands to Inspect, driven over a local Harbor environment.

``HarborSandboxEnvironment`` is the only way Inspect's sandbox tools, the in-sandbox model
proxy and a bridged MCP server reach a Harbor trial, so each mismatch it bridges is
exercised for real here: commands run in a local ``sh`` through the adapter's own wrapper,
stdin really goes inline or through a staged file, and files really move. What differs
between Harbor's providers (stream merging, how a timeout is reported, what a missing file
raises) is chosen per test on ``LocalHarborEnvironment``. The adapter types Harbor as a
protocol and needs no Harbor install.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
import pathlib
import shutil
import signal
import sys
import types
from typing import Literal, override

import pytest

from harbor_sandbox_agent_proxy import sandbox_adapter

Streams = Literal["merged", "separate"]
Timeouts = Literal["raise", "status"]
MissingFile = Literal["file-not-found", "provider", "silent"]


class SandboxFilesystemNotFoundError(Exception):
    """Stand in for a provider's missing-file error."""


@dataclasses.dataclass(frozen=True)
class LocalExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int


@dataclasses.dataclass(frozen=True)
class RecordedExec:
    command: str
    user: str | int | None
    cwd: str | None
    env: dict[str, str] | None
    timeout_sec: int | None


@dataclasses.dataclass
class LocalHarborEnvironment:
    streams: Streams = "merged"
    timeouts: Timeouts = "raise"
    missing_file: MissingFile = "file-not-found"
    execs: list[RecordedExec] = dataclasses.field(default_factory=list[RecordedExec])
    uploads: list[str] = dataclasses.field(default_factory=list[str])

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        self.execs.append(RecordedExec(command, user, cwd, env, timeout_sec))
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            command,
            cwd=cwd,
            env=None if env is None else {**os.environ, **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=(
                asyncio.subprocess.STDOUT
                if self.streams == "merged"
                else asyncio.subprocess.PIPE
            ),
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            message = f"Command timed out after {timeout_sec} seconds"
            if self.timeouts == "raise":
                raise RuntimeError(message) from None
            return LocalExecResult(stdout=None, stderr=message, return_code=124)
        return LocalExecResult(
            stdout=stdout.decode("utf-8") if stdout else None,
            stderr=stderr.decode("utf-8") if stderr else None,
            return_code=process.returncode if process.returncode is not None else 0,
        )

    async def upload_file(
        self, source_path: pathlib.Path | str, target_path: str
    ) -> None:
        self.uploads.append(target_path)
        target = pathlib.Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    async def download_file(
        self, source_path: str, target_path: pathlib.Path | str
    ) -> None:
        source = pathlib.Path(source_path)
        if not source.is_file():
            if self.missing_file == "file-not-found":
                raise FileNotFoundError(source_path)
            if self.missing_file == "provider":
                raise SandboxFilesystemNotFoundError(f"{source_path}: no such file")
            return
        shutil.copyfile(source, target_path)


local_environment = types.SimpleNamespace(
    Streams=Streams,
    Timeouts=Timeouts,
    MissingFile=MissingFile,
    LocalExecResult=LocalExecResult,
    LocalHarborEnvironment=LocalHarborEnvironment,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the local Harbor environment runs POSIX sh"
)


@pytest.mark.parametrize("streams", ["merged", "separate"])
async def test_stdout_and_stderr_come_back_apart_whatever_the_provider_does(
    streams: Streams,
) -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment(streams=streams)
    )

    result = await sandbox.exec(
        ["sh", "-c", "printf 'out\\n'; printf 'err\\n' >&2; printf more; exit 3"]
    )

    assert result.stdout == "out\nmore"
    assert result.stderr == "err\n"
    assert result.returncode == 3
    assert result.success is False


async def test_argv_reaches_the_command_verbatim() -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment()
    )

    result = await sandbox.exec(["printf", "%s|", "a b", "$HOME", "'q'", '"d"', ""])

    assert result.success
    assert result.stdout == "a b|$HOME|'q'|\"d\"||"
    assert result.stderr == ""


async def test_cwd_and_env_are_passed_through(tmp_path: pathlib.Path) -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)

    result = await sandbox.exec(
        ["sh", "-c", 'pwd; printf %s "$GREETING"'],
        cwd=str(tmp_path),
        env={"GREETING": "hello there"},
        timeout=30,
    )

    assert result.stdout == f"{tmp_path}\nhello there"
    [call] = environment.execs
    assert call.cwd == str(tmp_path)
    assert call.env == {"GREETING": "hello there"}
    assert call.timeout_sec == 30


async def test_small_stdin_is_fed_byte_exact_without_staging_a_file() -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)
    data = b'\x00\xff\x01it\'s "quoted"\n$HOME\n'

    result = await sandbox.exec(["sha256sum"], input=data)

    assert result.stdout.split()[0] == hashlib.sha256(data).hexdigest()
    assert environment.uploads == []


async def test_text_stdin_is_fed_as_utf8() -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment()
    )

    result = await sandbox.exec(["cat"], input="café ✓\nsecond line")

    assert result.stdout == "café ✓\nsecond line"


async def test_large_stdin_is_staged_through_a_file_that_is_removed_as_root() -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        environment, default_user="agent"
    )
    # Past the inline limit, as Inspect's multi-megabyte sandbox-tools install is.
    data = bytes(range(256)) * (sandbox_adapter.INLINE_STDIN_LIMIT // 256 + 1)

    result = await sandbox.exec(["sha256sum"], input=data)

    assert result.stdout.split()[0] == hashlib.sha256(data).hexdigest()
    [staged] = environment.uploads
    assert staged.startswith("/tmp/.inspect-stdin-")
    assert not (await sandbox.exec(["test", "-e", staged])).success
    [removal] = [call for call in environment.execs if call.command.startswith("rm ")]
    assert removal.user == "root"
    assert staged in removal.command


class EnvironmentEnded(RuntimeError):
    """A provider failure that is no command's outcome (as a provider's pod gone)."""


@dataclasses.dataclass
class StagedStdinStays(LocalHarborEnvironment):
    """A provider whose `rm` of a staged stdin file exits non-zero, as a refused `rm` or a
    failing `su`/`timeout` wrapper does, and that returns rather than raises."""

    command_raises: bool = False

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if command.startswith("rm -f /tmp/.inspect-stdin-"):
            return LocalExecResult(
                stdout=None,
                stderr="rm: cannot remove: Operation not permitted",
                return_code=1,
            )
        if self.command_raises:
            raise EnvironmentEnded("the environment ended before answering")
        return await super().exec(command, cwd, env, timeout_sec, user)


async def test_a_staged_stdin_file_that_stays_behind_raises_naming_it() -> None:
    environment = StagedStdinStays()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)
    data = b"x" * (sandbox_adapter.INLINE_STDIN_LIMIT + 1)

    with pytest.raises(
        RuntimeError, match="Could not remove the staged stdin file"
    ) as raised:
        await sandbox.exec(["wc", "-c"], input=data)

    [staged] = environment.uploads
    assert staged in str(raised.value)
    assert "Operation not permitted" in str(raised.value)


async def test_a_failed_command_keeps_its_own_error_when_its_staged_stdin_stays() -> (
    None
):
    environment = StagedStdinStays(command_raises=True)
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)
    data = b"x" * (sandbox_adapter.INLINE_STDIN_LIMIT + 1)

    with pytest.raises(EnvironmentEnded) as raised:
        await sandbox.exec(["wc", "-c"], input=data)

    [staged] = environment.uploads
    [note] = raised.value.__notes__
    assert staged in note
    assert "Could not remove the staged stdin file" in note


async def test_stdin_whose_encoding_would_overflow_one_argument_is_staged() -> None:
    # Inline stdin travels base64-encoded inside ONE argv string of every shell a provider
    # nests (`sh -c`, `timeout ... sh -c`, `su -c`, `bash -c`), and the kernel refuses one
    # argv string past MAX_ARG_STRLEN, 131072 bytes (gVisor: `sh: 1: timeout: File name too
    # long`). 100 KB encodes past the cap, as a browser screenshot relayed by the proxy
    # does; the local `sh -c` enforces the same cap.
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)
    data = bytes(range(256)) * (100_000 // 256)

    result = await sandbox.exec(["sha256sum"], input=data)

    assert result.stdout.split()[0] == hashlib.sha256(data).hexdigest()
    [staged] = environment.uploads
    assert staged.startswith("/tmp/.inspect-stdin-")


async def test_stdin_at_the_inline_limit_leaves_headroom_under_one_argument() -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)
    data = b"\xff" * sandbox_adapter.INLINE_STDIN_LIMIT

    result = await sandbox.exec(["sha256sum"], input=data)

    assert result.stdout.split()[0] == hashlib.sha256(data).hexdigest()
    assert environment.uploads == []
    [call] = environment.execs
    # Room for the provider's own nesting around this command (a Kubernetes provider's
    # `timeout -k ... sh -c 'su ... -c "bash -c ..."'` adds well under a kilobyte here).
    assert len(call.command.encode()) < 131072 - 16384


async def test_a_command_with_no_user_runs_as_the_default_user() -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment, default_user="root")

    await sandbox.exec(["true"])
    await sandbox.exec(["true"], user="agent")

    assert [call.user for call in environment.execs] == ["root", "agent"]


async def test_without_a_default_user_harbor_picks_the_user() -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(environment)

    await sandbox.exec(["true"])

    # None leaves Harbor's own fallback, the trial's current default user, in charge.
    assert [call.user for call in environment.execs] == [None]


async def test_files_round_trip_and_missing_parents_are_created_as_root(
    tmp_path: pathlib.Path,
) -> None:
    environment = local_environment.LocalHarborEnvironment()
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        environment, default_user="agent"
    )
    text_path = tmp_path / "deep" / "er" / "note.txt"
    bytes_path = tmp_path / "blob.bin"

    await sandbox.write_file(str(text_path), "café\n")
    await sandbox.write_file(str(bytes_path), b"\x00\xff")

    assert text_path.read_bytes() == "café\n".encode()
    assert await sandbox.read_file(str(text_path)) == "café\n"
    assert await sandbox.read_file(str(bytes_path), text=False) == b"\x00\xff"
    [mkdir, _] = environment.execs
    assert mkdir.command == f"mkdir -p {text_path.parent}"
    assert mkdir.user == "root"


async def test_a_parent_that_cannot_be_created_is_a_permission_error(
    tmp_path: pathlib.Path,
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment()
    )

    with pytest.raises(PermissionError, match="Could not create"):
        await sandbox.write_file(str(blocker / "child" / "x.txt"), "x")


@pytest.mark.parametrize("missing_file", ["file-not-found", "provider", "silent"])
async def test_a_missing_file_is_file_not_found_on_every_provider(
    tmp_path: pathlib.Path, missing_file: MissingFile
) -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment(missing_file=missing_file)
    )

    with pytest.raises(FileNotFoundError):
        await sandbox.read_file(str(tmp_path / "absent.txt"))


@dataclasses.dataclass
class BrokenTransport(LocalHarborEnvironment):
    """A provider whose transport fails with errors that are not "no such file"."""

    @override
    async def download_file(
        self, source_path: str, target_path: pathlib.Path | str
    ) -> None:
        raise RuntimeError(f"connection reset while reading {source_path}")

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        raise RuntimeError("ssh: connect to host timed out")


async def test_a_transport_failure_reading_a_file_is_not_file_not_found(
    tmp_path: pathlib.Path,
) -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(BrokenTransport())

    with pytest.raises(RuntimeError, match="connection reset"):
        await sandbox.read_file(str(tmp_path / "x.txt"))


async def test_a_timed_out_transport_without_a_requested_timeout_stays_a_runtime_error() -> (
    None
):
    sandbox = sandbox_adapter.HarborSandboxEnvironment(BrokenTransport())

    with pytest.raises(RuntimeError, match="connect to host timed out") as raised:
        await sandbox.exec(["true"])

    assert not isinstance(raised.value, TimeoutError)


@dataclasses.dataclass
class EndedEnvironment(LocalHarborEnvironment):
    """A provider whose environment ended while a command waited on it."""

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        raise EnvironmentEnded(
            "the environment ended before answering: "
            + "Agent execution timed out after 600.0 seconds"
        )


async def test_an_ended_environment_is_never_a_command_timeout() -> None:
    """A provider's plain ``RuntimeError("... timed out ...")`` is its exec timing out; a
    ``RuntimeError`` subclass is some other failure, whatever its text says: a
    ``TimeoutError`` reads to Inspect's tools as one command that ran out of time, and the
    agent goes on.
    """
    sandbox = sandbox_adapter.HarborSandboxEnvironment(EndedEnvironment())

    with pytest.raises(EnvironmentEnded):
        await sandbox.exec(["true"], timeout=30)


@pytest.mark.parametrize("timeouts", ["raise", "status"])
async def test_a_command_past_its_timeout_raises_timeout_error_on_every_provider(
    timeouts: Timeouts,
) -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment(timeouts=timeouts)
    )

    with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
        await sandbox.exec(["sleep", "30"], timeout=1)


@pytest.mark.parametrize("timeouts", ["raise", "status"])
async def test_a_command_that_exits_124_itself_is_not_a_timeout(
    timeouts: Timeouts,
) -> None:
    sandbox = sandbox_adapter.HarborSandboxEnvironment(
        local_environment.LocalHarborEnvironment(timeouts=timeouts)
    )

    result = await sandbox.exec(["sh", "-c", "printf partial; exit 124"], timeout=30)

    assert result.returncode == 124
    assert result.stdout == "partial"


async def test_the_polling_interval_is_the_callers_or_inspects_default() -> None:
    environment = local_environment.LocalHarborEnvironment()
    default = sandbox_adapter.HarborSandboxEnvironment(
        environment
    ).default_polling_interval()

    assert (
        sandbox_adapter.HarborSandboxEnvironment(
            environment, polling_interval=7.5
        ).default_polling_interval()
        == 7.5
    )
    assert default > 0
