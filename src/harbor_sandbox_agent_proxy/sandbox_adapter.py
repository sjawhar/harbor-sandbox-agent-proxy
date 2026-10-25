"""Adapt a Harbor environment to Inspect's sandbox primitives without importing Harbor."""

from __future__ import annotations

import base64
import pathlib
import shlex
import tempfile
import uuid
from typing import TYPE_CHECKING, Final, Literal, Protocol, overload, override

import inspect_ai.util

if TYPE_CHECKING:
    from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentConfigType

STDERR_BOUNDARY: Final = "\x1e\x1e__INSPECT_HARBOR_ADAPTER_STDERR_BOUNDARY__\x1e\x1e"
_SPLIT_SCRIPT: Final = (
    "__o=$(mktemp); __e=$(mktemp); "
    + 'eval "$1" >"$__o" 2>"$__e"; __rc=$?; '
    + 'cat "$__o"; printf %s "'
    + STDERR_BOUNDARY
    + '"; cat "$__e"; '
    + 'rm -f "$__o" "$__e"; exit $__rc'
)
_MAX_ARG_STRLEN: Final = 131072
_INLINE_STDIN_ENCODED_LIMIT: Final = _MAX_ARG_STRLEN - 32 * 1024
INLINE_STDIN_LIMIT: Final = _INLINE_STDIN_ENCODED_LIMIT // 4 * 3
_TIMEOUT_EXIT_CODE: Final = 124
_FILE_TRANSFER_USER: Final = "root"


class HarborExecResult(Protocol):
    @property
    def stdout(self) -> str | None: ...

    @property
    def stderr(self) -> str | None: ...

    @property
    def return_code(self) -> int: ...


class HarborEnvironment(Protocol):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> HarborExecResult: ...

    async def upload_file(
        self, source_path: pathlib.Path | str, target_path: str
    ) -> None: ...

    async def download_file(
        self, source_path: str, target_path: pathlib.Path | str
    ) -> None: ...


def _is_not_found(error: Exception) -> bool:
    """Recognize a provider-specific missing-file error without its optional SDK."""
    return "NotFound" in type(error).__name__


class HarborSandboxEnvironment(inspect_ai.util.SandboxEnvironment):
    """Use an already-running Harbor environment as an Inspect sandbox."""

    def __init__(
        self,
        environment: HarborEnvironment,
        *,
        default_user: str | None = None,
        polling_interval: float | None = None,
    ) -> None:
        super().__init__()
        self._environment: HarborEnvironment = environment
        self._default_user: str | None = default_user
        self._polling_interval: float | None = polling_interval

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
    ) -> inspect_ai.util.ExecResult[str]:
        del timeout_retry, concurrency
        inner = shlex.join(cmd)
        staged_stdin: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            if len(data) <= INLINE_STDIN_LIMIT:
                encoded = base64.b64encode(data).decode("ascii")
                inner = f"printf '%s' {shlex.quote(encoded)} | base64 -d | {inner}"
            else:
                staged_stdin = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
                await self._upload_bytes(data, staged_stdin)
                inner = f"{inner} < {shlex.quote(staged_stdin)}"
        wrapped = f"sh -c {shlex.quote(_SPLIT_SCRIPT)} sh {shlex.quote(inner)}"
        command_error: BaseException | None = None
        try:
            result = await self._environment.exec(
                command=wrapped,
                cwd=cwd,
                env=env,
                timeout_sec=timeout,
                user=user if user is not None else self._default_user,
            )
        except RuntimeError as error:
            if (
                timeout is not None
                and type(error) is RuntimeError
                and "timed out" in str(error)
            ):
                command_error = TimeoutError(str(error))
                raise command_error from error
            command_error = error
            raise
        except BaseException as error:
            command_error = error
            raise
        finally:
            if staged_stdin is not None:
                await self._remove_staged_stdin(staged_stdin, command_error)
        merged = result.stdout or ""
        if STDERR_BOUNDARY in merged:
            stdout, _, stderr = merged.partition(STDERR_BOUNDARY)
            if result.stderr:
                stderr += result.stderr
        else:
            if timeout is not None and result.return_code == _TIMEOUT_EXIT_CODE:
                raise TimeoutError(
                    result.stderr or f"Command timed out after {timeout} seconds"
                )
            stdout, stderr = merged, result.stderr or ""
        return inspect_ai.util.ExecResult[str](
            success=result.return_code == 0,
            returncode=result.return_code,
            stdout=stdout,
            stderr=stderr,
        )

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        data = contents.encode("utf-8") if isinstance(contents, str) else contents
        parent = str(pathlib.PurePosixPath(file).parent)
        if parent not in ("", "."):
            created = await self._environment.exec(
                command=f"mkdir -p {shlex.quote(parent)}", user=_FILE_TRANSFER_USER
            )
            if created.return_code != 0:
                raise PermissionError(
                    f"Could not create {parent} in the sandbox: "
                    + (created.stderr or created.stdout or "no output")
                )
        await self._upload_bytes(data, file)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        with tempfile.TemporaryDirectory() as directory:
            local_path = pathlib.Path(directory) / "file"
            try:
                await self._environment.download_file(file, local_path)
            except FileNotFoundError:
                raise
            except Exception as error:
                if _is_not_found(error):
                    raise FileNotFoundError(file) from error
                raise
            if not local_path.is_file():
                raise FileNotFoundError(file)
            data = local_path.read_bytes()
        return data.decode("utf-8") if text else data

    @override
    def default_polling_interval(self) -> float:
        return (
            super().default_polling_interval()
            if self._polling_interval is None
            else self._polling_interval
        )

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        del task_name, config, environments, interrupted

    async def _upload_bytes(self, data: bytes, target: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "upload"
            source.write_bytes(data)
            await self._environment.upload_file(source, target)

    async def _remove_staged_stdin(
        self, path: str, command_error: BaseException | None
    ) -> None:
        try:
            removed = await self._environment.exec(
                command=f"rm -f {shlex.quote(path)}", user=_FILE_TRANSFER_USER
            )
        except Exception as error:
            if command_error is None:
                raise
            command_error.add_note(
                f"Removing the staged stdin file {path} also failed: {error!r}"
            )
            return
        if removed.return_code == 0:
            return
        output = (removed.stderr or removed.stdout or "").strip() or "no output"
        message = (
            f"Could not remove the staged stdin file {path}: "
            + f"rm exited {removed.return_code}: {output}"
        )
        if command_error is None:
            raise RuntimeError(message)
        command_error.add_note(message)
