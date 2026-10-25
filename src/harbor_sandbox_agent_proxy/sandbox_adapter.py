"""Adapt a Harbor environment to Inspect's sandbox primitives without importing Harbor."""

from __future__ import annotations

import base64
import pathlib
import shlex
import tempfile
import uuid
from typing import Literal, Protocol, overload, override

import inspect_ai.util

STDERR_BOUNDARY = "\x1e\x1e__INSPECT_HARBOR_ADAPTER_STDERR_BOUNDARY__\x1e\x1e"
_SPLIT_SCRIPT = (
    "__o=$(mktemp); __e=$(mktemp); "
    + 'eval "$1" >"$__o" 2>"$__e"; __rc=$?; '
    + 'cat "$__o"; printf %s "'
    + STDERR_BOUNDARY
    + '"; cat "$__e"; rm -f "$__o" "$__e"; exit $__rc'
)
_MAX_ARG_STRLEN = 131072
INLINE_STDIN_LIMIT = (_MAX_ARG_STRLEN - 32 * 1024) // 4 * 3
_FILE_TRANSFER_USER = "root"


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

    async def upload_file(self, source_path: pathlib.Path | str, target_path: str) -> None: ...

    async def download_file(self, source_path: str, target_path: pathlib.Path | str) -> None: ...


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
        self._environment = environment
        self._default_user = default_user
        self._polling_interval = polling_interval

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
        command = shlex.join(cmd)
        staged: str | None = None
        if input is not None:
            data = input.encode() if isinstance(input, str) else input
            if len(data) <= INLINE_STDIN_LIMIT:
                encoded = base64.b64encode(data).decode("ascii")
                command = f"printf '%s' {shlex.quote(encoded)} | base64 -d | {command}"
            else:
                staged = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
                await self._upload_bytes(data, staged)
                command = f"{command} < {shlex.quote(staged)}"
        wrapped = f"sh -c {shlex.quote(_SPLIT_SCRIPT)} sh {shlex.quote(command)}"
        failure: BaseException | None = None
        try:
            result = await self._environment.exec(
                wrapped,
                cwd=cwd,
                env=env,
                timeout_sec=timeout,
                user=user if user is not None else self._default_user,
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            if staged is not None:
                await self._remove_staged(staged, failure)
        merged = result.stdout or ""
        if STDERR_BOUNDARY in merged:
            stdout, _, stderr = merged.partition(STDERR_BOUNDARY)
            stderr += result.stderr or ""
        else:
            if timeout is not None and result.return_code == 124:
                raise TimeoutError(result.stderr or f"Command timed out after {timeout} seconds")
            stdout, stderr = merged, result.stderr or ""
        return inspect_ai.util.ExecResult(
            success=result.return_code == 0,
            returncode=result.return_code,
            stdout=stdout,
            stderr=stderr,
        )

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        data = contents.encode() if isinstance(contents, str) else contents
        parent = str(pathlib.PurePosixPath(file).parent)
        if parent not in {"", "."}:
            made = await self._environment.exec(f"mkdir -p {shlex.quote(parent)}", user=_FILE_TRANSFER_USER)
            if made.return_code != 0:
                raise PermissionError(made.stderr or made.stdout or f"Could not create {parent}")
        await self._upload_bytes(data, file)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        with tempfile.TemporaryDirectory() as directory:
            local = pathlib.Path(directory) / "file"
            try:
                await self._environment.download_file(file, local)
            except FileNotFoundError:
                raise
            except Exception as error:
                if "NotFound" in type(error).__name__:
                    raise FileNotFoundError(file) from error
                raise
            if not local.is_file():
                raise FileNotFoundError(file)
            data = local.read_bytes()
        return data.decode() if text else data

    @override
    def default_polling_interval(self) -> float:
        return super().default_polling_interval() if self._polling_interval is None else self._polling_interval

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: inspect_ai.util.SandboxEnvironmentConfigType | None,
        environments: dict[str, inspect_ai.util.SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        del task_name, config, environments, interrupted

    async def _upload_bytes(self, data: bytes, target: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "upload"
            source.write_bytes(data)
            await self._environment.upload_file(source, target)

    async def _remove_staged(self, path: str, failure: BaseException | None) -> None:
        try:
            result = await self._environment.exec(f"rm -f {shlex.quote(path)}", user=_FILE_TRANSFER_USER)
        except Exception as error:
            if failure is None:
                raise
            failure.add_note(f"Removing staged input {path} also failed: {error!r}")
            return
        if result.return_code == 0:
            return
        message = f"Could not remove staged input {path}: {result.stderr or result.stdout or 'no output'}"
        if failure is None:
            raise RuntimeError(message)
        failure.add_note(message)
