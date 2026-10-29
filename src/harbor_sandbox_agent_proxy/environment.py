"""Harbor environment wrapper that confines credentials and locks verifier tests."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import json
import logging
import shlex
import tempfile
from collections.abc import Collection, Generator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, override, runtime_checkable

import harbor.environments.base as harbor_environment_base
import pydantic
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import (
    EnvironmentConfig,
    HealthcheckConfig,
    NetworkPolicy,
    TaskOS,
)
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.utils.env import resolve_env_vars
from harbor.utils.import_path import import_class

from .config import RelayConfig
from .outcome import EventSink, JsonOutcomeSink, RelayOutcome, TrialIdentity
from .routing import ModelCall, Route, default_protected_values, default_route
from .task_metadata import BridgedMcpServer, read_bridged_servers

logger = logging.getLogger(__name__)

_TESTS_STAGE_TEMPLATE = ".sandbox-agent-proxy-tests.XXXXXXXXXX"
_TESTS_STAGE_PREFIX = _TESTS_STAGE_TEMPLATE.rstrip("X")
_TESTS_MIRROR_TEMPLATE = ".sandbox-agent-proxy-tests-mirror.XXXXXXXXXX"
_TESTS_MIRROR_PREFIX = _TESTS_MIRROR_TEMPLATE.rstrip("X")
_ROOT_UID = 0
_REFUSED_EXIT = 3

MANAGED_PROVIDER_VARIABLES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VERTEX_API_KEY",
        "GOOGLE_VERTEX_BASE_URL",
    }
)


class SeparateVerifierEnvironmentError(RuntimeError):
    """The relay needs Harbor's shared trial environment for verification."""


class RunnerSecretExposureError(RuntimeError):
    """A protected host value would be exposed to a Harbor trial."""


SEPARATE_VERIFIER_UNSUPPORTED = (
    "sandbox-agent-proxy does not support Harbor's separate verifier environment; "
    + "run the task's verifier in the trial's own environment "
    + "(verifier environment_mode 'shared')"
)

_MOUNTS = pydantic.TypeAdapter(list[dict[str, object]])


def binds_trial_agent_dir(trial_paths: TrialPaths, mounts: object) -> bool:
    """Whether Harbor mounted the trial agent directory into this environment."""
    try:
        entries = _MOUNTS.validate_python(mounts or [])
    except pydantic.ValidationError:
        return False
    agent_dir = trial_paths.agent_dir.resolve()
    return any(
        isinstance(source := entry.get("source"), str)
        and Path(source).resolve() == agent_dir
        for entry in entries
    )


class _ExistingTests(enum.StrEnum):
    ABSENT = "absent"
    COPIED = "copied"
    SUPERSEDED = "superseded"
    DISCARDED = "discarded"


@dataclasses.dataclass(frozen=True)
class _TestsMirror:
    """The wrapper-owned root-only copy of its last installed tests tree."""

    holder: PurePosixPath
    emptied: bool = False


def _uid_of(user: str | int) -> str:
    text = str(user)
    if text == "root":
        return "0"
    if text.isdigit():
        return text
    return f"$(id -u {shlex.quote(text)})"


def _resolved_env(value: object, *, label: str) -> dict[str, str] | None:
    """Resolve one Harbor environment mapping before it reaches an inner backend."""
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(entry, str)
        for name, entry in value.items()
    ):
        raise TypeError(f"{label} must be a dict[str, str]")
    return resolve_env_vars(value)


def _without_protected(
    environment: Mapping[str, str], protected: Collection[str]
) -> dict[str, str]:
    """Remove values that contain a host value the secret collaborator protects."""
    return {
        name: value
        for name, value in environment.items()
        if not any(secret and secret in value for secret in protected)
    }


def _mirror_check(
    holder: PurePosixPath, *, copy: PurePosixPath | None = None
) -> tuple[str, ...]:
    quoted = shlex.quote(str(holder))
    copy_check = ""
    if copy is not None:
        copied = shlex.quote(str(copy))
        copy_check = f" \\\n+  || [ -L {copied} ] || [ ! -d {copied} ]"
    return (
        f"if [ -L {quoted} ] || [ ! -d {quoted} ] \\",
        f'  || [ "$(stat -c %u {quoted})" != "$stager_uid" ] \\',
        f'  || [ "$(stat -c %a {quoted})" != 700 ]{copy_check}; then',
        "  echo 'tests mirror is not the directory only root can enter' >&2",
        f"  exit {_REFUSED_EXIT}",
        "fi",
    )


def _held_files(tests: str) -> tuple[str, ...]:
    return (
        f"  held=$(find -P {tests} -mindepth 1 -print -quit) || fail",
        '  if [ -n "$held" ]; then',
        f"    existing={_ExistingTests.DISCARDED}",
        "  else",
        f"    existing={_ExistingTests.SUPERSEDED}",
        "  fi",
    )


def _stage_tests_script(
    tests_dir: PurePosixPath, owner: str | int, mirror: _TestsMirror | None
) -> str:
    """Create a root-only stage beside ``tests_dir`` and print its path and state."""
    tests = shlex.quote(str(tests_dir))
    parent = shlex.quote(str(tests_dir.parent))
    staged = f'"$stage"/{shlex.quote(tests_dir.name)}'
    check_copy: tuple[str, ...] = ()
    build_on_copy: tuple[str, ...] = ()
    if mirror is None:
        untrusted = (
            f"find -P {tests} \\( \\( -type l ! -user {_ROOT_UID} \\) -o \\( ! -type l"
            + f" \\( ! -user {_ROOT_UID} -o -perm /022 \\) \\) \\) -print -quit"
        )
        existing: tuple[str, ...] = (
            f"  untrusted=$({untrusted}) || fail",
            '  if [ -z "$untrusted" ]; then',
            f"    cp -a {tests}/. {staged}/ || fail",
            f"    existing={_ExistingTests.COPIED}",
            "  else",
            *(f"  {line}" for line in _held_files(tests)),
            "  fi",
        )
    elif mirror.emptied:
        existing = _held_files(tests)
    else:
        existing = (f"  existing={_ExistingTests.SUPERSEDED}",)
        copy = mirror.holder / tests_dir.name
        check_copy = _mirror_check(mirror.holder, copy=copy)
        build_on_copy = (f"cp -a {shlex.quote(str(copy))}/. {staged}/ || fail",)
    return "\n".join(
        (
            "umask 077",
            f"owner_uid={_uid_of(owner)} || exit 1",
            "stager_uid=$(id -u) || exit 1",
            f"parent_uid=$(stat -c %u {parent}) && parent_mode=$(stat -c %a {parent})"
            + " || exit 1",
            'if { [ "$parent_uid" != 0 ] && [ "$parent_uid" != "$owner_uid" ] \\',
            '  && [ "$parent_uid" != "$stager_uid" ]; } \\',
            "  || [ $((0$parent_mode & 022)) -ne 0 ]; then",
            "  printf '%s is owned by uid %s with mode %s; users other than root "
            + f'could rename a stage there\\n\' {parent} "$parent_uid" "$parent_mode" >&2',
            f"  exit {_REFUSED_EXIT}",
            "fi",
            *check_copy,
            "stage=$(mktemp -d "
            + shlex.quote(str(tests_dir.parent / _TESTS_STAGE_TEMPLATE))
            + ") || exit 1",
            'fail() { rm -rf -- "$stage"; exit 1; }',
            "refuse() {",
            '  rm -rf -- "$stage"',
            f'  printf "%s %s; verifier tests are not put there\\n" {tests} "$1" >&2',
            f"  exit {_REFUSED_EXIT}",
            "}",
            f"if [ -L {tests} ]; then",
            f'  refuse "is a symbolic link to $(readlink {tests})"',
            f"elif [ -e {tests} ] && [ ! -d {tests} ]; then",
            '  refuse "is not a directory"',
            "fi",
            f"mkdir {staged} || fail",
            f"existing={_ExistingTests.ABSENT}",
            f"if [ -d {tests} ]; then",
            *existing,
            "fi",
            *build_on_copy,
            'printf "%s\\n%s\\n" "$stage" "$existing"',
        )
    )


def _move_tests_in_script(
    tests_dir: PurePosixPath,
    stage: PurePosixPath,
    owner: str | int,
    existing: _ExistingTests,
    holder: PurePosixPath | None,
) -> str:
    """Lock staged tests, make a root-only copy, then atomically move them in place."""
    tests = shlex.quote(str(tests_dir))
    staged = shlex.quote(str(stage / tests_dir.name))
    copy = shlex.quote(str(stage / "mirror"))
    if existing is _ExistingTests.ABSENT:
        clear: tuple[str, ...] = (
            f"if [ -e {tests} ] || [ -L {tests} ]; then",
            f"  echo '{tests_dir} appeared while verifier tests were staged' >&2",
            f"  exit {_REFUSED_EXIT}",
            "fi",
        )
    else:
        clear = (
            f"if [ -L {tests} ] || [ ! -d {tests} ]; then",
            f"  echo '{tests_dir} changed while verifier tests were staged' >&2",
            f"  exit {_REFUSED_EXIT}",
            "fi",
            f"mv -T {tests} {shlex.quote(str(stage / 'previous'))} || exit 1",
        )
    if holder is None:
        check: tuple[str, ...] = ()
        keep: tuple[str, ...] = (
            "mirror=$(mktemp -d "
            + shlex.quote(str(tests_dir.parent / _TESTS_MIRROR_TEMPLATE))
            + ") || exit 1",
            f'mv -T {copy} "$mirror"/{shlex.quote(tests_dir.name)}'
            + ' || { rm -rf -- "$mirror"; exit 1; }',
            'printf "%s\\n" "$mirror"',
        )
    else:
        kept = shlex.quote(str(holder / tests_dir.name))
        check = ("stager_uid=$(id -u) || exit 1", *_mirror_check(holder))
        keep = (f"rm -rf -- {kept} || exit 1", f"mv -T {copy} {kept} || exit 1")
    return "\n".join(
        (
            *check,
            f"chown -R {shlex.quote(str(owner))} {staged} || exit 1",
            f"chmod -R u=rwX,go= {staged} || exit 1",
            f"cp -a {staged} {copy} || exit 1",
            *clear,
            f"mv -T {staged} {tests} || exit 1",
            *keep,
            f"rm -rf -- {shlex.quote(str(stage))}",
        )
    )


def _made_mirror(output: str | None, tests_dir: PurePosixPath) -> PurePosixPath:
    lines = (output or "").strip().splitlines()
    holder = PurePosixPath(lines[-1].strip()) if lines else None
    if (
        holder is None
        or holder.parent != tests_dir.parent
        or not holder.name.startswith(_TESTS_MIRROR_PREFIX)
    ):
        raise RuntimeError(
            "Installing verifier tests printed no root-only mirror directory: "
            + repr(output)
        )
    return holder


class _RelayPhase(Protocol):
    @property
    def verifier_conversation(self) -> object | None: ...

    @property
    def caller_keys(self) -> frozenset[str]: ...

    def phase_env(self) -> dict[str, str]: ...

    async def close(self) -> RelayOutcome: ...


@runtime_checkable
class _RemoteExecutable(Protocol):
    async def exec_remote(self, *args: object, **kwargs: object) -> object: ...


class RelayEnvironment(BaseEnvironment):
    """Wrap one Harbor environment with the relay's host extension seam.

    Hosts may subclass this environment to decide model routing, which values remain
    outside the sandbox and records, where trial events are recorded, and which task
    MCP servers the relay bridges. The defaults use public provider credentials from
    the process running Harbor and task metadata from ``task.toml``.
    """

    _inner: BaseEnvironment
    _configured_environments: tuple[tuple[str, Mapping[str, str]], ...]
    _tests_mirror: _TestsMirror | None
    _sink: EventSink | None
    _setup_env: dict[str, str] | None
    relay: _RelayPhase | None
    outcome: RelayOutcome | None
    caller_keys: frozenset[str]

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
        **kwargs: Any,
    ) -> None:
        if "mounts" in kwargs and not binds_trial_agent_dir(
            trial_paths, kwargs["mounts"]
        ):
            raise SeparateVerifierEnvironmentError(SEPARATE_VERIFIER_UNSUPPORTED)
        self.config = RelayConfig.model_validate(relay)
        protected = self.protected_values()
        resolved_task_env = (
            resolve_env_vars(task_env_config.env) if task_env_config.env else {}
        )
        confined_task_env = _without_protected(resolved_task_env, protected)
        confined_task_config = task_env_config.model_copy(
            update={"env": confined_task_env}
        )
        confined_kwargs = dict(kwargs)
        persistent_env = _resolved_env(
            confined_kwargs.get("persistent_env"), label="persistent_env"
        )
        if persistent_env is not None:
            confined_kwargs["persistent_env"] = _without_protected(
                persistent_env, protected
            )
        self._configured_environments = (
            ("the task environment", resolved_task_env),
            ("the persistent environment", persistent_env or {}),
        )

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
            task_env_config=confined_task_config,
            **raw_kwargs,
            **confined_kwargs,
        )

        self._tests_mirror = None
        self._sink = None
        self._setup_env = None
        self.relay = None
        self.outcome = None
        self.caller_keys = frozenset()
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            confined_task_config,
            **confined_kwargs,
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
    def os(self) -> TaskOS:
        return self._inner.os

    @property
    @override
    def default_user(self) -> str | int | None:
        return self._inner.default_user

    @default_user.setter
    @override
    def default_user(self, value: str | int | None) -> None:
        self._inner.default_user = value

    @property
    def trial(self) -> TrialIdentity:
        """Return this Harbor trial's stable record identity."""
        return TrialIdentity(
            self.trial_paths.trial_dir.name, self.trial_paths.trial_dir
        )

    async def route(self, call: ModelCall) -> Route:
        """Select the default public-provider destination for one model call."""
        return await default_route(call)

    def protected_values(self) -> Collection[str]:
        """Return host values that must never enter this sandbox or its records."""
        return default_protected_values()

    async def open_event_sink(self, trial: TrialIdentity) -> EventSink:
        """Open the default redacted outcome file for ``trial``."""
        return JsonOutcomeSink(
            trial,
            protected=lambda: {*self.protected_values(), *self.caller_keys},
        )

    async def bridged_tool_servers(self) -> Sequence[BridgedMcpServer]:
        """Read this task's bridged MCP server definitions."""
        return read_bridged_servers(self.environment_dir.parent)

    async def open_sink(self) -> EventSink:
        """Open the host sink once and retain it for this trial's lifecycle."""
        if self._sink is None:
            self._sink = await self.open_event_sink(self.trial)
        return self._sink

    @contextlib.contextmanager
    def scoped_agent_setup_env(
        self, environment: Mapping[str, str]
    ) -> Generator[None, None, None]:
        """Temporarily supply an installed agent's declared proxy connection."""
        previous = self._setup_env
        self._setup_env = dict(environment)
        try:
            yield
        finally:
            self._setup_env = previous

    def attach_relay(self, relay: _RelayPhase) -> None:
        """Associate this trial's installed-agent relay with its environment."""
        self.relay = relay

    def record_runner_loop_outcome(self, outcome: RelayOutcome) -> None:
        """Retain a runner-loop outcome until ``stop`` delivers it to the host sink."""
        self.outcome = outcome

    def _validate_definition(self) -> None:
        self._inner._validate_definition()

    @override
    def _startup_env(self) -> dict[str, str]:
        startup_env = self._inner._startup_env()
        self._refuse_protected(startup_env, where="the sandbox's startup environment")
        return (
            self._confined(startup_env, where="the sandbox's startup environment") or {}
        )

    @override
    async def start(self, force_build: bool) -> None:
        self._refuse_configured_protected()
        _ = self._startup_env()
        self._tests_mirror = None
        await self._inner.start(force_build)

    @override
    async def stop(self, delete: bool) -> None:
        relay, self.relay = self.relay, None
        self._tests_mirror = None
        try:
            await self._inner.stop(delete)
        finally:
            try:
                if relay is not None:
                    self.caller_keys = relay.caller_keys
                    self.outcome = await relay.close()
            finally:
                if self.outcome is not None:
                    await (await self.open_sink()).close(self.outcome)

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._inner.upload_file(source_path, target_path)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        tests_dir = EnvironmentPaths.for_os(self.os).tests_dir
        if PurePosixPath(target_dir) != tests_dir:
            await self._inner.upload_dir(source_dir, target_dir)
            return
        await self._install_tests(source_dir, tests_dir)

    async def _install_tests(
        self, source_dir: Path | str, tests_dir: PurePosixPath
    ) -> None:
        """Install verifier tests only after they have a private, locked staging tree."""
        owner = self.default_user if self.default_user is not None else "root"
        mirror = self._tests_mirror
        stage, existing = await self._stage_tests(tests_dir, owner, mirror)
        try:
            await self._inner.upload_dir(source_dir, str(stage / tests_dir.name))
            await self._write_conversation(stage / tests_dir.name)
            result = await self._exec_as_root(
                _move_tests_in_script(
                    tests_dir,
                    stage,
                    owner,
                    existing,
                    None if mirror is None else mirror.holder,
                ),
                what=f"lock verifier tests and move them to {tests_dir}",
            )
            holder = (
                _made_mirror(result.stdout, tests_dir)
                if mirror is None
                else mirror.holder
            )
        except Exception as error:
            self._tests_mirror = None
            await self._discard_stage(stage, error)
            raise
        self._tests_mirror = _TestsMirror(holder=holder)
        if existing is not _ExistingTests.DISCARDED:
            return
        if mirror is None:
            logger.warning(
                "Replaced %s, whose files a user other than root could have written, "
                + "with the verifier's tests; what it held was discarded unread",
                tests_dir,
            )
        else:
            logger.warning(
                "Replaced %s, which held files again after Harbor emptied it, with "
                + "the verifier's tests; what it held was discarded unread",
                tests_dir,
            )

    async def _write_conversation(self, staged: PurePosixPath) -> None:
        relay = self.relay
        conversation = None if relay is None else relay.verifier_conversation
        if conversation is None:
            return
        target = staged / self.config.conversation_path.relative_to("/tests")
        _ = await self._exec_as_root(
            f"umask 077 && mkdir -p {shlex.quote(str(target.parent))}",
            what=f"make {target.parent}",
        )
        conversation_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", delete=False
            ) as conversation_file:
                json.dump(conversation, conversation_file)
                conversation_path = Path(conversation_file.name)
            await self._inner.upload_file(conversation_path, str(target))
        finally:
            if conversation_path is not None:
                conversation_path.unlink(missing_ok=True)

    async def _stage_tests(
        self,
        tests_dir: PurePosixPath,
        owner: str | int,
        mirror: _TestsMirror | None,
    ) -> tuple[PurePosixPath, _ExistingTests]:
        result = await self._exec_as_root(
            _stage_tests_script(tests_dir, owner, mirror),
            what=f"stage verifier tests beside {tests_dir}",
        )
        lines = (result.stdout or "").strip().splitlines()
        try:
            stage = PurePosixPath(lines[-2].strip())
            existing = _ExistingTests(lines[-1].strip())
        except (IndexError, ValueError):
            stage, existing = None, None
        if (
            stage is None
            or existing is None
            or stage.parent != tests_dir.parent
            or not stage.name.startswith(_TESTS_STAGE_PREFIX)
        ):
            raise RuntimeError(
                "Staging verifier tests printed no stage and state: "
                + repr(result.stdout)
            )
        return stage, existing

    async def _discard_stage(self, stage: PurePosixPath, error: Exception) -> None:
        try:
            result = await self.exec(
                f"rm -rf -- {shlex.quote(str(stage))}", user="root"
            )
        except Exception as cleanup_error:  # noqa: BLE001
            error.add_note(f"Removing {stage} failed too: {cleanup_error!r}")
            return
        if result.return_code != 0:
            output = (result.stderr or result.stdout or "no output").strip()
            error.add_note(
                f"Removing {stage} failed too: exit {result.return_code}: {output}"
            )

    async def _exec_as_root(self, command: str, *, what: str) -> ExecResult:
        result = await self.exec(command, user="root")
        if result.return_code != 0:
            output = (result.stderr or result.stdout or "no output").strip()
            raise RuntimeError(
                f"Could not {what}: exited {result.return_code}: {output}"
            )
        return result

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._inner.download_file(source_path, target_path)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._inner.download_dir(source_dir, target_dir)

    @override
    async def download_dir_filtered(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        protect: Sequence[str] | None = None,
    ) -> None:
        await self._inner.download_dir_filtered(
            source_dir=source_dir,
            target_dir=target_dir,
            include=include,
            exclude=exclude,
            protect=protect,
        )

    @override
    @contextlib.contextmanager
    def scoped_output_callback(
        self, callback: harbor_environment_base.OutputCallback | None
    ) -> Generator[None, None, None]:
        with super().scoped_output_callback(callback):
            with self._inner.scoped_output_callback(callback):
                yield

    @override
    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return await self._inner.service_exec(
            command,
            service=service,
            cwd=cwd,
            env=self._confined(env, where="a service exec environment"),
            timeout_sec=timeout_sec,
            user=user,
        )

    @override
    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        await self._inner.service_download_file(
            source_path, target_path, service=service
        )

    @override
    async def service_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        await self._inner.service_download_dir(source_dir, target_dir, service=service)

    @override
    async def service_download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
        service: str | None = None,
    ) -> None:
        await self._inner.service_download_dir_with_exclusions(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=exclude,
            service=service,
        )

    @override
    async def service_is_dir(
        self,
        path: str,
        *,
        service: str | None = None,
        user: str | int | None = None,
    ) -> bool:
        return await self._inner.service_is_dir(path, service=service, user=user)

    async def exec_remote(self, *args: object, **kwargs: object) -> object:
        if not isinstance(self._inner, _RemoteExecutable):
            raise AttributeError(
                f"{self._inner.type()} does not expose an exec_remote operation."
            )
        return await self._inner.exec_remote(*args, **kwargs)

    @override
    async def run_healthcheck(
        self, healthcheck: HealthcheckConfig | None = None
    ) -> None:
        await self._inner.run_healthcheck(healthcheck)

    @override
    async def set_network_policy(self, network_policy: NetworkPolicy) -> None:
        await self._inner.set_network_policy(network_policy)

    @override
    async def stop_service(self, service: str) -> None:
        await self._inner.stop_service(service)

    @override
    async def prepare_logs_for_host(self) -> None:
        await self._inner.prepare_logs_for_host()

    @override
    async def empty_dirs(
        self,
        dirs: Sequence[harbor_environment_base.EnvironmentPath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        tests_dir = EnvironmentPaths.for_os(self.os).tests_dir
        if self._tests_mirror is not None and any(
            PurePosixPath(directory) == tests_dir for directory in dirs
        ):
            self._tests_mirror = dataclasses.replace(self._tests_mirror, emptied=True)
        return await self._inner.empty_dirs(dirs, chmod=chmod)

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
            env=self._confined(env, where="an exec environment"),
            timeout_sec=timeout_sec,
            user=user,
        )

    def _refuse_configured_protected(self) -> None:
        for where, environment in self._configured_environments:
            self._refuse_protected(environment, where=where)

    def _refuse_protected(
        self, environment: Mapping[str, str] | None, *, where: str
    ) -> None:
        if not environment:
            return
        protected = self.protected_values()
        exposed = sorted(
            name
            for name, value in environment.items()
            if any(secret and secret in value for secret in protected)
        )
        if exposed:
            raise RunnerSecretExposureError(
                f"{where} carries a protected value under {', '.join(exposed)}; "
                + "protected values never enter a Harbor trial"
            )

    def _confined(
        self, env: dict[str, str] | None, *, where: str
    ) -> dict[str, str] | None:
        merged = self._merge_env(env) or {}
        phase_env = (
            self.relay.phase_env() if self.relay is not None else self._setup_env or {}
        )
        managed = MANAGED_PROVIDER_VARIABLES | phase_env.keys()
        merged = {name: value for name, value in merged.items() if name not in managed}
        self._refuse_protected(merged, where=where)
        merged.update(phase_env)
        return merged or None
