"""Private Inspect compatibility boundary for the sandbox-agent proxy.

All private Inspect dependencies used by this package are resolved here so an
upstream move fails at import time with an actionable compatibility error.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextvars import ContextVar
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, NoReturn, TypeAlias, cast

from inspect_ai.agent import SandboxAgentBridge
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_ai.util import (
    ExecRemoteStreamingOptions,
    SandboxEnvironment,
    sandbox_service,
)
from pydantic import JsonValue

TESTED_RANGE = ("0.3.278", "0.3.285")

__all__ = [
    "Descent",
    "ExecRemoteStreamingOptions",
    "MODEL_SERVICE",
    "MessageFingerprint",
    "MCPServerConfigHTTP",
    "PROVIDER_ERROR_KEY",
    "SANDBOX_CLI",
    "SandboxAgentBridge",
    "SandboxEnvironmentProxy",
    "TESTED_RANGE",
    "call_tool",
    "condensed_fingerprint",
    "extends",
    "inner_exception",
    "list_tools",
    "message_fingerprint",
    "monitor_failure",
    "monitor_proxy",
    "position_descent",
    "provider_error_payload",
    "register_bridged_tools",
    "sandbox_default_context_var",
    "sandbox_environments_context_var",
    "sandbox_service",
    "sandbox_tools_user",
    "sandbox_with_environments_context_var",
    "sandbox_with_injected_tools",
]


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _moved(name: str, error: ImportError | AttributeError) -> NoReturn:
    message = (
        f"inspect-ai {_installed_version('inspect-ai')} has moved {name}; "
        + "harbor-sandbox-agent-proxy "
        + f"{_installed_version('harbor-sandbox-agent-proxy')} was tested with "
        + f"inspect-ai {TESTED_RANGE}"
    )
    raise ImportError(message) from error


def _inspect_value(module: str, name: str) -> object:
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as error:
        _moved(f"{module.removeprefix('inspect_ai.')}.{name}", error)


PROVIDER_ERROR_KEY = cast(
    str, _inspect_value("inspect_ai.agent._bridge._errors", "PROVIDER_ERROR_KEY")
)
provider_error_payload = cast(
    Callable[[Exception], Mapping[str, JsonValue]],
    _inspect_value("inspect_ai.agent._bridge._errors", "provider_error_payload"),
)
monitor_failure = cast(
    Callable[..., Coroutine[Any, Any, None]],
    _inspect_value("inspect_ai.agent._bridge.sandbox.bridge", "_monitor_failure"),
)
monitor_proxy = cast(
    Callable[..., Coroutine[Any, Any, None]],
    _inspect_value("inspect_ai.agent._bridge.sandbox.bridge", "_monitor_proxy"),
)
register_bridged_tools = cast(
    Callable[..., MCPServerConfigHTTP],
    _inspect_value(
        "inspect_ai.agent._bridge.sandbox.bridge", "_register_bridged_tools"
    ),
)
MODEL_SERVICE = cast(
    str,
    _inspect_value("inspect_ai.agent._bridge.sandbox.service", "MODEL_SERVICE"),
)
call_tool = cast(
    Callable[..., Callable[..., Awaitable[JsonValue]]],
    _inspect_value("inspect_ai.agent._bridge.sandbox.service", "call_tool"),
)
list_tools = cast(
    Callable[..., Callable[..., Awaitable[JsonValue]]],
    _inspect_value("inspect_ai.agent._bridge.sandbox.service", "list_tools"),
)

MessageFingerprint: TypeAlias = Any
Descent: Any = _inspect_value("inspect_ai.agent._bridge.types", "_Descent")
message_fingerprint = cast(
    Callable[..., MessageFingerprint],
    _inspect_value("inspect_ai.agent._bridge.types", "_message_fingerprint"),
)
condensed_fingerprint = cast(
    Callable[..., MessageFingerprint],
    _inspect_value("inspect_ai.agent._bridge.types", "_condensed_fingerprint"),
)
extends = cast(
    Callable[..., bool],
    _inspect_value("inspect_ai.agent._bridge.types", "_extends"),
)
position_descent = cast(
    Callable[..., Any],
    _inspect_value("inspect_ai.agent._bridge.types", "_position_descent"),
)
sandbox_with_injected_tools = cast(
    Callable[..., Awaitable[SandboxEnvironment]],
    _inspect_value(
        "inspect_ai.tool._sandbox_tools_utils.sandbox", "sandbox_with_injected_tools"
    ),
)
inner_exception = cast(
    Callable[[Exception], Exception],
    _inspect_value("inspect_ai.util._anyio", "inner_exception"),
)
SANDBOX_CLI = cast(str, _inspect_value("inspect_ai.util._sandbox._cli", "SANDBOX_CLI"))
sandbox_environments_context_var = cast(
    ContextVar[Mapping[str, SandboxEnvironment]],
    _inspect_value(
        "inspect_ai.util._sandbox.context", "sandbox_environments_context_var"
    ),
)
sandbox_with_environments_context_var = cast(
    ContextVar[dict[str, SandboxEnvironment]],
    _inspect_value(
        "inspect_ai.util._sandbox.context", "sandbox_with_environments_context_var"
    ),
)
sandbox_default_context_var = cast(
    ContextVar[str],
    _inspect_value("inspect_ai.util._sandbox.context", "sandbox_default_context_var"),
)
SandboxEnvironmentProxy = cast(
    Callable[[SandboxEnvironment], SandboxEnvironment],
    _inspect_value("inspect_ai.util._sandbox.events", "SandboxEnvironmentProxy"),
)


def sandbox_tools_user(sandbox: SandboxEnvironment) -> str | None:
    """Return Inspect's injected-tools user or raise a compatibility error."""
    try:
        return cast(str | None, getattr(sandbox, "_tools_user"))
    except AttributeError as error:
        _moved("SandboxEnvironment._tools_user", error)
