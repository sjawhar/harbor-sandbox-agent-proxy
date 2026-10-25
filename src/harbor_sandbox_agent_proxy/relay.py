"""Provider-dialect forwarding with an injected per-request upstream resolver."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import hmac
import json
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Literal

import httpx
import pydantic

from .collaborators import RequestDialect, Upstream

PROVIDER_ERROR_KEY = "__inspect_provider_error__"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_CALLS = 10_000
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_RECORDED_BYTES = 2 * 1024 * 1024 * 1024
GATEWAY_TIMEOUT = httpx.Timeout(900.0, connect=30.0)

RelayPhase = Literal["agent", "verifier"]
PolicyGetter = Callable[[], tuple[RelayPhase, "RelayPolicy"]]
RecordCall = Callable[["RelayCall"], Awaitable[None]]
GenerateMethod = Callable[
    [dict[str, pydantic.JsonValue], Mapping[str, str] | None],
    Awaitable[dict[str, pydantic.JsonValue]],
]


class _ToolKind(enum.StrEnum):
    CLIENT = "client"
    NAMESPACE = "namespace"
    CLIENT_TOOL_SEARCH = "client-tool-search"
    SERVER = "server"


_CODEX_UNSAFE_NAME_CHARS: Final = re.compile(r"[^A-Za-z0-9_]")


@dataclasses.dataclass(frozen=True)
class RelayCall:
    """One request handled by the relay, with all authentication values omitted."""

    phase: RelayPhase
    method: str
    dialect: RequestDialect
    model: str | None
    status: int | None
    latency_ms: float
    request: Mapping[str, pydantic.JsonValue]
    response: pydantic.JsonValue | None
    headers: Mapping[str, str]
    authenticated: bool
    stripped: tuple[str, ...]
    refused: str | None = None


@dataclasses.dataclass(frozen=True)
class RelayPolicy:
    """The current sandbox caller and the model/tool limits for one relay phase."""

    caller_key: str
    allowed_models: frozenset[str]
    tools: Literal["any", "bridged-only"] = "any"
    permitted_tool_names: frozenset[str] = frozenset()
    permitted_tool_prefixes: tuple[str, ...] = ()
    bridged: Mapping[str, frozenset[str]] = dataclasses.field(
        default_factory=lambda: {}
    )
    allowed_betas: frozenset[str] | None = None
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_calls: int = DEFAULT_MAX_CALLS
    max_recorded_bytes: int = DEFAULT_MAX_RECORDED_BYTES

    def permits_tool(self, name: str | None) -> bool:
        if self.tools == "any":
            return True
        if not name:
            return False
        if name in self.permitted_tool_names:
            return True
        for server, served in self.bridged.items():
            for prefix in bridged_tool_prefixes(server):
                if name.startswith(prefix):
                    return name.removeprefix(prefix) in served
        return name.startswith(self.permitted_tool_prefixes)

    def namespace_tools(self, namespace: str | None) -> frozenset[str] | None:
        """Return a bridged server's tools when ``namespace`` is its own."""
        if not namespace:
            return None
        for server, served in self.bridged.items():
            if namespace == bridged_namespace(server):
                return served
        return None

    def permits_namespaced_tool(self, namespace: str | None, name: str | None) -> bool:
        """Return whether a Responses namespace tool is within this phase's surface."""
        if self.tools == "any":
            return True
        if not name:
            return False
        served = self.namespace_tools(namespace)
        return name in self.permitted_tool_names or (
            served is not None and name in served
        )


def mint_caller_key() -> str:
    """Create the sole credential the relay exposes to a sandbox phase."""
    return secrets.token_urlsafe(32)


def bridged_namespace(server: str) -> str:
    """Return the Responses namespace form of a bridged server name."""
    return "mcp__" + _CODEX_UNSAFE_NAME_CHARS.sub("_", server)


def bridged_tool_prefixes(server: str) -> tuple[str, ...]:
    """Return the Messages and Responses function-name prefixes for one server."""
    prefixes = (f"mcp__{server}__", f"{bridged_namespace(server)}__")
    return tuple(dict.fromkeys(prefixes))


def relay_methods(
    policy_getter: PolicyGetter,
    upstream: Upstream,
    *,
    record: RecordCall,
    client: httpx.AsyncClient,
) -> Mapping[str, GenerateMethod]:
    """Build handlers for the proxy's four provider-dialect endpoints."""
    limiter = _PhaseLimiter()

    def handler(method: str, dialect: RequestDialect) -> GenerateMethod:
        async def forward(
            json_data: dict[str, pydantic.JsonValue],
            headers: Mapping[str, str] | None = None,
        ) -> dict[str, pydantic.JsonValue]:
            phase, policy = policy_getter()
            started = datetime.datetime.now(datetime.UTC)
            model = _model(json_data.get("model"))
            request = json_data
            if not _authenticated(headers, policy.caller_key):
                return await _refuse(
                    record,
                    phase=phase,
                    method=method,
                    dialect=dialect,
                    model=model,
                    request=request,
                    reason="the sandbox caller is not authenticated",
                )
            try:
                _validate_model(model, policy.allowed_models)
                encoded = json.dumps(request, separators=(",", ":")).encode("utf-8")
                if len(encoded) > policy.max_request_bytes:
                    raise _RejectedRequest("request cap exceeded")
                limiter.reserve(phase, len(encoded), policy)
                stripped = _confine(request, dialect, policy)
                if dialect is RequestDialect.CHAT_COMPLETIONS and not request.get(
                    "tools"
                ):
                    request.pop("parallel_tool_calls", None)
                if dialect is RequestDialect.GEMINI:
                    request.pop("model", None)
                target = await upstream.resolve(dialect=dialect, model=model or "")
                headers, beta_stripped = _headers(
                    target.headers, dialect, headers, policy
                )
                stripped.extend(beta_stripped)
                response = await client.post(target.url, json=request, headers=headers)
                response_body = _response_body(response)
                if dialect is RequestDialect.GEMINI and response.is_success:
                    response_body = _gemini_body(response_body)
                completed = datetime.datetime.now(datetime.UTC)
                safe_headers = _safe_headers(headers)
                call = RelayCall(
                    phase=phase,
                    method=method,
                    dialect=dialect,
                    model=model,
                    status=response.status_code,
                    latency_ms=(completed - started).total_seconds() * 1000,
                    request=request,
                    response=response_body,
                    headers=safe_headers,
                    authenticated=True,
                    stripped=tuple(stripped),
                )
                await record(call)
                if response.is_error:
                    return _error(
                        response.status_code,
                        _upstream_message(method, response.status_code, response_body),
                        response_body,
                    )
                return _json_object(response_body)
            except _RejectedRequest as error:
                return await _refuse(
                    record,
                    phase=phase,
                    method=method,
                    dialect=dialect,
                    model=model,
                    request=request,
                    reason=error.message,
                )
            except (httpx.HTTPError, ValueError) as error:
                return _error(None, f"{method}: upstream request failed: {error}", None)

        return forward

    return {
        "generate_anthropic": handler("generate_anthropic", RequestDialect.MESSAGES),
        "generate_completions": handler(
            "generate_completions", RequestDialect.CHAT_COMPLETIONS
        ),
        "generate_responses": handler("generate_responses", RequestDialect.RESPONSES),
        "generate_google": handler("generate_google", RequestDialect.GEMINI),
    }


class _PhaseLimiter:
    """Per-forwarder caps that reset when the relay phase changes."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._bytes: dict[str, int] = {}

    def reserve(
        self, phase: RelayPhase, request_bytes: int, policy: RelayPolicy
    ) -> None:
        count = self._counts.get(phase, 0)
        used = self._bytes.get(phase, 0)
        if count >= policy.max_calls:
            raise _RejectedRequest("call cap exceeded")
        if used + request_bytes > policy.max_recorded_bytes:
            raise _RejectedRequest("recorded-byte cap exceeded")
        self._counts[phase] = count + 1
        self._bytes[phase] = used + request_bytes


@dataclasses.dataclass(frozen=True)
class _RejectedRequest(Exception):
    message: str


def _authenticated(headers: Mapping[str, str] | None, key: str) -> bool:
    if headers is None:
        return False
    supplied = next(
        (
            value
            for name, value in headers.items()
            if name.lower() in {"x-api-key", "authorization"}
        ),
        "",
    )
    if supplied.startswith("Bearer "):
        supplied = supplied.removeprefix("Bearer ")
    return hmac.compare_digest(supplied, key)


def _model(value: pydantic.JsonValue | None) -> str | None:
    return value if isinstance(value, str) else None


def _normalised_model(model: str) -> str:
    prefix, separator, value = model.partition("/")
    return value if separator and prefix in {"anthropic", "openai", "google"} else model


def _validate_model(model: str | None, allowed: frozenset[str]) -> None:
    if model is None:
        raise _RejectedRequest("request names no model")
    permitted = {_normalised_model(item) for item in allowed}
    if _normalised_model(model) not in permitted:
        raise _RejectedRequest("model is not one the launch declared")


def _headers(
    upstream: Mapping[str, str],
    dialect: RequestDialect,
    incoming: Mapping[str, str] | None,
    policy: RelayPolicy,
) -> tuple[dict[str, str], list[str]]:
    headers = dict(upstream)
    stripped: list[str] = []
    if dialect is RequestDialect.MESSAGES:
        headers["anthropic-version"] = (
            _header(incoming, "anthropic-version") or DEFAULT_ANTHROPIC_VERSION
        )
        beta = _header(incoming, "anthropic-beta")
        if beta:
            kept: list[str] = []
            for item in (part.strip() for part in beta.split(",")):
                if policy.allowed_betas is None or item in policy.allowed_betas:
                    kept.append(item)
                else:
                    stripped.append(f"anthropic-beta:{item}")
            if kept:
                headers["anthropic-beta"] = ",".join(kept)
    return headers, stripped


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    return next(
        (value for key, value in (headers or {}).items() if key.lower() == name), None
    )


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in {"authorization", "x-api-key", "x-goog-api-key"}
    }


def _response_body(response: httpx.Response) -> pydantic.JsonValue:
    try:
        return pydantic.TypeAdapter(pydantic.JsonValue).validate_python(response.json())
    except ValueError:
        return response.text


def _json_object(value: pydantic.JsonValue) -> dict[str, pydantic.JsonValue]:
    return value if isinstance(value, dict) else {"value": value}


def _gemini_body(value: pydantic.JsonValue) -> pydantic.JsonValue:
    if isinstance(value, str):
        events = [
            line.removeprefix("data:").strip()
            for line in value.splitlines()
            if line.startswith("data:")
        ]
        if events:
            return pydantic.TypeAdapter(pydantic.JsonValue).validate_json(events[-1])
    return value


def _upstream_message(method: str, status: int, body: pydantic.JsonValue) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return f"{method}: upstream returned HTTP {status}: {message}"
    return f"{method}: upstream returned HTTP {status}"


def _error(
    status: int | None, message: str, body: pydantic.JsonValue | None
) -> dict[str, pydantic.JsonValue]:
    detail: dict[str, pydantic.JsonValue] = {"status": status, "message": message}
    if body is not None:
        detail["body"] = body
    return {PROVIDER_ERROR_KEY: detail}


async def _refuse(
    record: RecordCall,
    *,
    phase: RelayPhase,
    method: str,
    dialect: RequestDialect,
    model: str | None,
    request: Mapping[str, pydantic.JsonValue],
    reason: str,
) -> dict[str, pydantic.JsonValue]:
    await record(
        RelayCall(
            phase=phase,
            method=method,
            dialect=dialect,
            model=model,
            status=None,
            latency_ms=0.0,
            request=request,
            response=None,
            headers={},
            authenticated=False,
            stripped=(),
            refused=reason,
        )
    )
    return _error(None, reason, None)


def _confine(
    body: dict[str, pydantic.JsonValue], dialect: RequestDialect, policy: RelayPolicy
) -> list[str]:
    """Remove provider tools and account-file references from a request in place."""
    stripped = (
        _confine_responses_tools(body, policy)
        if dialect is RequestDialect.RESPONSES
        else _confine_standard_tools(body, policy)
    )
    for key in ("mcp_servers", "container", "tool_resources"):
        if key in body:
            body.pop(key)
            stripped.append(key)
    _strip_file_references(body, stripped)
    return stripped


def _confine_standard_tools(
    body: dict[str, pydantic.JsonValue], policy: RelayPolicy
) -> list[str]:
    """Confine non-Responses client tools in place."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    stripped: list[str] = []
    kept: list[pydantic.JsonValue] = []
    for tool in tools:
        if not isinstance(tool, dict):
            stripped.append("tool:malformed")
            continue
        name = _tool_name(tool)
        kind = tool.get("type")
        if kind not in {None, "function", "custom"}:
            stripped.append(f"server-tool:{kind}")
        elif policy.permits_tool(name):
            kept.append(tool)
        else:
            stripped.append(f"tool:{name}")
    if kept:
        body["tools"] = kept
    else:
        body.pop("tools", None)
    return stripped


def _responses_tool(
    tool: dict[str, pydantic.JsonValue],
) -> tuple[_ToolKind, str | None]:
    """Classify a Responses tool as client-owned or provider-owned."""
    kind = tool.get("type")
    name = tool.get("name")
    tool_name = name if isinstance(name, str) else None
    if kind in {"function", "custom"}:
        return _ToolKind.CLIENT, tool_name
    if kind == "namespace":
        return _ToolKind.NAMESPACE, tool_name
    if kind == "tool_search" and tool.get("execution") == "client":
        return _ToolKind.CLIENT_TOOL_SEARCH, "tool_search"
    return _ToolKind.SERVER, str(kind) if kind is not None else "untyped"


def _confine_responses_tool_list(
    tools: list[pydantic.JsonValue], policy: RelayPolicy
) -> tuple[list[pydantic.JsonValue], list[str]]:
    """Return the Responses tools this phase admits and the tools it removed."""
    stripped: list[str] = []
    kept: list[pydantic.JsonValue] = []
    for tool in tools:
        if not isinstance(tool, dict):
            stripped.append("tool:malformed")
            continue
        kind, name = _responses_tool(tool)
        if kind is _ToolKind.SERVER:
            stripped.append(f"server-tool:{name}")
        elif kind is _ToolKind.CLIENT_TOOL_SEARCH:
            kept.append(tool)
        elif kind is _ToolKind.CLIENT:
            if policy.permits_tool(name):
                kept.append(tool)
            else:
                stripped.append(f"tool:{name}")
        else:
            namespace, namespace_stripped = _confine_responses_namespace(
                tool, name, policy
            )
            stripped.extend(namespace_stripped)
            if namespace is not None:
                kept.append(namespace)
    return kept, stripped


def _confine_responses_namespace(
    namespace: dict[str, pydantic.JsonValue],
    name: str | None,
    policy: RelayPolicy,
) -> tuple[dict[str, pydantic.JsonValue] | None, list[str]]:
    """Confine client functions nested in one Responses namespace."""
    stripped: list[str] = []
    kept: list[pydantic.JsonValue] = []
    tools = namespace.get("tools")
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            stripped.append(f"tool:{name}/malformed")
            continue
        kind, tool_name = _responses_tool(tool)
        if kind is _ToolKind.SERVER:
            stripped.append(f"server-tool:{tool_name}")
        elif kind is _ToolKind.CLIENT and policy.permits_namespaced_tool(
            name, tool_name
        ):
            kept.append(tool)
        else:
            stripped.append(f"tool:{name}/{tool_name}")
    if not kept:
        return None, [*stripped, f"namespace:{name}"]
    return {**namespace, "tools": kept}, stripped


def _confine_responses_tools(
    body: dict[str, pydantic.JsonValue], policy: RelayPolicy
) -> list[str]:
    """Apply Responses confinement at each Codex tool declaration site."""
    stripped: list[str] = []
    tools = body.get("tools")
    if isinstance(tools, list):
        kept, tool_stripped = _confine_responses_tool_list(tools, policy)
        stripped.extend(tool_stripped)
        if kept:
            body["tools"] = kept
        else:
            body.pop("tools", None)
    items = body.get("input")
    if not isinstance(items, list):
        return stripped
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in {
            "additional_tools",
            "tool_search_output",
        }:
            continue
        declared = item.get("tools")
        if not isinstance(declared, list):
            continue
        kept, item_stripped = _confine_responses_tool_list(declared, policy)
        item["tools"] = kept
        stripped.extend(item_stripped)
    return stripped


def _tool_name(tool: Mapping[str, pydantic.JsonValue]) -> str | None:
    value = tool.get("name")
    if isinstance(value, str):
        return value
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return name
    return None


def _strip_file_references(
    value: pydantic.JsonValue, stripped: list[str]
) -> pydantic.JsonValue:
    if isinstance(value, list):
        result: list[pydantic.JsonValue] = []
        for item in value:
            if isinstance(item, dict) and ("file_id" in item or "fileData" in item):
                stripped.append("content:file")
                continue
            result.append(_strip_file_references(item, stripped))
        return result
    if isinstance(value, dict):
        for key, item in tuple(value.items()):
            value[key] = _strip_file_references(item, stripped)
    return value
