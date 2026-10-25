"""Provider-dialect forwarding with an injected per-request upstream resolver."""

from __future__ import annotations

import dataclasses
import datetime
import hmac
import json
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

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
    bridged: Mapping[str, frozenset[str]] = dataclasses.field(default_factory=dict)
    allowed_betas: frozenset[str] | None = None
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_calls: int = DEFAULT_MAX_CALLS
    max_recorded_bytes: int = DEFAULT_MAX_RECORDED_BYTES

    def permits_tool(self, name: str | None) -> bool:
        if self.tools == "any":
            return True
        if name is None:
            return False
        if name in self.permitted_tool_names or name.startswith(self.permitted_tool_prefixes):
            return True
        return any(
            name.startswith(f"mcp__{server}__") and name.removeprefix(f"mcp__{server}__") in tools
            for server, tools in self.bridged.items()
        )


def mint_caller_key() -> str:
    """Create the sole credential the relay exposes to a sandbox phase."""
    return secrets.token_urlsafe(32)


def bridged_namespace(server: str) -> str:
    """Return the Responses namespace form of a bridged server name."""
    return "mcp__" + "".join(character if character.isalnum() or character == "_" else "_" for character in server)


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
            body: dict[str, pydantic.JsonValue], incoming_headers: Mapping[str, str] | None = None
        ) -> dict[str, pydantic.JsonValue]:
            phase, policy = policy_getter()
            started = datetime.datetime.now(datetime.UTC)
            model = _model(body.get("model"))
            request = body
            if not _authenticated(incoming_headers, policy.caller_key):
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
                if dialect is RequestDialect.CHAT_COMPLETIONS and not request.get("tools"):
                    request.pop("parallel_tool_calls", None)
                if dialect is RequestDialect.GEMINI:
                    request.pop("model", None)
                target = await upstream.resolve(dialect=dialect, model=model or "")
                headers, beta_stripped = _headers(target.headers, dialect, incoming_headers, policy)
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
                    return _error(response.status_code, _upstream_message(method, response.status_code, response_body), response_body)
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
        "generate_completions": handler("generate_completions", RequestDialect.CHAT_COMPLETIONS),
        "generate_responses": handler("generate_responses", RequestDialect.RESPONSES),
        "generate_google": handler("generate_google", RequestDialect.GEMINI),
    }


class _PhaseLimiter:
    """Per-forwarder caps that reset when the relay phase changes."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._bytes: dict[str, int] = {}

    def reserve(self, phase: RelayPhase, request_bytes: int, policy: RelayPolicy) -> None:
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
    supplied = next((value for name, value in headers.items() if name.lower() in {"x-api-key", "authorization"}), "")
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
        headers["anthropic-version"] = _header(incoming, "anthropic-version") or DEFAULT_ANTHROPIC_VERSION
        beta = _header(incoming, "anthropic-beta")
        if beta:
            kept = []
            for item in (part.strip() for part in beta.split(",")):
                if policy.allowed_betas is None or item in policy.allowed_betas:
                    kept.append(item)
                else:
                    stripped.append(f"anthropic-beta:{item}")
            if kept:
                headers["anthropic-beta"] = ",".join(kept)
    return headers, stripped


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    return next((value for key, value in (headers or {}).items() if key.lower() == name), None)


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in {"authorization", "x-api-key", "x-goog-api-key"}}


def _response_body(response: httpx.Response) -> pydantic.JsonValue:
    try:
        return pydantic.TypeAdapter(pydantic.JsonValue).validate_python(response.json())
    except ValueError:
        return response.text


def _json_object(value: pydantic.JsonValue) -> dict[str, pydantic.JsonValue]:
    return value if isinstance(value, dict) else {"value": value}


def _gemini_body(value: pydantic.JsonValue) -> pydantic.JsonValue:
    if isinstance(value, str):
        events = [line.removeprefix("data:").strip() for line in value.splitlines() if line.startswith("data:")]
        if events:
            return json.loads(events[-1])
    return value


def _upstream_message(method: str, status: int, body: pydantic.JsonValue) -> str:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
        if isinstance(message, str):
            return f"{method}: upstream returned HTTP {status}: {message}"
    return f"{method}: upstream returned HTTP {status}"


def _error(status: int | None, message: str, body: pydantic.JsonValue | None) -> dict[str, pydantic.JsonValue]:
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
    """Remove provider-owned tools and account-file references from a request in place."""
    stripped: list[str] = []
    tools = body.get("tools")
    if isinstance(tools, list):
        kept: list[pydantic.JsonValue] = []
        for tool in tools:
            if not isinstance(tool, dict):
                stripped.append("tool:malformed")
                continue
            name = _tool_name(tool)
            kind = tool.get("type")
            client_tool = kind in {None, "function", "custom"}
            if not client_tool:
                stripped.append(f"server-tool:{kind}")
            elif policy.permits_tool(name):
                kept.append(tool)
            else:
                stripped.append(f"tool:{name}")
        if kept:
            body["tools"] = kept
        else:
            body.pop("tools", None)
    for key in ("mcp_servers", "container", "tool_resources"):
        if key in body:
            body.pop(key)
            stripped.append(key)
    _strip_file_references(body, stripped)
    return stripped


def _tool_name(tool: Mapping[str, pydantic.JsonValue]) -> str | None:
    value = tool.get("name")
    if isinstance(value, str):
        return value
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _strip_file_references(value: pydantic.JsonValue, stripped: list[str]) -> pydantic.JsonValue:
    if isinstance(value, list):
        result = []
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
