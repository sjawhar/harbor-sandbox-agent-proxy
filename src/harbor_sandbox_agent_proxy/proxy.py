"""In-sandbox model proxy backed by caller-supplied host handlers."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from typing import cast

import anyio
from inspect_ai.agent import AgentState, BridgedToolsSpec
from inspect_ai.tool import MCPServerConfigHTTP, ToolCall, ToolInfo
from inspect_ai.util import (
    ExecRemoteStreamingOptions,
    LimitExceededError,
    SandboxEnvironment,
    sandbox_service,
)
from pydantic import JsonValue
from shortuuid import uuid

from ._inspect_compat import (
    MODEL_SERVICE,
    PROVIDER_ERROR_KEY,
    SANDBOX_CLI,
    SandboxAgentBridge,
    call_tool,
    inner_exception,
    list_tools,
    monitor_failure,
    monitor_proxy,
    provider_error_payload,
    register_bridged_tools,
    sandbox_tools_user,
    sandbox_with_injected_tools,
)

logger = getLogger(__name__)

SandboxServiceMethod = Callable[..., Awaitable[JsonValue]]
"""One service method handled by the in-sandbox proxy."""

ModelProxyMethod = Callable[..., Awaitable[JsonValue]]
"""A ``sandbox_model_proxy`` handler, called with the keyword arguments the proxy sends."""

__all__ = [
    "sandbox_model_proxy",
    "ModelProxy",
    "ModelProxyError",
    "ModelProxyMethod",
]

_GENERATE_METHODS = {
    "generate_completions": "the OpenAI Chat Completions API",
    "generate_responses": "the OpenAI Responses API",
    "generate_anthropic": "the Anthropic Messages API",
    "generate_google": "the Google Gemini API",
}
_TOOL_METHODS = ("list_tools", "call_tool")


@dataclass(frozen=True)
class ModelProxy:
    """A model proxy running inside a sandbox (see ``sandbox_model_proxy``)."""

    port: int
    """Port the proxy listens on at ``localhost`` inside the sandbox."""

    mcp_server_configs: list[MCPServerConfigHTTP] = field(default_factory=list)
    """MCP server configs for the bridged tools, one per ``BridgedToolsSpec``."""


class ModelProxyError(Exception):
    """Raised by a ``sandbox_model_proxy`` handler to answer with a provider error.

    The proxy responds with ``status_code`` and the API's own error body built from
    ``message``. Proxy builds that forward it merge ``body`` into the error object of
    the two OpenAI APIs (clients read fields such as ``code`` from it); the Anthropic
    and Gemini routes do not use it.
    """

    status_code: int
    message: str
    body: dict[str, JsonValue] | None

    def __init__(
        self, status_code: int, message: str, body: dict[str, JsonValue] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body


@contextlib.asynccontextmanager
async def sandbox_model_proxy(
    sandbox: SandboxEnvironment,
    *,
    methods: Mapping[str, ModelProxyMethod],
    port: int = 13131,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    polling_interval: float | None = None,
) -> AsyncGenerator[ModelProxy]:
    """Run the agent bridge's in-sandbox model proxy with caller-supplied handlers.

    Injects Inspect's sandbox tools into ``sandbox`` and starts their ``model_proxy``
    process, which serves the OpenAI Chat Completions (``/v1/chat/completions``),
    OpenAI Responses (``/v1/responses``), Anthropic Messages (``/v1/messages``) and
    Google Gemini (``/v1beta/models/*``, ``/models/*``) APIs on ``localhost:<port>``
    inside the sandbox. Each request is passed to the matching handler in ``methods``
    on the host, over the sandbox service file queue, and the handler's result is
    returned to the client. This is the transport ``sandbox_agent_bridge`` uses,
    without Inspect's model layer: a handler may forward the request anywhere, for
    example unchanged to a model gateway. Point an agent in the sandbox at it with
    ``OPENAI_BASE_URL=http://localhost:<port>/v1``,
    ``ANTHROPIC_BASE_URL=http://localhost:<port>`` or
    ``GOOGLE_GEMINI_BASE_URL=http://localhost:<port>`` and a placeholder API key.

    ``bridged_tools`` are served as MCP servers at
    ``http://localhost:<port>/mcp/<name>`` (``ModelProxy.mcp_server_configs``) under
    ``sandbox_agent_bridge``'s rule: a bridged tool runs only for a call the model
    proposed, once per proposal, unless its spec sets ``require_proposal=False``.
    Proposals are read from each successful generate result: a tool call in it (an
    Anthropic ``tool_use`` block, a Chat Completions ``tool_calls`` entry, a Responses
    ``function_call`` item, a Gemini ``functionCall`` part) whose name the request
    declared as a function tool, with a description matching a bridged tool's, grants
    that tool one execution with exactly those arguments. A tool declared any other
    way (a Responses namespace or ``tool_search`` result) grants nothing, so a call to
    it is denied. Unlike ``sandbox_agent_bridge``, this function has no ``approval``
    parameter and applies no ``ApprovalPolicy`` to a bridged tool call: gating is
    execution grants only.

    Args:
        sandbox: Sandbox to run the proxy in. It need not belong to an Inspect sample.
        methods: Handlers by service method name: any of ``generate_completions``,
            ``generate_responses``, ``generate_anthropic`` and ``generate_google``, and,
            only without ``bridged_tools``, both ``list_tools`` and ``call_tool``.
        port: Port the proxy listens on inside the sandbox.
        bridged_tools: Host-side Inspect tools to expose to the sandboxed agent over
            MCP, as for ``sandbox_agent_bridge``.
        polling_interval: Seconds between the host's polls of the sandbox for requests.

    Yields:
        The running proxy, stopped when the block exits.

    Raises:
        ValueError: ``methods`` names an unknown method, only one of ``list_tools`` and
            ``call_tool``, or either of them together with ``bridged_tools``; or two
            bridged tool specs share a name.
        TypeError: A value in ``methods`` is not callable.
        RuntimeError: The proxy process exits while the block runs.
    """
    _validate_methods(methods, bridged_tools)

    bridge = SandboxAgentBridge(
        state=AgentState(messages=[]),
        filter=None,
        retry_refusals=None,
        compaction=None,
        port=port,
        model=None,
    )
    _register_bridged_tool_specs(bridge, bridged_tools or [], port)

    service_methods: dict[str, SandboxServiceMethod] = {
        method: _serve_generate(method, methods.get(method), bridge)
        for method in _GENERATE_METHODS
    }
    if bridged_tools:
        service_methods["list_tools"] = list_tools(bridge)
        service_methods["call_tool"] = call_tool(bridge)
    else:
        service_methods.update(
            {name: methods[name] for name in _TOOL_METHODS if name in methods}
        )

    sandbox_env = await sandbox_with_injected_tools(sandbox=sandbox)
    async with _model_proxy_service(
        sandbox_env,
        service_methods,
        bridge,
        port=port,
        polling_interval=polling_interval,
        caller="sandbox_model_proxy",
    ):
        yield ModelProxy(port=port, mcp_server_configs=list(bridge.mcp_server_configs))


@contextlib.asynccontextmanager
async def _model_proxy_service(
    sandbox_env: SandboxEnvironment,
    methods: dict[str, SandboxServiceMethod],
    bridge: SandboxAgentBridge,
    *,
    port: int,
    polling_interval: float | None,
    caller: str,
) -> AsyncGenerator[None]:
    """Run the in-sandbox model proxy against ``methods`` for the duration of a block."""
    instance = f"proxy_{uuid()}"
    block_completed = False

    try:
        async with anyio.create_task_group() as task_group:

            async def run_service() -> None:
                await sandbox_service(
                    name=MODEL_SERVICE,
                    methods=methods,
                    until=lambda: False,
                    sandbox=sandbox_env,
                    instance=instance,
                    polling_interval=polling_interval,
                    started=started,
                    requires_python=False,
                )

            started = anyio.Event()
            task_group.start_soon(run_service)
            await started.wait()

            proxy = await sandbox_env.exec_remote(
                cmd=[SANDBOX_CLI, "model_proxy"],
                options=ExecRemoteStreamingOptions(
                    concurrency=False,
                    user=sandbox_tools_user(sandbox_env),
                    env={
                        f"{MODEL_SERVICE.upper()}_PORT": str(port),
                        f"{MODEL_SERVICE.upper()}_INSTANCE": instance,
                    },
                    poll_timeout=600,
                ),
            )

            task_group.start_soon(monitor_proxy, proxy)
            task_group.start_soon(monitor_failure, bridge)

            try:
                yield
                block_completed = True
            finally:
                with anyio.CancelScope(shield=True):
                    await proxy.kill()
                task_group.cancel_scope.cancel()
    except Exception as error:
        if block_completed:
            logger.warning(
                "Error during %s cleanup (agent completed successfully): %s",
                caller,
                inner_exception(error),
            )
        else:
            raise inner_exception(error) from error


def _register_bridged_tool_specs(
    bridge: SandboxAgentBridge, specs: Sequence[BridgedToolsSpec], port: int
) -> None:
    """Register each bridged tool spec with the bridge and record its MCP config."""
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate bridged_tools name: '{spec.name}'. "
                + "Each BridgedToolsSpec must have a unique name."
            )
        seen_names.add(spec.name)
        config = register_bridged_tools(bridge, spec, port)
        bridge.mcp_server_configs.append(config)
    bridge.warn_indistinct_tools()


def _validate_methods(
    methods: Mapping[str, ModelProxyMethod],
    bridged_tools: Sequence[BridgedToolsSpec] | None,
) -> None:
    supported = [*_GENERATE_METHODS, *_TOOL_METHODS]
    unknown = sorted(name for name in methods if name not in supported)
    if unknown:
        raise ValueError(
            f"Unknown model proxy method(s): {', '.join(unknown)} "
            + f"(supported: {', '.join(supported)})."
        )
    for name, handler in methods.items():
        if not callable(handler):
            raise TypeError(f"Model proxy method '{name}' is not callable.")
    tool_methods = [name for name in _TOOL_METHODS if name in methods]
    if tool_methods and bridged_tools:
        raise ValueError(
            "Pass either bridged_tools or the list_tools and call_tool methods, "
            + "not both: bridged_tools are served by the proxy's own list_tools "
            + "and call_tool."
        )
    if len(tool_methods) == 1:
        raise ValueError(
            f"Model proxy method '{tool_methods[0]}' needs its counterpart: pass "
            + "both list_tools and call_tool, or neither."
        )


def _serve_generate(
    method: str, handler: ModelProxyMethod | None, bridge: SandboxAgentBridge
) -> SandboxServiceMethod:
    """Serve one model method, forwarding errors and minting bridged-tool grants."""

    async def generate(**params: JsonValue) -> JsonValue:
        if handler is None:
            return _provider_error(
                {
                    "status": 404,
                    "message": (
                        f"This model proxy does not serve {_GENERATE_METHODS[method]}."
                    ),
                }
            )
        try:
            result = await handler(**params)
        except LimitExceededError:
            raise
        except ModelProxyError as error:
            payload: dict[str, JsonValue] = {
                "status": error.status_code,
                "message": error.message,
            }
            if error.body is not None:
                payload["body"] = error.body
            return _provider_error(payload)
        except Exception as error:
            recovered = provider_error_payload(error)
            if recovered["status"] is None:
                logger.warning(
                    "Model proxy handler '%s' raised an exception with no HTTP "
                    + "status (forwarding to the client as an error response): %s",
                    method,
                    error,
                    exc_info=True,
                )
            return _provider_error(dict(recovered))

        if (
            bridge.bridged_tools
            and isinstance(result, dict)
            and PROVIDER_ERROR_KEY not in result
        ):
            calls = _proposed_calls(method, result)
            if calls:
                bridge.register_tool_execution_grants(
                    calls, _declared_tools(method, params.get("json_data"))
                )
        return result

    return generate


def _provider_error(payload: dict[str, JsonValue]) -> JsonValue:
    return {PROVIDER_ERROR_KEY: payload}


def _declared_tools(method: str, request: JsonValue) -> list[ToolInfo]:
    """Return the function tools ``request`` declares to the model."""
    if not isinstance(request, dict):
        return []
    declared: list[tuple[JsonValue, JsonValue]] = []
    for tool in _dicts(request.get("tools")):
        match method:
            case "generate_anthropic":
                if tool.get("type", "custom") == "custom":
                    declared.append((tool.get("name"), tool.get("description")))
            case "generate_completions":
                function = tool.get("function")
                if tool.get("type") == "function" and isinstance(function, dict):
                    declared.append((function.get("name"), function.get("description")))
            case "generate_responses":
                if tool.get("type") == "function":
                    declared.append((tool.get("name"), tool.get("description")))
            case "generate_google":
                for key in ("functionDeclarations", "function_declarations"):
                    declared.extend(
                        (declaration.get("name"), declaration.get("description"))
                        for declaration in _dicts(tool.get(key))
                    )
            case _:
                pass
    if method == "generate_completions":
        declared.extend(
            (function.get("name"), function.get("description"))
            for function in _dicts(request.get("functions"))
        )
    return [
        ToolInfo(
            name=name, description=description if isinstance(description, str) else ""
        )
        for name, description in declared
        if isinstance(name, str)
    ]


def _proposed_calls(method: str, response: dict[str, JsonValue]) -> list[ToolCall]:
    """Return valid function calls the model proposed in a raw provider response."""
    calls: list[ToolCall] = []

    def add(identifier: JsonValue, function: JsonValue, arguments: JsonValue) -> None:
        if (
            isinstance(identifier, str)
            and isinstance(function, str)
            and isinstance(arguments, dict)
        ):
            calls.append(
                ToolCall(id=identifier, function=function, arguments=arguments)
            )

    match method:
        case "generate_anthropic":
            for block in _dicts(response.get("content")):
                if block.get("type") == "tool_use":
                    add(block.get("id"), block.get("name"), block.get("input"))
        case "generate_completions":
            for choice in _dicts(response.get("choices")):
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                for tool_call in _dicts(message.get("tool_calls")):
                    function = tool_call.get("function")
                    if tool_call.get("type", "function") == "function" and isinstance(
                        function, dict
                    ):
                        add(
                            tool_call.get("id"),
                            function.get("name"),
                            _json_object(function.get("arguments")),
                        )
                function_call = message.get("function_call")
                if isinstance(function_call, dict):
                    add(
                        "function_call",
                        function_call.get("name"),
                        _json_object(function_call.get("arguments")),
                    )
        case "generate_responses":
            for item in _dicts(response.get("output")):
                if item.get("type") == "function_call":
                    add(
                        item.get("call_id"),
                        item.get("name"),
                        _json_object(item.get("arguments")),
                    )
        case "generate_google":
            for candidate in _dicts(response.get("candidates")):
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                for part in _dicts(content.get("parts")):
                    function_call = part.get("functionCall", part.get("function_call"))
                    if isinstance(function_call, dict):
                        name = function_call.get("name")
                        add(
                            function_call.get("id", name),
                            name,
                            function_call.get("args"),
                        )
        case _:
            pass
    return calls


def _dicts(value: JsonValue) -> list[dict[str, JsonValue]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _json_object(value: JsonValue) -> JsonValue:
    """Return ``value`` parsed as a JSON object, otherwise ``None``."""
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return cast(JsonValue, parsed) if isinstance(parsed, dict) else None
