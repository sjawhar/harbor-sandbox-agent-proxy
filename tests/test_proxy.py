"""Tests for ``sandbox_model_proxy``."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncGenerator
from importlib import import_module
from typing import Any, final
from unittest.mock import MagicMock

import pytest
from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import tool
from inspect_ai.util import LimitExceededError, SandboxEnvironment, sandbox
from pydantic import JsonValue

from harbor_sandbox_agent_proxy import proxy as proxy_module
from harbor_sandbox_agent_proxy._inspect_compat import (
    PROVIDER_ERROR_KEY,
    SandboxAgentBridge,
)
from harbor_sandbox_agent_proxy.proxy import (
    ModelProxy,
    ModelProxyError,
    sandbox_model_proxy,
)
from tests.conftest import skip_if_no_docker

inspect_eval = getattr(import_module("inspect_ai"), "eval")

GENERATE_METHODS = {
    "generate_completions",
    "generate_responses",
    "generate_anthropic",
    "generate_google",
}

ADD_DESCRIPTION = "Add two numbers."

FIXED_MESSAGE: dict[str, JsonValue] = {
    "id": "msg_fixed_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-fixed",
    "content": [{"type": "text", "text": "The proxy returned this unchanged."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 7},
}


@tool
def calculator_add(call_log: list[dict[str, int]]):
    async def execute(x: int, y: int) -> str:
        """Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.
        """
        call_log.append({"x": x, "y": y})
        return str(x + y)

    return execute


def _unused_sandbox() -> MagicMock:
    return MagicMock(spec=SandboxEnvironment)


async def _ok(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
    del json_data
    return FIXED_MESSAGE


async def test_unknown_method_is_refused() -> None:
    sandbox_env = _unused_sandbox()
    with pytest.raises(ValueError, match="generate_bedrock"):
        async with sandbox_model_proxy(
            sandbox_env, methods={"generate_anthropic": _ok, "generate_bedrock": _ok}
        ):
            pass
    assert sandbox_env.method_calls == []


async def test_non_callable_method_is_refused() -> None:
    sandbox_env = _unused_sandbox()
    methods: dict[str, Any] = {"generate_anthropic": "not a handler"}
    with pytest.raises(TypeError, match="generate_anthropic"):
        async with sandbox_model_proxy(sandbox_env, methods=methods):
            pass
    assert sandbox_env.method_calls == []


async def test_tool_methods_with_bridged_tools_are_refused() -> None:
    async def list_tools(server: str) -> JsonValue:
        del server
        return []

    async def call_tool(server: str, tool: str, arguments: JsonValue) -> JsonValue:
        del server, tool, arguments
        return ""

    with pytest.raises(ValueError, match="not both"):
        async with sandbox_model_proxy(
            _unused_sandbox(),
            methods={"list_tools": list_tools, "call_tool": call_tool},
            bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
        ):
            pass


async def test_one_tool_method_without_the_other_is_refused() -> None:
    async def list_tools(server: str) -> JsonValue:
        del server
        return []

    with pytest.raises(ValueError, match="counterpart"):
        async with sandbox_model_proxy(
            _unused_sandbox(), methods={"list_tools": list_tools}
        ):
            pass


class _ServedTable:
    """Captures what ``sandbox_model_proxy`` hands the proxy runner."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.methods: dict[str, proxy_module.SandboxServiceMethod] = {}
        self.kwargs: dict[str, Any] = {}

        async def inject(**kwargs: Any) -> SandboxEnvironment:
            del kwargs
            return _unused_sandbox()

        @contextlib.asynccontextmanager
        async def runner(
            sandbox_env: SandboxEnvironment,
            methods: dict[str, proxy_module.SandboxServiceMethod],
            bridge: SandboxAgentBridge,
            **kwargs: Any,
        ) -> AsyncGenerator[None]:
            del sandbox_env, bridge
            self.methods = methods
            self.kwargs = kwargs
            yield

        monkeypatch.setattr(proxy_module, "sandbox_with_injected_tools", inject)
        monkeypatch.setattr(proxy_module, "_model_proxy_service", runner)


async def test_seam_serves_every_generate_method_and_its_bridged_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={"generate_anthropic": _ok},
        port=14141,
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
        polling_interval=0.5,
    ) as model_proxy:
        assert set(served.methods) == GENERATE_METHODS | {"list_tools", "call_tool"}
        assert served.kwargs["polling_interval"] == 0.5
        assert model_proxy == ModelProxy(
            port=14141, mcp_server_configs=model_proxy.mcp_server_configs
        )
        assert [config.url for config in model_proxy.mcp_server_configs] == [
            "http://localhost:14141/mcp/calc"
        ]
        listed = await served.methods["list_tools"](server="calc")
        assert isinstance(listed, list) and isinstance(listed[0], dict)
        assert (listed[0]["name"], listed[0]["description"]) == (
            "calculator_add",
            ADD_DESCRIPTION,
        )


async def test_seam_serves_caller_tool_methods_without_bridged_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_tools(server: str) -> JsonValue:
        return [{"name": f"{server}-tool"}]

    async def call_tool(server: str, tool: str, arguments: JsonValue) -> JsonValue:
        del arguments
        return f"{server}/{tool}"

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"list_tools": list_tools, "call_tool": call_tool}
    ) as model_proxy:
        assert model_proxy.mcp_server_configs == []
        assert await served.methods["list_tools"](server="s") == [{"name": "s-tool"}]
        assert (
            await served.methods["call_tool"](server="s", tool="t", arguments={})
            == "s/t"
        )


@final
class _StatusError(Exception):
    status_code = 503


@pytest.mark.parametrize(
    "raised,payload",
    [
        (
            ModelProxyError(529, "overloaded", body={"code": "busy"}),
            {"status": 529, "message": "overloaded", "body": {"code": "busy"}},
        ),
        (ModelProxyError(429, "slow down"), {"status": 429, "message": "slow down"}),
        (
            _StatusError("gateway unavailable"),
            {"status": 503, "message": "gateway unavailable"},
        ),
        (RuntimeError("handler bug"), {"status": None, "message": "handler bug"}),
    ],
)
async def test_handler_exception_becomes_provider_error(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, payload: dict[str, JsonValue]
) -> None:
    async def failing(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        del json_data
        raise raised

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_responses": failing}
    ):
        result = await served.methods["generate_responses"](json_data={"model": "m"})
    assert result == {PROVIDER_ERROR_KEY: payload}


async def test_handler_limit_exceeded_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def over_limit(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        del json_data
        raise LimitExceededError("token", value=10, limit=5)

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_anthropic": over_limit}
    ):
        with pytest.raises(LimitExceededError):
            await served.methods["generate_anthropic"](json_data={"model": "m"})


async def test_missing_generate_method_answers_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_anthropic": _ok}
    ):
        result = await served.methods["generate_completions"](
            json_data={"model": "m"}, headers={"x-api-key": "placeholder"}
        )
    assert result == {
        PROVIDER_ERROR_KEY: {
            "status": 404,
            "message": "This model proxy does not serve the OpenAI Chat Completions API.",
        }
    }


async def test_handler_receives_proxy_params_and_result_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, JsonValue]] = []
    error_result: JsonValue = {PROVIDER_ERROR_KEY: {"status": 418, "message": "teapot"}}

    async def recording(**params: JsonValue) -> JsonValue:
        received.append(params)
        return error_result if params.get("headers") else FIXED_MESSAGE

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_anthropic": recording}
    ):
        method = served.methods["generate_anthropic"]
        assert await method(json_data={"model": "m"}) == FIXED_MESSAGE
        assert (
            await method(
                json_data={"model": "m"},
                headers={"anthropic-beta": "b"},
                metadata_headers=None,
            )
            == error_result
        )
    assert received == [
        {"json_data": {"model": "m"}},
        {
            "json_data": {"model": "m"},
            "headers": {"anthropic-beta": "b"},
            "metadata_headers": None,
        },
    ]


_ADD_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
}

_DIALECTS: list[tuple[str, dict[str, JsonValue], dict[str, JsonValue]]] = [
    (
        "generate_anthropic",
        {
            "model": "m",
            "tools": [
                {
                    "name": "mcp__calc__calculator_add",
                    "description": ADD_DESCRIPTION,
                    "input_schema": _ADD_SCHEMA,
                }
            ],
        },
        {
            "content": [
                {"type": "text", "text": "adding"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "mcp__calc__calculator_add",
                    "input": {"x": 2, "y": 3},
                },
            ]
        },
    ),
    (
        "generate_completions",
        {
            "model": "m",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calc_add",
                        "description": ADD_DESCRIPTION,
                        "parameters": _ADD_SCHEMA,
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calc_add",
                                    "arguments": '{"x": 2, "y": 3}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    ),
    (
        "generate_responses",
        {
            "model": "m",
            "tools": [
                {
                    "type": "function",
                    "name": "calc_add",
                    "description": ADD_DESCRIPTION,
                    "parameters": _ADD_SCHEMA,
                }
            ],
        },
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "calc_add",
                    "arguments": '{"x": 2, "y": 3}',
                }
            ]
        },
    ),
    (
        "generate_google",
        {
            "model": "m",
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "calc_add",
                            "description": ADD_DESCRIPTION,
                            "parameters": _ADD_SCHEMA,
                        }
                    ]
                }
            ],
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "calc_add",
                                    "args": {"x": 2, "y": 3},
                                }
                            }
                        ],
                    }
                }
            ]
        },
    ),
]


@pytest.mark.parametrize("method,request_body,response", _DIALECTS)
async def test_raw_response_grants_one_execution_of_the_proposed_call(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    request_body: dict[str, JsonValue],
    response: dict[str, JsonValue],
) -> None:
    call_log: list[dict[str, int]] = []

    async def raw(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        del json_data
        return response

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={method: raw},
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add(call_log)])],
    ):
        call_tool = served.methods["call_tool"]
        arguments = {"x": 2, "y": 3}

        # nothing proposed yet
        with pytest.raises(PermissionError, match="not proposed"):
            await call_tool(server="calc", tool="calculator_add", arguments=arguments)

        # the raw response is returned unchanged and grants the proposed call once
        assert await served.methods[method](json_data=request_body) == response
        with pytest.raises(PermissionError):
            await call_tool(
                server="calc", tool="calculator_add", arguments={"x": 2, "y": 4}
            )
        assert (
            await call_tool(server="calc", tool="calculator_add", arguments=arguments)
            == "5"
        )
        with pytest.raises(PermissionError):
            await call_tool(server="calc", tool="calculator_add", arguments=arguments)

    assert call_log == [{"x": 2, "y": 3}]


async def test_call_to_an_undeclared_name_grants_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, request_body, response = _DIALECTS[0]
    undeclared = {**request_body, "tools": []}

    async def raw(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        del json_data
        return response

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={method: raw},
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
    ):
        assert await served.methods[method](json_data=undeclared) == response
        with pytest.raises(PermissionError):
            await served.methods["call_tool"](
                server="calc", tool="calculator_add", arguments={"x": 2, "y": 3}
            )


async def test_provider_error_result_grants_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, request_body, response = _DIALECTS[0]
    error_with_calls: dict[str, JsonValue] = {
        **response,
        PROVIDER_ERROR_KEY: {"status": 500, "message": "x"},
    }

    async def raw(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        del json_data
        return error_with_calls

    served = _ServedTable(monkeypatch)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={method: raw},
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
    ):
        await served.methods[method](json_data=request_body)
        with pytest.raises(PermissionError):
            await served.methods["call_tool"](
                server="calc", tool="calculator_add", arguments={"x": 2, "y": 3}
            )


PORT = 13131


async def _curl(
    path: str, body: dict[str, JsonValue] | None = None, *, method: str = "POST"
) -> tuple[int, str]:
    """POST ``body`` to the proxy from inside the sandbox; (HTTP status, response body)."""
    cmd = [
        "curl",
        "-sS",
        "--retry",
        "30",
        "--retry-connrefused",
        "--retry-delay",
        "1",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
        "-w",
        "\n%{http_code}",
        f"http://localhost:{PORT}{path}",
    ]
    if body is not None:
        cmd[-1:-1] = ["-d", json.dumps(body)]
    result = await sandbox().exec(cmd, timeout=120)
    assert result.success, result.stderr
    response, _, status = result.stdout.rpartition("\n")
    return int(status), response


async def _mcp(
    method: str, params: dict[str, JsonValue] | None = None
) -> dict[str, Any]:
    request: dict[str, JsonValue] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        request["params"] = params
    status, response = await _curl("/mcp/calc", request)
    assert status == 200, response
    parsed: dict[str, Any] = json.loads(response)
    return parsed


def _anthropic_request(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def _parse_with_anthropic_sdk(sse: str) -> Any:
    """Read ``sse`` as the Anthropic SDK would read a streamed Messages response."""
    import anthropic
    import httpx2

    def respond(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode("utf-8"),
        )

    client = anthropic.Anthropic(
        api_key="placeholder",
        base_url=f"http://localhost:{PORT}",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )
    with client.messages.stream(
        model="claude-fixed",
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        return stream.get_final_message()


@task
def proxy_task(test_solver: Any) -> Task:
    return Task(
        dataset=[Sample(input="Test", target="Test")],
        solver=[test_solver],
        scorer=includes(),
        sandbox="docker",
    )


def _assert_proxy_checks(
    checks: dict[str, Any], handled: list[str], call_log: list[dict[str, int]]
) -> None:
    assert checks["mcp_urls"] == [f"http://localhost:{PORT}/mcp/calc"]
    assert checks["fixed"] == (200, json.dumps(FIXED_MESSAGE))

    status, sse = checks["stream"]
    assert status == 200
    assert "event: message_start" in sse and "event: message_stop" in sse
    message = _parse_with_anthropic_sdk(sse)
    assert (message.id, message.model) in {
        (FIXED_MESSAGE["id"], FIXED_MESSAGE["model"]),
        (message.id, "fixed"),
    }
    assert message.id.startswith("msg_")
    assert message.stop_reason == "end_turn"
    assert [block.text for block in message.content] == [
        "The proxy returned this unchanged."
    ]
    assert message.usage.output_tokens == 7

    assert checks["overloaded"] == (
        529,
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "overloaded right now",
                },
            }
        ),
    )
    status, body = checks["bug"]
    assert status == 400
    assert json.loads(body)["error"]["message"] == "the handler has a bug"
    status, body = checks["missing"]
    assert status == 404
    assert json.loads(body)["error"]["message"] == (
        "This model proxy does not serve the OpenAI Chat Completions API."
    )
    assert checks["after_errors"] == (200, json.dumps(FIXED_MESSAGE))
    assert handled == ["fixed", "fixed", "overloaded", "bug", "fixed", "tool"]

    assert [
        tool_info["name"] for tool_info in checks["tools_list"]["result"]["tools"]
    ] == ["calculator_add"]
    assert "was not proposed" in checks["call_unproposed"]["error"]["message"]
    assert checks["propose"][0] == 200
    assert json.loads(checks["propose"][1])["content"][0]["type"] == "tool_use"
    assert checks["call_proposed"]["result"]["content"] == [
        {"type": "text", "text": "8"}
    ]
    assert "was not proposed" in checks["call_again"]["error"]["message"]
    assert call_log == [{"x": 5, "y": 3}]


@pytest.mark.docker
def test_sandbox_model_proxy_end_to_end() -> None:
    skip_if_no_docker()
    call_log: list[dict[str, int]] = []
    handled: list[str] = []
    checks: dict[str, Any] = {}

    async def generate_anthropic(
        json_data: dict[str, JsonValue], **_: JsonValue
    ) -> JsonValue:
        model = json_data["model"]
        assert isinstance(model, str)
        handled.append(model)
        match model:
            case "overloaded":
                raise ModelProxyError(529, "overloaded right now")
            case "bug":
                raise RuntimeError("the handler has a bug")
            case "tool":
                return {
                    **FIXED_MESSAGE,
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "mcp__calc__calculator_add",
                            "input": {"x": 5, "y": 3},
                        }
                    ],
                    "stop_reason": "tool_use",
                }
            case _:
                return FIXED_MESSAGE

    @solver
    def proxy_solver() -> Any:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            del generate
            async with sandbox_model_proxy(
                sandbox(),
                methods={"generate_anthropic": generate_anthropic},
                port=PORT,
                bridged_tools=[
                    BridgedToolsSpec(name="calc", tools=[calculator_add(call_log)])
                ],
            ) as model_proxy:
                checks["mcp_urls"] = [
                    config.url for config in model_proxy.mcp_server_configs
                ]

                # a non-streaming request returns the handler's dict unchanged
                checks["fixed"] = await _curl(
                    "/v1/messages", _anthropic_request("fixed")
                )

                # a streaming request gets server-sent events built by the proxy
                checks["stream"] = await _curl(
                    "/v1/messages", _anthropic_request("fixed", stream=True)
                )

                # a raising handler yields a provider-shaped error; the proxy stays up
                checks["overloaded"] = await _curl(
                    "/v1/messages", _anthropic_request("overloaded")
                )
                checks["bug"] = await _curl("/v1/messages", _anthropic_request("bug"))
                checks["missing"] = await _curl(
                    "/v1/chat/completions",
                    {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                )
                checks["after_errors"] = await _curl(
                    "/v1/messages", _anthropic_request("fixed")
                )

                # bridged tools: listed, denied until proposed, run once per proposal
                checks["tools_list"] = await _mcp("tools/list")
                call: dict[str, JsonValue] = {
                    "name": "calculator_add",
                    "arguments": {"x": 5, "y": 3},
                }
                checks["call_unproposed"] = await _mcp("tools/call", call)
                checks["propose"] = await _curl(
                    "/v1/messages",
                    _anthropic_request(
                        "tool",
                        tools=[
                            {
                                "name": "mcp__calc__calculator_add",
                                "description": ADD_DESCRIPTION,
                                "input_schema": _ADD_SCHEMA,
                            }
                        ],
                    ),
                )
                checks["call_proposed"] = await _mcp("tools/call", call)
                checks["call_again"] = await _mcp("tools/call", call)
            return state

        return solve

    log = inspect_eval(proxy_task(proxy_solver()), model=get_model("mockllm/model"))[0]
    assert log.status == "success", log.error

    _assert_proxy_checks(checks, handled, call_log)
