"""Contracts for the per-trial relay lifecycle and its failure accounting."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import datetime
import json
import types
from collections.abc import AsyncGenerator, Callable, Collection, Generator, Mapping
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import inspect_ai.event
import inspect_ai.model
import inspect_ai.tool
import inspect_ai.util
import pydantic
import pytest
from harbor.environments.base import BaseEnvironment

from harbor_sandbox_agent_proxy import model_events, trial_relay
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.connection import AgentConnection, served_connection_env
from harbor_sandbox_agent_proxy.outcome import EventSink, RelayOutcome, TrialIdentity
from harbor_sandbox_agent_proxy.relay import RelayCall, RelayPhase, relay_methods
from harbor_sandbox_agent_proxy.routing import (
    ModelCall,
    RequestDialect,
    Route,
    auth_headers,
    match_declared,
)
from harbor_sandbox_agent_proxy.task_metadata import BridgedMcpServer
from harbor_sandbox_agent_proxy.trial_relay import TrialRelay


class _Sink(EventSink):
    def __init__(self) -> None:
        self.events: list[inspect_ai.event.Event] = []
        self.outcome: RelayOutcome | None = None

    async def record(self, event: inspect_ai.event.Event) -> None:
        self.events.append(event)

    async def close(self, outcome: RelayOutcome) -> None:
        self.outcome = outcome


_CONNECTION = AgentConnection(
    "anthropic",
    ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    ("ANTHROPIC_BASE_URL",),
)


def _protected() -> frozenset[str]:
    return frozenset({"host-key"})


async def _route(call: ModelCall) -> Route:
    return Route(
        "https://upstream.test",
        auth_headers(call.api, "host-key"),
        match_declared(call),
    )


def _settings(**overrides: object) -> RelayConfig:
    return RelayConfig.model_validate(
        {
            "declared_models": ["anthropic/claude-test"],
            "max_request_bytes": 1024,
            "max_calls": 10,
            **overrides,
        }
    )


def _relay(
    *,
    settings: RelayConfig | None = None,
    environment: BaseEnvironment | None = None,
    protected: Callable[[], Collection[str]] = _protected,
) -> TrialRelay:
    return TrialRelay(
        trial=TrialIdentity(name="example__1", directory=Path("/tmp/example__1")),
        config=settings or _settings(),
        route=_route,
        sink=_Sink(),
        connection=_CONNECTION,
        protected=protected,
        environment=environment,
        bridged_servers=(
            BridgedMcpServer(
                name="task-tools",
                command="python3",
                args=["/opt/task_tools.py"],
                exclude_tools=["admin"],
            ),
        ),
    )


def _call(*, phase: RelayPhase = "agent") -> RelayCall:
    now = datetime.datetime.now(datetime.UTC)
    return RelayCall(
        phase=phase,
        ts_request=now,
        ts_response=now,
        method="generate_anthropic",
        provider="anthropic",
        dialect=RequestDialect.MESSAGES,
        model="claude-test",
        status=200,
        latency_ms=1.0,
        request={"model": "claude-test", "messages": []},
        response={
            "id": "message-id",
            "model": "claude-test",
            "content": [],
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        headers={},
    )


def _tool_names(body: object) -> list[str]:
    if not isinstance(body, dict):
        return []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [
        name
        for tool in tools
        if isinstance(tool, dict) and isinstance(name := tool.get("name"), str)
    ]


def test_mcp_server_configs_are_known_before_the_proxy_starts() -> None:
    configured = _relay().mcp_server_configs()

    assert [(server.name, server.transport, server.url) for server in configured] == [
        ("task-tools", "streamable-http", "http://localhost:13131/mcp/task-tools")
    ]


async def test_the_jobs_polling_interval_reaches_the_proxy_and_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters: list[dict[str, object]] = []
    proxies: list[dict[str, object]] = []

    def adapter(*args: object, **kwargs: object) -> object:
        del args
        adapters.append(kwargs)
        return object()

    @contextlib.contextmanager
    def bind(environment: object) -> Generator[object]:
        yield environment

    @contextlib.asynccontextmanager
    async def proxy(sandbox: object, **kwargs: object) -> AsyncGenerator[object]:
        del sandbox
        proxies.append(kwargs)
        yield types.SimpleNamespace(port=13131)

    async def answered(
        sandbox: inspect_ai.util.SandboxEnvironment, *, port: int
    ) -> None:
        del sandbox, port

    monkeypatch.setattr(trial_relay, "HarborSandboxEnvironment", adapter)
    monkeypatch.setattr(trial_relay, "bind_sandbox_environment", bind)
    monkeypatch.setattr(trial_relay, "sandbox_model_proxy", proxy)
    monkeypatch.setattr(trial_relay, "_wait_for_proxy", answered)
    relay_instance = _relay(
        settings=_settings(polling_interval=0.25),
        environment=MagicMock(spec=BaseEnvironment),
    )
    relay_instance.bridged_servers = ()

    await relay_instance.open_agent_phase()
    _ = await relay_instance.close()

    assert [call["polling_interval"] for call in adapters] == [0.25]
    assert [call["polling_interval"] for call in proxies] == [0.25]


async def test_proxy_readiness_does_not_require_curl_in_the_sandbox() -> None:
    sandbox = MagicMock(spec=inspect_ai.util.SandboxEnvironment)
    sandbox.exec = AsyncMock(
        return_value=types.SimpleNamespace(success=True, stdout="200", stderr="")
    )

    await trial_relay._wait_for_proxy(sandbox, port=13131)

    (command,), kwargs = sandbox.exec.await_args
    assert kwargs == {"user": "root"}
    assert command[:2] == ["python3", "-c"]
    assert "curl" not in command[-1]
    assert "HTTPConnection" in command[-1]


async def test_proxy_readiness_accepts_a_live_route_that_returns_not_found() -> None:
    sandbox = MagicMock(spec=inspect_ai.util.SandboxEnvironment)
    sandbox.exec = AsyncMock(
        return_value=types.SimpleNamespace(success=True, stdout="404", stderr="")
    )

    await trial_relay._wait_for_proxy(sandbox, port=13131)

    (command,), _ = sandbox.exec.await_args
    assert "HTTPConnection" in command[-1]


async def test_the_relay_registers_both_caller_keys_for_redaction() -> None:
    relay_instance = _relay()
    agent_env = _CONNECTION.proxy_env(key=relay_instance.agent_key, port=13131)

    assert relay_instance.caller_keys == frozenset(
        {relay_instance.agent_key, relay_instance.verifier_key}
    )
    assert relay_instance.protected_values() == frozenset(
        {*relay_instance.caller_keys, "host-key"}
    )
    assert relay_instance.phase_env() == agent_env

    await relay_instance.end_agent_phase()
    first_verifier_key = relay_instance.verifier_key
    await relay_instance.resume_agent_phase()
    await relay_instance.end_agent_phase()

    expected_verifier_env = {
        **served_connection_env(key=relay_instance.verifier_key, port=13131),
        **_CONNECTION.proxy_env(key=relay_instance.verifier_key, port=13131),
    }
    assert first_verifier_key in relay_instance.caller_keys
    assert relay_instance.verifier_key in relay_instance.caller_keys
    assert relay_instance.phase_env() == expected_verifier_env
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in relay_instance.phase_env()
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in relay_instance.phase_env()


async def test_close_does_not_swallow_its_own_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay_instance = _relay(environment=MagicMock(spec=BaseEnvironment))

    async def hold_past_stop(
        *, ready: asyncio.Future[None], stop: asyncio.Event
    ) -> None:
        del stop
        ready.set_result(None)
        _ = await asyncio.Event().wait()

    monkeypatch.setattr(relay_instance, "_hold_proxy", hold_past_stop)
    await relay_instance.open_agent_phase()
    closing = asyncio.create_task(relay_instance.close())
    await asyncio.sleep(0)
    _ = closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await closing


async def test_close_returns_the_outcome_whatever_the_proxy_task_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay_instance = _relay(environment=MagicMock(spec=BaseEnvironment))
    assert isinstance(relay_instance.sink, _Sink)

    async def hold_then_lose_the_sandbox(
        *, ready: asyncio.Future[None], stop: asyncio.Event
    ) -> None:
        ready.set_result(None)
        _ = await stop.wait()
        raise ConnectionError("the sandbox pod is gone")

    monkeypatch.setattr(relay_instance, "_hold_proxy", hold_then_lose_the_sandbox)
    await relay_instance.open_agent_phase()
    first = _model_event([_USER], "first reply")
    await _record_events(monkeypatch, relay_instance, first)

    outcome = await relay_instance.close()

    assert outcome.events == [first]
    assert relay_instance.sink.outcome is None
    assert outcome.answered_agent_calls == 1
    assert outcome.proxy_exit is not None
    assert outcome.proxy_exit["error"] == "ConnectionError('the sandbox pod is gone')"
    assert outcome.mode == "relay"


async def test_each_swap_to_the_verifier_mints_a_fresh_key_and_retires_the_last() -> (
    None
):
    relay_instance = _relay()
    await relay_instance.end_agent_phase()
    first = relay_instance.verifier_key
    await relay_instance.resume_agent_phase()
    await relay_instance.end_agent_phase()

    second = relay_instance.verifier_key
    assert second != first
    assert {relay_instance.agent_key, first, second} <= relay_instance.caller_keys
    assert {first, second} <= relay_instance.protected_values()
    assert relay_instance.phase_env()["ANTHROPIC_API_KEY"] == second
    calls: list[RelayCall] = []

    async def record(call: RelayCall) -> None:
        calls.append(call)

    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        generate = relay_methods(
            relay_instance.current_policy,
            relay_instance.route,
            record=record,
            client=client,
        )["generate_anthropic"]
        for key in (first, second):
            _ = await generate(
                {"model": "claude-test", "messages": []}, {"x-api-key": key}
            )

    assert [(call.phase, call.authenticated) for call in calls] == [
        ("verifier", False),
        ("verifier", True),
    ]


_JUDGE_REQUEST: dict[str, pydantic.JsonValue] = {
    "model": "claude-test",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "Grade the transcript."}],
    "tools": [
        {"name": "StructuredOutput", "input_schema": {"type": "object"}},
        {"name": "Read", "input_schema": {"type": "object"}},
        {"name": "mcp__task-tools__ping", "input_schema": {"type": "object"}},
        {"type": "web_search_20250305", "name": "web_search"},
    ],
    "mcp_servers": [{"type": "url", "url": "https://mcp.example", "name": "m"}],
}


@pytest.mark.parametrize(
    ("settings", "agent_tools", "verifier_tools"),
    [
        pytest.param(
            {"tools": "bridged-only", "verifier_tools": "any"},
            [],
            ["StructuredOutput", "Read", "mcp__task-tools__ping"],
            id="bridged-only-agent-any-verifier",
        ),
        pytest.param(
            {"tools": "bridged-only"}, [], [], id="the-verifier-inherits-the-agents"
        ),
        pytest.param(
            {
                "tools": "any",
                "verifier_tools": "bridged-only",
                "permitted_tool_names": ["Read"],
            },
            ["StructuredOutput", "Read", "mcp__task-tools__ping"],
            ["Read"],
            id="bridged-only-verifier-keeps-only-permitted-names",
        ),
    ],
)
async def test_each_phase_confines_tools_by_its_own_rule(
    settings: dict[str, object], agent_tools: list[str], verifier_tools: list[str]
) -> None:
    relay_instance = _relay(settings=_settings(**settings))
    sent: list[dict[str, object]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"content": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        generate = relay_methods(
            relay_instance.current_policy,
            relay_instance.route,
            record=relay_instance.record,
            client=client,
        )["generate_anthropic"]
        _ = await generate(
            copy.deepcopy(_JUDGE_REQUEST), {"x-api-key": relay_instance.agent_key}
        )
        await relay_instance.end_agent_phase()
        _ = await generate(
            copy.deepcopy(_JUDGE_REQUEST), {"x-api-key": relay_instance.verifier_key}
        )

    assert [_tool_names(body) for body in sent] == [agent_tools, verifier_tools]
    assert ["mcp_servers" in body for body in sent] == [False, False]


async def test_end_agent_phase_swaps_to_verifier_credentials_and_refuses_an_empty_conversation() -> (
    None
):
    relay_instance = _relay(
        settings=_settings(
            allowed_betas=["agent-beta"], verifier_allowed_betas=["judge-beta"]
        )
    )

    await relay_instance.end_agent_phase()

    assert relay_instance.phase == "verifier"
    assert relay_instance.verifier_conversation == {
        "refused": "the agent made no answered model call through the relay"
    }
    assert (
        relay_instance.refusal_reason
        == "the agent made no answered model call through the relay"
    )
    assert (
        relay_instance.phase_env()["ANTHROPIC_API_KEY"] == relay_instance.verifier_key
    )


async def test_record_failure_is_re_raised_and_preserved_in_the_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay_instance = _relay()

    async def fail_conversion(_call: RelayCall) -> object:
        raise ValueError("cannot convert this call")

    monkeypatch.setattr(model_events, "model_event_from_call", fail_conversion)
    with pytest.raises(ValueError, match="cannot convert this call"):
        await relay_instance.record(_call())

    outcome = await relay_instance.close()
    assert outcome.record_error == "ValueError('cannot convert this call')"
    assert len(outcome.calls) == 1
    assert outcome.counts["agent_calls"] == 1


async def test_record_failure_keeps_only_the_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay_instance = _relay()
    failures = iter((ValueError("first"), RuntimeError("second")))

    async def fail_conversion(_call: RelayCall) -> object:
        raise next(failures)

    monkeypatch.setattr(model_events, "model_event_from_call", fail_conversion)
    with pytest.raises(ValueError, match="first"):
        await relay_instance.record(_call())
    with pytest.raises(RuntimeError, match="second"):
        await relay_instance.record(_call())

    assert (await relay_instance.close()).record_error == "ValueError('first')"


def _model_event(
    messages: list[inspect_ai.model.ChatMessage],
    reply: str | inspect_ai.model.ChatMessageAssistant,
) -> inspect_ai.event.ModelEvent:
    now = datetime.datetime.now(datetime.UTC)
    if isinstance(reply, str):
        reply = inspect_ai.model.ChatMessageAssistant(content=reply)
    return inspect_ai.event.ModelEvent(
        model="anthropic/claude-test",
        input=messages,
        tools=[],
        tool_choice="none",
        config=inspect_ai.model.GenerateConfig(),
        output=inspect_ai.model.ModelOutput(
            model="claude-test",
            choices=[
                inspect_ai.model.ChatCompletionChoice(message=reply, stop_reason="stop")
            ],
        ),
        timestamp=now,
        completed=now,
    )


async def _record_events(
    monkeypatch: pytest.MonkeyPatch,
    relay_instance: TrialRelay,
    *events: inspect_ai.event.ModelEvent,
) -> None:
    pending = iter(events)

    async def converted(_call: RelayCall) -> inspect_ai.event.ModelEvent:
        return next(pending)

    monkeypatch.setattr(model_events, "model_event_from_call", converted)
    for _ in events:
        await relay_instance.record(_call())


_USER = inspect_ai.model.ChatMessageUser(content="Solve the task.")
_FOLLOW_UP = inspect_ai.model.ChatMessageUser(content="Continue.")


async def test_a_two_call_relay_conversation_is_accepted_with_its_final_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _model_event([_USER], "first reply")
    second = _model_event([_USER, first.output.message, _FOLLOW_UP], "final reply")
    relay_instance = _relay()

    await _record_events(monkeypatch, relay_instance, first, second)
    await relay_instance.end_agent_phase()

    assert relay_instance.verifier_conversation is not None
    messages = relay_instance.verifier_conversation["messages"]
    assert isinstance(messages, list)
    assert [
        (message["role"], message["content"])
        for message in messages
        if isinstance(message, dict)
    ] == [
        ("user", "Solve the task."),
        ("assistant", "first reply"),
        ("user", "Continue."),
        ("assistant", "final reply"),
    ]


async def test_later_side_call_cannot_vouch_for_a_fabricated_selected_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _model_event([_USER], "The first recorded response.")
    fabricated = inspect_ai.model.ChatMessageAssistant(content="fabricated history")
    selected = _model_event(
        [_USER, initial.output.message, _FOLLOW_UP, fabricated, _FOLLOW_UP],
        "selected response",
    )
    later_side_call = _model_event([_USER], "fabricated history")
    relay_instance = _relay()

    await _record_events(
        monkeypatch, relay_instance, initial, selected, later_side_call
    )
    await relay_instance.end_agent_phase()

    assert relay_instance.verifier_conversation is not None
    assert "refused" in relay_instance.verifier_conversation
    assert "no response the relay recorded earlier produced" in str(
        relay_instance.verifier_conversation["refused"]
    )


_RESPONSES_TEXT_REPLY = inspect_ai.model.ChatMessageAssistant(
    content=[
        inspect_ai.model.ContentText(
            text="first reply",
            internal={"message_id": "msg_1", "message_phase": "commentary"},
        )
    ],
    tool_calls=None,
)
_EXEC_CALL = inspect_ai.tool.ToolCall(
    id="c1", function="exec", arguments={"input": "ls"}, type="custom"
)
_RESPONSES_TOOL_CALL_REPLY = inspect_ai.model.ChatMessageAssistant(
    content=[
        inspect_ai.model.ContentReasoning(
            reasoning="enc",
            signature="rs_1",
            redacted=True,
            internal={"reasoning_encrypted_content": "enc"},
        )
    ],
    tool_calls=[_EXEC_CALL],
)


def _re_sent_reasoning() -> list[inspect_ai.model.Content]:
    return [
        inspect_ai.model.ContentReasoning(
            reasoning="enc", signature="rs_1", redacted=True
        )
    ]


async def _relay_after_re_sending(
    monkeypatch: pytest.MonkeyPatch,
    recorded: inspect_ai.model.ChatMessageAssistant,
    re_sent: inspect_ai.model.ChatMessageAssistant,
) -> TrialRelay:
    first = _model_event([_USER], recorded)
    second = _model_event([_USER, re_sent, _FOLLOW_UP], "final reply")
    relay_instance = _relay()
    await _record_events(monkeypatch, relay_instance, first, second)
    await relay_instance.end_agent_phase()
    return relay_instance


async def _conversation_after_re_sending(
    monkeypatch: pytest.MonkeyPatch,
    recorded: inspect_ai.model.ChatMessageAssistant,
    re_sent: inspect_ai.model.ChatMessageAssistant,
) -> dict[str, pydantic.JsonValue]:
    relay_instance = await _relay_after_re_sending(monkeypatch, recorded, re_sent)
    assert relay_instance.verifier_conversation is not None
    return relay_instance.verifier_conversation


@pytest.mark.parametrize(
    ("recorded", "re_sent"),
    [
        pytest.param(
            _RESPONSES_TEXT_REPLY,
            inspect_ai.model.ChatMessageAssistant(
                content=[inspect_ai.model.ContentText(text="first reply")],
                tool_calls=[],
            ),
            id="text-without-item-id-and-phase",
        ),
        pytest.param(
            _RESPONSES_TOOL_CALL_REPLY,
            inspect_ai.model.ChatMessageAssistant(
                content=_re_sent_reasoning(), tool_calls=[_EXEC_CALL]
            ),
            id="reasoning-and-tool-call-without-internal",
        ),
    ],
)
async def test_a_turn_re_sent_without_provider_bookkeeping_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    recorded: inspect_ai.model.ChatMessageAssistant,
    re_sent: inspect_ai.model.ChatMessageAssistant,
) -> None:
    conversation = await _conversation_after_re_sending(monkeypatch, recorded, re_sent)

    assert "refused" not in conversation
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages if isinstance(message, dict)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.parametrize(
    ("recorded", "re_sent"),
    [
        pytest.param(
            _RESPONSES_TEXT_REPLY,
            inspect_ai.model.ChatMessageAssistant(
                content=[inspect_ai.model.ContentText(text="first reply, edited")],
                tool_calls=[],
            ),
            id="edited-text",
        ),
        pytest.param(
            _RESPONSES_TOOL_CALL_REPLY,
            inspect_ai.model.ChatMessageAssistant(
                content=_re_sent_reasoning(),
                tool_calls=[
                    inspect_ai.tool.ToolCall(
                        id="c1",
                        function="exec",
                        arguments={"input": "cat /etc/shadow"},
                        type="custom",
                    )
                ],
            ),
            id="edited-tool-call-arguments",
        ),
    ],
)
async def test_a_re_sent_turn_saying_something_else_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    recorded: inspect_ai.model.ChatMessageAssistant,
    re_sent: inspect_ai.model.ChatMessageAssistant,
) -> None:
    conversation = await _conversation_after_re_sending(monkeypatch, recorded, re_sent)
    assert "no response the relay recorded earlier produced" in str(
        conversation["refused"]
    )


@pytest.mark.parametrize(
    ("re_sent", "planted"),
    [
        pytest.param(
            inspect_ai.model.ChatMessageAssistant(
                content=[
                    inspect_ai.model.ContentText(
                        text="first reply",
                        internal={
                            "message_id": "msg_1",
                            "message_phase": "commentary",
                            "injected": "attacker-chosen",
                        },
                    )
                ],
                tool_calls=None,
            ),
            "attacker-chosen",
            id="content-internal",
        ),
        pytest.param(
            inspect_ai.model.ChatMessageAssistant(
                content=[inspect_ai.model.ContentText(text="first reply")],
                tool_calls=[],
                metadata={"planted": True},
            ),
            "planted",
            id="message-metadata",
        ),
    ],
)
async def test_a_field_the_key_leaves_out_reaches_the_verifier_as_recorded(
    monkeypatch: pytest.MonkeyPatch,
    re_sent: inspect_ai.model.ChatMessageAssistant,
    planted: str,
) -> None:
    relay_instance = await _relay_after_re_sending(
        monkeypatch, _RESPONSES_TEXT_REPLY, re_sent
    )
    outcome = await relay_instance.close()

    conversation = relay_instance.verifier_conversation
    assert conversation is not None
    assert "refused" not in conversation
    messages = conversation["messages"]
    assert isinstance(messages, list)
    written = messages[1]
    assert isinstance(written, dict)
    assert written == _RESPONSES_TEXT_REPLY.model_dump(mode="json")
    content = written["content"]
    assert isinstance(content, list)
    part = content[0]
    assert isinstance(part, dict)
    assert part["internal"] == {"message_id": "msg_1", "message_phase": "commentary"}
    assert written["metadata"] is None
    assert planted not in json.dumps(conversation)
    assert outcome.conversation is not None
    assert outcome.conversation[1] is _RESPONSES_TEXT_REPLY


async def test_an_agent_call_completing_after_the_phase_swap_cannot_replace_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _model_event([_USER], "first reply")
    second = _model_event([_USER, first.output.message, _FOLLOW_UP], "final reply")
    late = _model_event(
        [
            _USER,
            first.output.message,
            _FOLLOW_UP,
            second.output.message,
            _FOLLOW_UP,
            inspect_ai.model.ChatMessageAssistant(content="fabricated after the check"),
            _FOLLOW_UP,
        ],
        "late reply",
    )
    relay_instance = _relay()
    await _record_events(monkeypatch, relay_instance, first, second)
    await relay_instance.end_agent_phase()
    checked = relay_instance.verifier_conversation
    await _record_events(monkeypatch, relay_instance, late)
    outcome = await relay_instance.close()

    assert relay_instance.verifier_conversation == checked
    assert outcome.conversation is not None
    assert [message.text for message in outcome.conversation] == [
        "Solve the task.",
        "first reply",
        "Continue.",
        "final reply",
    ]


async def test_a_later_step_reenters_the_agent_phase_and_is_checked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _model_event([_USER], "first reply")
    second = _model_event([_USER, first.output.message, _FOLLOW_UP], "second reply")
    relay_instance = _relay()
    await _record_events(monkeypatch, relay_instance, first)
    await relay_instance.end_agent_phase()
    assert relay_instance.phase == "verifier"
    await relay_instance.resume_agent_phase()
    assert relay_instance.phase == "agent"
    assert relay_instance.verifier_conversation is None
    assert relay_instance.phase_env()["ANTHROPIC_API_KEY"] == relay_instance.agent_key
    await _record_events(monkeypatch, relay_instance, second)
    await relay_instance.end_agent_phase()
    outcome = await relay_instance.close()
    assert outcome.conversation is not None
    assert [message.text for message in outcome.conversation] == [
        "Solve the task.",
        "first reply",
        "Continue.",
        "second reply",
    ]
    assert outcome.answered_agent_calls == 2


async def test_configured_policy_reaches_the_forwarder(tmp_path: Path) -> None:
    relay = TrialRelay(
        trial=TrialIdentity(name="trial", directory=tmp_path),
        config=RelayConfig(
            declared_models=("anthropic/claude-test",),
            tools="bridged-only",
            permitted_tool_names=("kept",),
            permitted_tool_prefixes=("trusted_",),
            allowed_betas=("allowed",),
            max_request_bytes=1_024,
            max_calls=1,
            max_recorded_bytes=4_096,
        ),
        route=_route,
        sink=_Sink(),
        connection=_CONNECTION,
    )
    observed_bodies: list[dict[str, object]] = []
    observed_headers: list[Mapping[str, str]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed_bodies.append(json.loads(request.content))
        observed_headers.append(request.headers)
        return httpx.Response(200, json={"content": [], "role": "assistant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        generate = relay_methods(
            relay.current_policy, relay.route, record=relay.record, client=client
        )["generate_anthropic"]
        _ = await generate(
            {
                "model": "claude-test",
                "messages": [],
                "tools": [
                    {"name": "kept", "input_schema": {"type": "object"}},
                    {"name": "trusted_prefix", "input_schema": {"type": "object"}},
                    {"name": "stripped", "input_schema": {"type": "object"}},
                ],
            },
            {"x-api-key": relay.agent_key, "anthropic-beta": "allowed, stripped"},
        )
        _ = await generate(
            {"model": "claude-test", "messages": []}, {"x-api-key": relay.agent_key}
        )

    assert _tool_names(observed_bodies[0]) == ["kept", "trusted_prefix"]
    assert observed_headers[0]["anthropic-beta"] == "allowed"
    assert len(observed_bodies) == 1
    assert len(relay.calls) == 1

    async def forwards(config: RelayConfig) -> int:
        bounded = TrialRelay(
            trial=TrialIdentity(name="bounded", directory=tmp_path),
            config=config,
            route=_route,
            sink=_Sink(),
            connection=_CONNECTION,
        )
        sent: list[httpx.Request] = []

        def count(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"content": [], "role": "assistant"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(count)) as client:
            generate = relay_methods(
                bounded.current_policy,
                bounded.route,
                record=bounded.record,
                client=client,
            )["generate_anthropic"]
            _ = await generate(
                {"model": "claude-test", "messages": []},
                {"x-api-key": bounded.agent_key},
            )
            _ = await generate(
                {"model": "claude-test", "messages": []},
                {"x-api-key": bounded.agent_key},
            )
        return len(sent)

    assert (
        await forwards(
            RelayConfig(
                declared_models=("anthropic/claude-test",),
                max_request_bytes=1,
                max_calls=10,
                max_recorded_bytes=4_096,
            )
        )
        == 0
    )
    assert (
        await forwards(
            RelayConfig(
                declared_models=("anthropic/claude-test",),
                max_request_bytes=1_024,
                max_calls=10,
                max_recorded_bytes=1,
            )
        )
        == 1
    )


async def test_usage_mode_still_passes_every_model_event_to_the_sink_live(
    tmp_path: Path,
) -> None:
    relay_instance = _usage_relay(tmp_path)
    sink = relay_instance.sink
    assert isinstance(sink, _Sink)
    for expected in range(1, 4):
        await relay_instance.record(_call())
        assert len(sink.events) == expected
    assert all(isinstance(e, inspect_ai.event.ModelEvent) for e in sink.events)
    assert all(line["trial"] == "usage" for line in _usage_lines(tmp_path))
    assert relay_instance.events == []


async def test_usage_mode_never_writes_a_caller_key_sent_as_the_model(
    tmp_path: Path,
) -> None:
    relay_instance = _usage_relay(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    ) as client:
        generate = relay_methods(
            relay_instance.current_policy,
            relay_instance.route,
            record=relay_instance.record,
            client=client,
        )["generate_anthropic"]
        _ = await generate(
            {"model": relay_instance.agent_key, "messages": []},
            {"x-api-key": relay_instance.agent_key},
        )
    text = (tmp_path / "sandbox-agent-proxy" / "usage.jsonl").read_text()
    assert relay_instance.agent_key not in text
    (line,) = _usage_lines(tmp_path)
    assert line["model"] is None
    assert line["refused"] == "model_not_declared"


@inspect_ai.tool.tool
def _echo() -> inspect_ai.tool.Tool:
    async def execute(text: str) -> str:
        """Echo the text.

        Args:
            text: The text.
        """
        return text

    return execute


async def test_usage_mode_keeps_no_refused_tool_call_arguments(tmp_path: Path) -> None:
    relay_instance = _usage_relay(tmp_path)
    guarded = trial_relay.phase_guarded_tool(
        _echo(),
        server="task-tools",
        phase=lambda: "verifier",
        refused=relay_instance._record_refused_tool_call,
    )
    for _ in range(3):
        with pytest.raises(inspect_ai.tool.ToolError):
            await guarded(text="private-request-text")
    assert relay_instance.events == []
    sink = relay_instance.sink
    assert isinstance(sink, _Sink)
    assert len(sink.events) == 3
    text = (tmp_path / "sandbox-agent-proxy" / "usage.jsonl").read_text()
    assert "private-request-text" not in text
    lines = _usage_lines(tmp_path)
    assert [line["refused"] for line in lines] == ["tool_outside_agent_phase"] * 3
    assert all(set(line) == _TOOL_ROW_KEYS for line in lines)
    assert lines[0]["call_id_source"] == "generated"
    assert lines[0]["refused"] in trial_relay.USAGE_REFUSAL_CODES


# The usage.jsonl schema; the task harness's recorder (HSAP_RECORD=usage) pins the same
# two key sets and refusal codes.
_MODEL_ROW_KEYS = {
    "call_id",
    "call_id_source",
    "dialect",
    "trial",
    "phase",
    "ts_request",
    "ts_response",
    "model",
    "status",
    "refused",
    "input_tokens",
    "output_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "request_bytes",
    "response_bytes",
    "recorded_bytes",
}
_TOOL_ROW_KEYS = {
    "call_id",
    "call_id_source",
    "trial",
    "phase",
    "kind",
    "tool",
    "refused",
}


def test_usage_refusal_codes_are_the_shared_set() -> None:
    assert trial_relay.USAGE_REFUSAL_CODES == {
        "unauthenticated",
        "model_not_declared",
        "call_cap",
        "recorded_bytes_cap",
        "request_too_large",
        "request_rejected",
        "tool_outside_agent_phase",
        "token_budget", "token_budget_insufficient", "token_budget_wait", "budget_closed",
        "usage_unknown", "unsupported_input", "unsupported_model_profile",
        "output_limit_required", "invalid_request_value", "invalid_reservation",
        "reservation_too_large", "unbounded_thinking", "bound_violated",
    }


async def test_usage_model_rows_have_the_shared_keys(tmp_path: Path) -> None:
    relay_instance = _usage_relay(tmp_path)
    await relay_instance.record(_call())
    assert await _forward(relay_instance, 1) == 1
    assert all(set(line) == _MODEL_ROW_KEYS for line in _usage_lines(tmp_path))


async def test_usage_mode_uses_the_hosts_call_id_and_a_closed_dialect(
    tmp_path: Path,
) -> None:
    relay_instance = _usage_relay(tmp_path)
    await relay_instance.record(
        _call().model_copy(update={"headers": {"x-bet-eval-call-id": "host-42"}})
    )
    await relay_instance.record(_call())
    first, second = _usage_lines(tmp_path)
    assert (first["call_id"], first["call_id_source"]) == ("host-42", "header")
    assert second["call_id_source"] == "generated"
    assert first["dialect"] in trial_relay.USAGE_DIALECTS


def _gemini_call(usage_metadata: object) -> RelayCall:
    now = datetime.datetime.now(datetime.UTC)
    response: dict[str, object] = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "modelVersion": "gemini-test",
    }
    if usage_metadata is not None:
        response["usageMetadata"] = usage_metadata
    return RelayCall(
        phase="agent",
        ts_request=now,
        ts_response=now,
        method="generate_google",
        provider="google",
        dialect=RequestDialect.GEMINI,
        model="gemini-test",
        status=200,
        latency_ms=1.0,
        request={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        response=response,
        headers={},
    )


# MACH's review cases (gs://theorem-build-io/in/bet/usage-review/20261104/
# test_gemini_usage_presence.py): Inspect's Google converter turns an absent
# promptTokenCount or candidatesTokenCount into 0; usage.jsonl must write null instead.
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            {"promptTokenCount": 41, "candidatesTokenCount": 6}, (41, 6), id="complete"
        ),
        pytest.param(
            {"promptTokenCount": 0, "candidatesTokenCount": 0}, (0, 0), id="real_zeroes"
        ),
        pytest.param({"candidatesTokenCount": 6}, (None, 6), id="missing_input"),
        pytest.param({"promptTokenCount": 41}, (41, None), id="missing_output"),
        pytest.param({}, (None, None), id="empty_metadata"),
        pytest.param(None, (None, None), id="absent_metadata"),
        *(
            pytest.param(
                {**{"promptTokenCount": 41, "candidatesTokenCount": 6}, field: bad},
                (None, 6) if field == "promptTokenCount" else (41, None),
                id=f"{field}_{name}",
            )
            for field in ("promptTokenCount", "candidatesTokenCount")
            for name, bad in (
                ("true", True),
                ("false", False),
                ("negative", -1),
                ("float", 1.0),
                ("string", "41"),
            )
        ),
    ],
)
async def test_usage_mode_writes_null_for_a_gemini_count_the_provider_did_not_report(
    tmp_path: Path, metadata: object, expected: tuple[int | None, int | None]
) -> None:
    relay_instance = _usage_relay(tmp_path, declared_models=["google/gemini-test"])
    await relay_instance.record(_gemini_call(metadata))
    (line,) = _usage_lines(tmp_path)
    assert (line["input_tokens"], line["output_tokens"]) == expected
    assert line["dialect"] == "gemini"
    assert set(line) == _MODEL_ROW_KEYS


def _error_call(response: object, *, status: int = 529) -> RelayCall:
    return _call().model_copy(update={"status": status, "response": response})


# The Sol review of proof-evals #64 (finding 1): a 529 whose body reports usage must keep
# its counts in usage mode, as full mode's reader keeps them from outcome.json.
async def test_usage_mode_keeps_the_counts_an_error_response_reports(
    tmp_path: Path,
) -> None:
    relay_instance = _usage_relay(tmp_path)
    ok = _call()
    ok = ok.model_copy(
        update={
            "response": {
                **ok.response,  # pyright: ignore[reportGeneralTypeIssues]
                "usage": {"input_tokens": 3, "output_tokens": 5},
            }
        }
    )
    await relay_instance.record(ok)
    await relay_instance.record(
        _error_call(
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
                "usage": {"input_tokens": 4, "output_tokens": 50},
            }
        )
    )
    lines = _usage_lines(tmp_path)
    assert [line["status"] for line in lines] == [200, 529]
    assert sum(line["output_tokens"] for line in lines) == 55
    assert (lines[1]["input_tokens"], lines[1]["output_tokens"]) == (4, 50)
    assert all(set(line) == _MODEL_ROW_KEYS for line in lines)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(
            {"type": "error", "error": {"type": "overloaded_error"}},
            (None, None),
            id="no_usage",
        ),
        pytest.param({"usage": {}}, (None, None), id="empty_usage"),
        pytest.param(
            {"usage": {"input_tokens": 4, "output_tokens": True}},
            (4, None),
            id="bool_output",
        ),
        pytest.param(
            {"usage": {"input_tokens": -1, "output_tokens": 50}},
            (None, 50),
            id="negative_input",
        ),
        pytest.param(
            {"usage": {"input_tokens": 0, "output_tokens": 0}},
            (0, 0),
            id="real_zeroes",
        ),
        pytest.param("upstream timed out", (None, None), id="not_an_object"),
    ],
)
async def test_usage_mode_reads_an_error_responses_counts_by_the_integer_rule(
    tmp_path: Path, response: object, expected: tuple[int | None, int | None]
) -> None:
    relay_instance = _usage_relay(tmp_path)
    await relay_instance.record(_error_call(response))
    (line,) = _usage_lines(tmp_path)
    assert (line["input_tokens"], line["output_tokens"]) == expected
    assert set(line) == _MODEL_ROW_KEYS


@pytest.mark.parametrize(
    ("method", "provider", "dialect", "response", "expected"),
    [
        pytest.param(
            "generate_completions",
            "openai",
            RequestDialect.CHAT_COMPLETIONS,
            {"usage": {"prompt_tokens": 30, "completion_tokens": 50}},
            (30, 50),
            id="chat-completions",
        ),
        pytest.param(
            "generate_google",
            "google",
            RequestDialect.GEMINI,
            {"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 50}},
            (30, 50),
            id="gemini",
        ),
        pytest.param(
            "generate_google",
            "google",
            RequestDialect.GEMINI,
            {"usageMetadata": {"promptTokenCount": 30}},
            (30, None),
            id="gemini_missing_output",
        ),
        pytest.param(
            "generate_responses",
            "openai",
            RequestDialect.RESPONSES,
            {"usage": {"input_tokens": 30, "output_tokens": 50}},
            (30, 50),
            id="responses",
        ),
    ],
)
async def test_usage_mode_reads_each_dialects_error_counts(
    tmp_path: Path,
    method: str,
    provider: str,
    dialect: RequestDialect,
    response: object,
    expected: tuple[int | None, int | None],
) -> None:
    relay_instance = _usage_relay(tmp_path)
    await relay_instance.record(
        _error_call(response).model_copy(
            update={"method": method, "provider": provider, "dialect": dialect}
        )
    )
    (line,) = _usage_lines(tmp_path)
    assert (line["input_tokens"], line["output_tokens"]) == expected


async def test_usage_mode_counts_answered_agent_calls_as_full_mode_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {}
    for mode in ("full", "usage"):
        relay_instance = TrialRelay(
            trial=TrialIdentity(name=mode, directory=tmp_path / mode),
            config=_settings(record=mode),
            route=_route,
            sink=_Sink(),
            connection=_CONNECTION,
        )
        await relay_instance.record(_call())
        counts[mode] = relay_instance._outcome().answered_agent_calls
    assert counts == {"full": 1, "usage": 1}


def _usage_relay(tmp_path: Path, **overrides: object) -> TrialRelay:
    return TrialRelay(
        trial=TrialIdentity(name="usage", directory=tmp_path),
        config=_settings(record="usage", **overrides),
        route=_route,
        sink=_Sink(),
        connection=_CONNECTION,
    )


async def _forward(relay_instance: TrialRelay, times: int) -> int:
    sent: list[httpx.Request] = []

    def count(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"content": [], "role": "assistant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(count)) as client:
        generate = relay_methods(
            relay_instance.current_policy,
            relay_instance.route,
            record=relay_instance.record,
            client=client,
        )["generate_anthropic"]
        for _ in range(times):
            _ = await generate(
                {"model": "claude-test", "messages": []},
                {"x-api-key": relay_instance.agent_key},
            )
    return len(sent)


def _usage_lines(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "sandbox-agent-proxy" / "usage.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_usage_mode_writes_one_line_per_call_and_holds_nothing(
    tmp_path: Path,
) -> None:
    relay_instance = _usage_relay(tmp_path)
    for _ in range(3):
        await relay_instance.record(_call())
    await relay_instance.record(_call(phase="verifier"))

    lines = _usage_lines(tmp_path)
    assert [line["phase"] for line in lines] == ["agent"] * 3 + ["verifier"]
    assert lines[0]["model"] == "claude-test"
    assert lines[0]["input_tokens"] == 1
    assert lines[0]["output_tokens"] == 1
    assert {"cache_write_tokens", "cache_read_tokens", "recorded_bytes"} <= set(
        lines[0]
    )
    assert len({line["call_id"] for line in lines}) == 4
    assert relay_instance.calls == []
    assert relay_instance.events == []
    assert relay_instance._answered_agent_events == []
    outcome = relay_instance._outcome()
    assert outcome.calls == [] and outcome.events == []
    assert outcome.counts["agent_calls"] == 3
    assert outcome.counts["verifier_calls"] == 1


async def test_usage_mode_refuses_the_verifier_conversation(tmp_path: Path) -> None:
    relay_instance = _usage_relay(tmp_path)
    await relay_instance.record(_call())
    await relay_instance.end_agent_phase()
    assert relay_instance.refusal_reason is not None
    assert relay_instance.verifier_conversation == {
        "refused": relay_instance.refusal_reason
    }


async def test_usage_mode_keeps_the_call_cap(tmp_path: Path) -> None:
    relay_instance = _usage_relay(tmp_path, max_calls=2)
    assert await _forward(relay_instance, 4) == 2
    assert relay_instance.calls == []
    assert len(_usage_lines(tmp_path)) >= 2


async def test_usage_mode_keeps_the_recorded_bytes_cap(tmp_path: Path) -> None:
    relay_instance = _usage_relay(tmp_path, max_recorded_bytes=1)
    assert await _forward(relay_instance, 2) == 1


async def test_full_mode_is_the_default_and_writes_no_usage_file(
    tmp_path: Path,
) -> None:
    assert RelayConfig(declared_models=("anthropic/claude-test",)).record == "full"
    relay_instance = TrialRelay(
        trial=TrialIdentity(name="full", directory=tmp_path),
        config=_settings(),
        route=_route,
        sink=_Sink(),
        connection=_CONNECTION,
    )
    await relay_instance.record(_call())
    assert len(relay_instance.calls) == 1
    assert len(relay_instance.events) == 1
    assert not (tmp_path / "sandbox-agent-proxy" / "usage.jsonl").exists()
