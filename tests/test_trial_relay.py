"""Contracts for the per-trial relay lifecycle and its failure accounting."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import datetime
import json
import types
from collections.abc import AsyncGenerator, Generator, Mapping
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

from harbor_sandbox_agent_proxy import trial_relay
from harbor_sandbox_agent_proxy.collaborators import (
    BridgedMcpServer,
    RequestDialect,
    UpstreamTarget,
)
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.outcome import (
    EventSink,
    RelayOutcome,
    TrialEvent,
    TrialIdentity,
)
from harbor_sandbox_agent_proxy.relay import RelayCall, RelayPhase, relay_methods
from harbor_sandbox_agent_proxy.trial_relay import TrialRelay


class _Sink(EventSink):
    def __init__(self) -> None:
        self.events: list[TrialEvent] = []
        self.outcome: RelayOutcome | None = None

    async def record(self, event: TrialEvent) -> None:
        self.events.append(event)

    async def close(self, outcome: RelayOutcome) -> None:
        self.outcome = outcome


class _Upstream:
    async def resolve(self, *, dialect: RequestDialect, model: str) -> UpstreamTarget:
        del dialect, model
        return UpstreamTarget(
            "https://upstream.test/v1/messages", {"x-api-key": "host-key"}
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
) -> TrialRelay:
    return TrialRelay(
        trial=TrialIdentity(name="example__1", directory=Path("/tmp/example__1")),
        config=settings or _settings(),
        upstream=_Upstream(),
        sink=_Sink(),
        environment=environment,
        bridged_servers=(
            BridgedMcpServer(
                name="task-tools",
                command="python3",
                args=("/opt/task_tools.py",),
                exclude_tools=frozenset({"admin"}),
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


def test_the_relay_registers_both_caller_keys_for_redaction() -> None:
    relay_instance = _relay()

    assert {
        relay_instance.agent_key,
        relay_instance.verifier_key,
    } <= relay_instance.protected_values()
    assert relay_instance.agent_key != relay_instance.verifier_key


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
    assert first in relay_instance.protected_values()
    assert second in relay_instance.protected_values()
    assert relay_instance.phase_env()["ANTHROPIC_API_KEY"] == second
    calls: list[RelayCall] = []

    async def record(call: RelayCall) -> None:
        calls.append(call)

    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        generate = relay_methods(
            relay_instance.current_policy,
            relay_instance.upstream,
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
            relay_instance.upstream,
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

    monkeypatch.setattr(trial_relay, "model_event_from_call", fail_conversion)
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

    monkeypatch.setattr(trial_relay, "model_event_from_call", fail_conversion)
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

    monkeypatch.setattr(trial_relay, "model_event_from_call", converted)
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
        upstream=_Upstream(),
        sink=_Sink(),
    )
    observed_bodies: list[dict[str, object]] = []
    observed_headers: list[Mapping[str, str]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed_bodies.append(json.loads(request.content))
        observed_headers.append(request.headers)
        return httpx.Response(200, json={"content": [], "role": "assistant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        generate = relay_methods(
            relay.current_policy, relay.upstream, record=relay.record, client=client
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
            upstream=_Upstream(),
            sink=_Sink(),
        )
        sent: list[httpx.Request] = []

        def count(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"content": [], "role": "assistant"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(count)) as client:
            generate = relay_methods(
                bounded.current_policy,
                bounded.upstream,
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
