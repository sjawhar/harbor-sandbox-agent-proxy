"""Per-trial lifecycle for the sandbox model relay.

One relay owns the proxy task, phase-specific caller keys, selected agent conversation,
and an injected event sink for one Harbor trial. The proxy remains open across all agent
and verifier phases; its request handlers read ``current_policy`` for every call.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Awaitable, TypeGuard

import anyio
import harbor.environments.base as harbor_environment_base
import harbor.models.task.config as harbor_task_config
import httpx
import inspect_ai.event
import inspect_ai.model
import inspect_ai.tool
import inspect_ai.util
import pydantic
from inspect_ai.agent import BridgedToolsSpec

from .collaborators import BridgedMcpServer, SecretSource, Upstream
from .config import RelayConfig
from .conversation import BridgeConversationSelector
from .outcome import EventSink, RelayOutcome, TrialEvent, TrialIdentity
from .proxy import sandbox_model_proxy
from .relay import (
    GATEWAY_TIMEOUT,
    RelayCall,
    RelayPhase,
    RelayPolicy,
    mint_caller_key,
    relay_methods,
)
from .sandbox_adapter import HarborSandboxEnvironment
from .sandbox_binding import bind_sandbox_environment

if TYPE_CHECKING:
    from anthropic.types import MessageParam
    from google.genai.types import ContentDict
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.responses import ResponseInputItemParam


_PROXY_READY_TIMEOUT_SEC = 30.0
_PROXY_READY_INTERVAL_SEC = 0.1
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrialRelay:
    """Run and record one installed Harbor agent's in-sandbox model proxy."""

    trial: TrialIdentity
    config: RelayConfig
    upstream: Upstream
    sink: EventSink
    bridged_servers: Sequence[BridgedMcpServer] = ()
    secrets: SecretSource | None = None
    environment: harbor_environment_base.BaseEnvironment | None = None
    agent_model: str | None = None
    agent_key: str = dataclasses.field(default_factory=mint_caller_key)
    port: int | None = None
    phase: RelayPhase = "agent"
    verifier_key: str = dataclasses.field(default_factory=mint_caller_key)
    calls: list[RelayCall] = dataclasses.field(default_factory=list)
    events: list[TrialEvent] = dataclasses.field(default_factory=list)
    verifier_conversation: dict[str, pydantic.JsonValue] | None = None
    refusal_reason: str | None = None
    record_error: Exception | None = None
    _answered_agent_calls: int = 0
    _counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: {
            "agent_calls": 0,
            "verifier_calls": 0,
            "failed_calls": 0,
            "refused_calls": 0,
        }
    )
    _dropped_betas: set[str] = dataclasses.field(default_factory=set)
    _selector: BridgeConversationSelector = dataclasses.field(
        default_factory=BridgeConversationSelector
    )
    _answered_agent_events: list[inspect_ai.event.ModelEvent] = dataclasses.field(
        default_factory=list
    )
    _checked_conversation: list[inspect_ai.model.ChatMessage] | None = None
    _proxy_task: asyncio.Task[None] | None = None
    _proxy_stop: asyncio.Event | None = None
    _proxy_exit: dict[str, object] | None = None
    _bridged_tools: tuple[BridgedToolsSpec, ...] = ()
    _caller_keys: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        if self.port is None:
            self.port = self.config.port
        self._caller_keys.update((self.agent_key, self.verifier_key))
        self._emit_key_minted("agent", self.agent_key)
        self._emit_key_minted("verifier", self.verifier_key)

    @property
    def declared_models(self) -> frozenset[str]:
        """Models admitted by this trial's immutable relay configuration."""
        return frozenset(self.config.declared_models)

    def protected_values(self) -> frozenset[str]:
        """Expose redaction values in memory only; records never receive key values."""
        source = () if self.secrets is None else self.secrets.protected_values()
        return frozenset({*self._caller_keys, *source})

    def mcp_server_configs(self) -> list[harbor_task_config.MCPServerConfig]:
        """The loopback MCP endpoints an installed agent receives when it starts."""
        assert self.port is not None
        return [
            harbor_task_config.MCPServerConfig(
                name=server.name,
                transport="streamable-http",
                url=f"http://localhost:{self.port}/mcp/{server.name}",
            )
            for server in self.bridged_servers
        ]

    def phase_env(self) -> dict[str, str]:
        """Every managed provider spelling for the current relay phase."""
        assert self.port is not None
        caller_key = self.agent_key if self.phase == "agent" else self.verifier_key
        environment = {
            "ANTHROPIC_API_KEY": caller_key,
            "ANTHROPIC_AUTH_TOKEN": caller_key,
            "ANTHROPIC_BASE_URL": f"http://localhost:{self.port}",
            "OPENAI_API_KEY": caller_key,
            "OPENAI_BASE_URL": f"http://localhost:{self.port}/v1",
            "OPENAI_API_BASE": f"http://localhost:{self.port}/v1",
            "GEMINI_API_KEY": caller_key,
            "GOOGLE_API_KEY": caller_key,
            "GOOGLE_GENERATIVE_AI_API_KEY": caller_key,
            "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{self.port}",
        }
        model = self.agent_model or next(iter(sorted(self.declared_models)), None)
        if model is not None:
            canonical = model.partition("/")[2] or model
            environment["ANTHROPIC_SMALL_FAST_MODEL"] = canonical
            environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = canonical
        return environment

    async def open_agent_phase(self) -> None:
        """Start the proxy and hold its task-bound resources until ``close``."""
        if self._proxy_task is not None:
            raise RuntimeError("the trial relay is already open")
        if self.environment is None:
            await self._record_phase("agent-opened")
            return
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        task = asyncio.create_task(self._hold_proxy(ready=ready, stop=stop))
        try:
            await ready
        except BaseException:
            stop.set()
            await task
            raise
        self._proxy_stop = stop
        self._proxy_task = task
        await self._record_phase("agent-opened")

    async def end_agent_phase(self) -> None:
        """Swap credentials and select the conversation the verifier may inspect."""
        self.phase = "verifier"
        self.verifier_key = mint_caller_key()
        self._caller_keys.add(self.verifier_key)
        await self._record_key_minted("verifier", self.verifier_key)
        selected = self._selector.selected
        if self._answered_agent_calls == 0 or selected is None:
            self.refusal_reason = (
                "the agent made no answered model call through the relay"
            )
            self.verifier_conversation = {"refused": self.refusal_reason}
            await self._record_phase("verifier-opened")
            return
        selected_index = next(
            (
                index
                for index, event in enumerate(self._answered_agent_events)
                if event.output is selected.output
            ),
            None,
        )
        if selected_index is None:
            self.refusal_reason = "the relay selected a conversation with no corresponding recorded response"
            self.verifier_conversation = {"refused": self.refusal_reason}
            await self._record_phase("verifier-opened")
            return
        checked = _conversation_as_recorded(
            selected.messages, self._answered_agent_events[:selected_index]
        )
        if isinstance(checked, str):
            self.refusal_reason = checked
            self.verifier_conversation = {"refused": checked}
        else:
            self._checked_conversation = checked
            self.verifier_conversation = {
                "messages": [message.model_dump(mode="json") for message in checked]
            }
        await self._record_phase("verifier-opened")

    async def resume_agent_phase(self) -> None:
        """Re-enter the agent phase for a later step of a multi-step trial."""
        self.phase = "agent"
        self.verifier_conversation = None
        self._checked_conversation = None
        self.refusal_reason = None
        await self._record_phase("agent-resumed")

    async def close(self) -> RelayOutcome:
        """Stop the proxy and return every fact the host-side sink needs."""
        task, self._proxy_task = self._proxy_task, None
        stop, self._proxy_stop = self._proxy_stop, None
        if stop is not None:
            stop.set()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                self._proxy_exit = {"at": _now_json(), "phase": self.phase}
            except Exception as error:
                logger.warning(
                    "The trial relay proxy exited with an error", exc_info=True
                )
                self._proxy_exit = {
                    "at": _now_json(),
                    "phase": self.phase,
                    "error": repr(error),
                }
        await self._record_phase("closed")
        outcome = self._outcome()
        await self.sink.close(outcome)
        return outcome

    def _outcome(self) -> RelayOutcome:
        return RelayOutcome(
            mode="relay",
            calls=list(self.calls),
            events=list(self.events),
            conversation=(
                None
                if self._checked_conversation is None
                else list(self._checked_conversation)
            ),
            answered_agent_calls=self._answered_agent_calls,
            counts=dict(self._counts),
            dropped_betas=frozenset(self._dropped_betas),
            refusal_reason=self.refusal_reason,
            proxy_exit=self._proxy_exit,
            runner_atif=None,
            runner_atif_sha256=None,
            record_error=repr(self.record_error)
            if self.record_error is not None
            else None,
        )

    def current_policy(self) -> tuple[RelayPhase, RelayPolicy]:
        """The phase and complete policy the next forwarder call applies."""
        allowed_betas = self.config.allowed_betas
        tools = self.config.tools
        if self.phase == "verifier":
            if self.config.verifier_allowed_betas is not None:
                allowed_betas = self.config.verifier_allowed_betas
            if self.config.verifier_tools is not None:
                tools = self.config.verifier_tools
        bridged = {
            spec.name: frozenset(
                inspect_ai.tool.ToolDef(tool).name for tool in spec.tools
            )
            for spec in self._bridged_tools
        }
        return self.phase, RelayPolicy(
            caller_key=self.agent_key if self.phase == "agent" else self.verifier_key,
            allowed_models=self.declared_models,
            tools=tools,
            permitted_tool_names=frozenset(self.config.permitted_tool_names),
            permitted_tool_prefixes=self.config.permitted_tool_prefixes,
            bridged=bridged if self.phase == "agent" else {},
            allowed_betas=(None if allowed_betas is None else frozenset(allowed_betas)),
            max_request_bytes=self.config.max_request_bytes,
            max_calls=self.config.max_calls,
            max_recorded_bytes=self.config.max_recorded_bytes,
        )

    async def _hold_proxy(
        self, *, ready: asyncio.Future[None], stop: asyncio.Event
    ) -> None:
        assert self.environment is not None
        assert self.port is not None
        try:
            async with contextlib.AsyncExitStack() as stack:
                sandbox = stack.enter_context(
                    bind_sandbox_environment(
                        HarborSandboxEnvironment(
                            self.environment,
                            default_user="root",
                            polling_interval=self.config.polling_interval,
                        )
                    )
                )
                self._bridged_tools = tuple(await self._create_bridged_tools(stack))
                async with httpx.AsyncClient(timeout=GATEWAY_TIMEOUT) as client:
                    proxy = await stack.enter_async_context(
                        sandbox_model_proxy(
                            sandbox,
                            methods=relay_methods(
                                self.current_policy,
                                self.upstream,
                                record=self.record,
                                client=client,
                            ),
                            port=self.port,
                            bridged_tools=self._bridged_tools,
                            polling_interval=self.config.polling_interval,
                        )
                    )
                    self.port = proxy.port
                    await _wait_for_proxy(sandbox, port=proxy.port)
                    if not ready.done():
                        ready.set_result(None)
                    await stop.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
                return
            self._proxy_exit = {"at": _now_json(), "phase": self.phase}
            raise

    async def _create_bridged_tools(
        self, stack: contextlib.AsyncExitStack
    ) -> list[BridgedToolsSpec]:
        specs: list[BridgedToolsSpec] = []
        for server in self.bridged_servers:
            mcp_server = inspect_ai.tool.mcp_server_sandbox(
                name=server.name, command=server.command, args=list(server.args)
            )
            source = inspect_ai.tool.mcp_tools(mcp_server)
            await stack.enter_async_context(inspect_ai.tool.mcp_connection(source))
            tools = await source.tools()
            specs.append(
                BridgedToolsSpec(
                    name=server.name,
                    tools=[
                        phase_guarded_tool(
                            tool,
                            server=server.name,
                            phase=lambda: self.phase,
                            refused=self._record_refused_tool_call,
                        )
                        for tool in tools
                        if inspect_ai.tool.ToolDef(tool).name
                        not in server.exclude_tools
                    ],
                    require_proposal=server.require_proposal,
                )
            )
        return specs

    async def _record_refused_tool_call(self, event: TrialEvent) -> None:
        self.events.append(event)
        await self.sink.record(event)

    async def record(self, call: RelayCall) -> None:
        """Append a call and its record, preserving the first sink failure."""
        self.calls.append(call)
        self._counts[f"{call.phase}_calls"] += 1
        self._dropped_betas.update(
            stripped.removeprefix("anthropic-beta:")
            for stripped in call.stripped
            if stripped.startswith("anthropic-beta:")
        )
        try:
            event = await model_event_from_call(call)
            if event is None:
                self._counts["refused_calls"] += 1
                await self._record_call_event(call, kind="refused-call")
                return
            if event.error is not None:
                self._counts["failed_calls"] += 1
            if (
                self.phase == "agent"
                and call.phase == "agent"
                and _answered(call, event)
            ):
                if self._answered_agent_events:
                    for tool_event in _tool_events_between(
                        self._answered_agent_events[-1], event.input
                    ):
                        await self._record_event(tool_event)
                self._selector.observe(event.input, event.output)
                self._answered_agent_events.append(event)
                self._answered_agent_calls += 1
            await self._record_event(event)
        except Exception as error:
            if self.record_error is None:
                self.record_error = error
            raise

    async def _record_call_event(self, call: RelayCall, *, kind: str) -> None:
        event = TrialEvent(
            kind=kind,
            phase=call.phase,
            data={
                "method": call.method,
                "model": call.model,
                "status": call.status,
                "refused": call.refused,
                "stripped": list(call.stripped),
            },
        )
        self.events.append(event)
        await self.sink.record(event)

    async def _record_event(
        self, event: inspect_ai.event.ModelEvent | inspect_ai.event.ToolEvent
    ) -> None:
        if isinstance(event, inspect_ai.event.ToolEvent):
            record = TrialEvent(
                kind="tool-call",
                phase=self.phase,
                data={
                    "id": event.id,
                    "function": event.function,
                    "arguments": event.arguments,
                    "result": event.result if isinstance(event.result, str) else "",
                    "error": None if event.error is None else event.error.message,
                    "metadata": event.metadata or {},
                },
            )
        else:
            record = TrialEvent(
                kind="model-call",
                phase=self.phase,
                data={
                    "model": event.model,
                    "error": event.error,
                    "metadata": event.metadata or {},
                },
            )
        self.events.append(record)
        await self.sink.record(record)

    def _emit_key_minted(self, phase: RelayPhase, key: str) -> None:
        self.events.append(
            TrialEvent(
                kind="caller-key-minted",
                phase=phase,
                data={"fingerprint": _fingerprint(key)},
            )
        )

    async def _record_key_minted(self, phase: RelayPhase, key: str) -> None:
        event = TrialEvent(
            kind="caller-key-minted",
            phase=phase,
            data={"fingerprint": _fingerprint(key)},
        )
        self.events.append(event)
        await self.sink.record(event)

    async def _record_phase(self, kind: str) -> None:
        event = TrialEvent(kind=kind, phase=self.phase, data={})
        self.events.append(event)
        await self.sink.record(event)


def phase_guarded_tool(
    tool: inspect_ai.tool.Tool,
    *,
    server: str,
    phase: Callable[[], RelayPhase],
    refused: Callable[[TrialEvent], Awaitable[None]],
) -> inspect_ai.tool.Tool:
    """Serve a bridged task tool only during the agent's turn."""
    inner = inspect_ai.tool.ToolDef(tool)

    async def guard(**arguments: pydantic.JsonValue) -> inspect_ai.tool.ToolResult:
        current = phase()
        if current == "agent":
            return await tool(**arguments)
        message = f"{server}/{inner.name} is not served outside the agent's turn (relay phase: {current})"
        await refused(
            TrialEvent(
                kind="refused-tool-call",
                phase=current,
                data={
                    "server": server,
                    "tool": inner.name,
                    "arguments": arguments,
                    "reason": "outside the agent's turn",
                },
            )
        )
        raise inspect_ai.tool.ToolError(message)

    return inspect_ai.tool.ToolDef(
        guard,
        name=inner.name,
        description=inner.description,
        parameters=inner.parameters,
        parallel=inner.parallel,
        viewer=inner.viewer,
        model_input=inner.model_input,
        max_output=inner.max_output,
        options=inner.options,
    ).as_tool()


async def model_event_from_call(call: RelayCall) -> inspect_ai.event.ModelEvent | None:
    """Convert a recorded successful provider call into the selector's Inspect event."""
    if not call.authenticated:
        return None
    failure = _call_error(call)
    messages: list[inspect_ai.model.ChatMessage] = []
    output: inspect_ai.model.ModelOutput | None = None
    try:
        if not isinstance(call.request, dict):
            raise ValueError("relay request is not an object")
        if call.method == "generate_anthropic":
            raw = call.request.get("messages")
            if not _is_anthropic_messages(raw):
                raise ValueError("Anthropic request holds no message list")
            messages = await inspect_ai.model.messages_from_anthropic(raw)
            if failure is None and isinstance(call.response, dict):
                output = await inspect_ai.model.model_output_from_anthropic(
                    call.response
                )
        elif call.method == "generate_completions":
            raw = call.request.get("messages")
            if not _is_completion_messages(raw):
                raise ValueError("Chat Completions request holds no message list")
            messages = await inspect_ai.model.messages_from_openai(
                raw, model=call.model
            )
            if failure is None and isinstance(call.response, dict):
                output = await inspect_ai.model.model_output_from_openai(call.response)
        elif call.method == "generate_responses":
            raw = call.request.get("input")
            if isinstance(raw, str):
                messages = await inspect_ai.model.messages_from_openai_responses(
                    [{"role": "user", "content": raw}], model=call.model
                )
            elif _is_response_items(raw):
                messages = await inspect_ai.model.messages_from_openai_responses(
                    raw, model=call.model
                )
            else:
                raise ValueError("Responses request holds no input")
            if failure is None and isinstance(call.response, dict):
                output = await inspect_ai.model.model_output_from_openai_responses(
                    call.response
                )
        elif call.method == "generate_google":
            raw = call.request.get("contents")
            if not _is_google_contents(raw):
                raise ValueError("Gemini request holds no contents")
            messages = await inspect_ai.model.messages_from_google(
                raw, model=call.model
            )
            if failure is None and isinstance(call.response, dict):
                output = await inspect_ai.model.model_output_from_google(
                    call.response, model=call.model
                )
        else:
            raise ValueError(f"relay method {call.method!r} has no converter")
    except (ValueError, TypeError, KeyError) as error:
        failure = (
            failure
            or f"{call.method}: conversion failed: {type(error).__name__}: {error}"
        )
        messages = []
    if output is None:
        output = inspect_ai.model.ModelOutput(
            model=call.model or "", choices=[], error=failure
        )
    return inspect_ai.event.ModelEvent(
        model=call.provider if call.model is None else f"{call.provider}/{call.model}",
        input=messages,
        tools=[],
        tool_choice="none",
        config=inspect_ai.model.GenerateConfig(),
        output=output,
        error=failure,
        timestamp=call.ts_request,
        completed=call.ts_response,
        working_time=call.latency_ms / 1000.0 if failure is None else None,
        metadata={
            "sandbox_agent_proxy": {
                "phase": call.phase,
                "status": call.status,
                "latency_ms": call.latency_ms,
                "stripped": list(call.stripped),
            }
        },
    )


def _is_anthropic_messages(value: object) -> TypeGuard[list[MessageParam]]:
    return _is_list_of_objects(value)


def _is_completion_messages(
    value: object,
) -> TypeGuard[list[ChatCompletionMessageParam]]:
    return _is_list_of_objects(value)


def _is_response_items(value: object) -> TypeGuard[list[ResponseInputItemParam]]:
    return _is_list_of_objects(value)


def _is_google_contents(value: object) -> TypeGuard[list[ContentDict]]:
    return _is_list_of_objects(value)


def _is_list_of_objects(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _tool_events_between(
    previous: inspect_ai.event.ModelEvent,
    next_input: Sequence[inspect_ai.model.ChatMessage],
) -> list[inspect_ai.event.ToolEvent]:
    if not previous.output.choices:
        return []
    results = {
        message.tool_call_id: message
        for message in next_input
        if isinstance(message, inspect_ai.model.ChatMessageTool)
        and message.tool_call_id
    }
    when = previous.completed or previous.timestamp
    events: list[inspect_ai.event.ToolEvent] = []
    for tool_call in previous.output.message.tool_calls or []:
        result = results.get(tool_call.id)
        if result is None:
            continue
        events.append(
            inspect_ai.event.ToolEvent(
                id=tool_call.id,
                function=tool_call.function,
                arguments=tool_call.arguments,
                result=result.text,
                error=result.error,
                timestamp=when,
                completed=when,
                metadata={"source": "sandbox_agent_proxy"},
            )
        )
    return events


def _answered(call: RelayCall, event: inspect_ai.event.ModelEvent) -> bool:
    return call.status is not None and 200 <= call.status < 300 and event.error is None


def _call_error(call: RelayCall) -> str | None:
    if (
        call.status is not None
        and 200 <= call.status < 300
        and isinstance(call.response, dict)
    ):
        return None
    body = json.dumps(call.response, separators=(",", ":"))
    if call.status is None:
        return f"{call.method}: the request never reached the upstream: {body}"
    return f"{call.method}: the upstream returned HTTP {call.status}: {body}"


def _assistant_key(message: inspect_ai.model.ChatMessageAssistant) -> str:
    turn = message.model_dump(
        mode="json",
        exclude={
            "id": True,
            "source": True,
            "model": True,
            "metadata": True,
            "internal": True,
            "content": {"__all__": {"internal"}},
        },
        exclude_none=True,
    )
    if not message.tool_calls:
        turn.pop("tool_calls", None)
    return json.dumps(turn, sort_keys=True)


def _conversation_as_recorded(
    messages: Sequence[inspect_ai.model.ChatMessage],
    preceding_events: Sequence[inspect_ai.event.ModelEvent],
) -> list[inspect_ai.model.ChatMessage] | str:
    """Replace each re-sent assistant turn with its recorded prior reply."""
    recorded: dict[str, inspect_ai.model.ChatMessageAssistant] = {}
    for event in preceding_events:
        if event.output.choices:
            recorded.setdefault(
                _assistant_key(event.output.message), event.output.message
            )
    conversation: list[inspect_ai.model.ChatMessage] = []
    for position, message in enumerate(messages[:-1]):
        if not isinstance(message, inspect_ai.model.ChatMessageAssistant):
            conversation.append(message)
            continue
        reply = recorded.get(_assistant_key(message))
        if reply is None:
            return (
                "the selected conversation carries an assistant turn at position "
                + f"{position} that no response the relay recorded earlier produced: "
                + "the transcript was not produced by the graded model"
            )
        conversation.append(reply)
    conversation.extend(messages[-1:])
    return conversation


async def _wait_for_proxy(
    sandbox: inspect_ai.util.SandboxEnvironment, *, port: int
) -> None:
    deadline = time.monotonic() + _PROXY_READY_TIMEOUT_SEC
    failure = "the proxy did not answer"
    probe = (
        "from http.client import HTTPConnection; "
        + f"connection = HTTPConnection('localhost', {port}, timeout=1); "
        + "connection.request('GET', '/'); print(connection.getresponse().status)"
    )
    while time.monotonic() < deadline:
        result = await sandbox.exec(["python3", "-c", probe], user="root")
        if result.success and result.stdout.strip() not in {"", "000"}:
            return
        failure = result.stderr.strip() or result.stdout.strip() or failure
        await anyio.sleep(_PROXY_READY_INTERVAL_SEC)
    raise RuntimeError(
        "the relay proxy did not answer inside the sandbox within "
        + f"{_PROXY_READY_TIMEOUT_SEC:.0f}s: {failure}"
    )


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _now_json() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
