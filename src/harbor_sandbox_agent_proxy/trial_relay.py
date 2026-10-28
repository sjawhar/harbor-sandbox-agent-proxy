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
import json
import logging
import time
from collections.abc import Callable, Collection, Sequence
from typing import Awaitable
from uuid import uuid4

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

from . import model_events
from .config import RelayConfig
from .connection import AgentConnection, served_connection_env
from .conversation import BridgeConversationSelector
from .outcome import EventSink, RelayOutcome, TrialIdentity
from .proxy import sandbox_model_proxy
from .relay import (
    GATEWAY_TIMEOUT,
    RelayCall,
    RelayPhase,
    RelayPolicy,
    mint_caller_key,
    relay_methods,
)
from .routing import RouteCall
from .sandbox_adapter import HarborSandboxEnvironment
from .sandbox_binding import bind_sandbox_environment
from .task_metadata import BridgedMcpServer

_PROXY_READY_TIMEOUT_SEC = 30.0
_PROXY_READY_INTERVAL_SEC = 0.1
logger = logging.getLogger(__name__)


def _nothing() -> Collection[str]:
    return ()


@dataclasses.dataclass
class TrialRelay:
    """Run and record one installed Harbor agent's in-sandbox model proxy."""

    trial: TrialIdentity
    config: RelayConfig
    route: RouteCall
    sink: EventSink
    connection: AgentConnection
    bridged_servers: Sequence[BridgedMcpServer] = ()
    protected: Callable[[], Collection[str]] = _nothing
    environment: harbor_environment_base.BaseEnvironment | None = None
    agent_key: str = dataclasses.field(default_factory=mint_caller_key)
    port: int | None = None
    phase: RelayPhase = "agent"
    verifier_key: str = dataclasses.field(default_factory=mint_caller_key)
    calls: list[RelayCall] = dataclasses.field(default_factory=list)
    events: list[inspect_ai.event.Event] = dataclasses.field(default_factory=list)
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

    @property
    def caller_keys(self) -> frozenset[str]:
        """Every caller key this relay minted across all of its phases."""
        return frozenset(self._caller_keys)

    @property
    def declared_models(self) -> frozenset[str]:
        """Models admitted by this trial's immutable relay configuration."""
        return frozenset(self.config.declared_models)

    def protected_values(self) -> frozenset[str]:
        """Expose redaction values in memory only; records never receive key values."""
        return frozenset({*self.caller_keys, *self.protected()})

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
        """The current phase's declared model-connection variables."""
        assert self.port is not None
        caller_key = self.agent_key if self.phase == "agent" else self.verifier_key
        connection = self.connection.proxy_env(key=caller_key, port=self.port)
        if self.phase == "agent":
            return connection
        return {
            **served_connection_env(key=caller_key, port=self.port),
            **connection,
        }

    async def open_agent_phase(self) -> None:
        """Start the proxy and hold its task-bound resources until ``close``."""
        if self._proxy_task is not None:
            raise RuntimeError("the trial relay is already open")
        if self.environment is None:
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

    async def end_agent_phase(self) -> None:
        """Swap credentials and select the conversation the verifier may inspect."""
        self.phase = "verifier"
        self.verifier_key = mint_caller_key()
        self._caller_keys.add(self.verifier_key)
        selected = self._selector.selected
        if self._answered_agent_calls == 0 or selected is None:
            self.refusal_reason = (
                "the agent made no answered model call through the relay"
            )
            self.verifier_conversation = {"refused": self.refusal_reason}
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
            return
        checked = _conversation_as_recorded(
            selected.messages, self._answered_agent_events[:selected_index]
        )
        if isinstance(checked, str):
            self.refusal_reason = checked
            self.verifier_conversation = {"refused": checked}
            return
        self._checked_conversation = checked
        self.verifier_conversation = {
            "messages": [message.model_dump(mode="json") for message in checked]
        }

    async def resume_agent_phase(self) -> None:
        """Re-enter the agent phase for a later step of a multi-step trial."""
        self.phase = "agent"
        self.verifier_conversation = None
        self._checked_conversation = None
        self.refusal_reason = None

    async def close(self) -> RelayOutcome:
        """Stop the proxy and return every fact the host-side environment needs."""
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
        return self._outcome()

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
                                self.route,
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

    async def _record_refused_tool_call(
        self, event: inspect_ai.event.ToolEvent
    ) -> None:
        self.events.append(event)
        await self.sink.record(event)

    async def record(self, call: RelayCall) -> None:
        """Append one call and its Inspect events, preserving the first sink failure."""
        self.calls.append(call)
        self._counts[f"{call.phase}_calls"] += 1
        self._dropped_betas.update(
            stripped.removeprefix("anthropic-beta:")
            for stripped in call.stripped
            if stripped.startswith("anthropic-beta:")
        )
        try:
            event = await model_events.model_event_from_call(call)
            if event is None:
                self._counts["refused_calls"] += 1
                return
            if event.error is not None:
                self._counts["failed_calls"] += 1
            if (
                self.phase == "agent"
                and call.phase == "agent"
                and _answered(call, event)
            ):
                if self._answered_agent_events:
                    for tool_event in model_events.tool_events_between(
                        self._answered_agent_events[-1], event.input
                    ):
                        self.events.append(tool_event)
                        await self.sink.record(tool_event)
                self._selector.observe(event.input, event.output)
                self._answered_agent_events.append(event)
                self._answered_agent_calls += 1
            self.events.append(event)
            await self.sink.record(event)
        except Exception as error:
            if self.record_error is None:
                self.record_error = error
            raise


def phase_guarded_tool(
    tool: inspect_ai.tool.Tool,
    *,
    server: str,
    phase: Callable[[], RelayPhase],
    refused: Callable[[inspect_ai.event.ToolEvent], Awaitable[None]],
) -> inspect_ai.tool.Tool:
    """Serve a bridged task tool only during the agent's turn."""
    inner = inspect_ai.tool.ToolDef(tool)

    async def guard(**arguments: pydantic.JsonValue) -> inspect_ai.tool.ToolResult:
        current = phase()
        if current == "agent":
            return await tool(**arguments)
        message = (
            f"{server}/{inner.name} is not served outside the agent's turn "
            f"(relay phase: {current})"
        )
        now = datetime.datetime.now(datetime.UTC)
        await refused(
            inspect_ai.event.ToolEvent(
                id=str(uuid4()),
                function=inner.name,
                arguments=arguments,
                result="",
                error=inspect_ai.tool.ToolCallError("permission", message),
                pending=False,
                timestamp=now,
                completed=now,
                metadata={
                    model_events.RELAY_METADATA_KEY: {
                        "phase": current,
                        "server": server,
                        "refused": "outside the agent's turn",
                    }
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


def _answered(call: RelayCall, event: inspect_ai.event.ModelEvent) -> bool:
    return call.status is not None and 200 <= call.status < 300 and event.error is None


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


def _now_json() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
