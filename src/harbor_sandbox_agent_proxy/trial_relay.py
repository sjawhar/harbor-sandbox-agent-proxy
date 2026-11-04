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
from .budget import TrialBudget
from .budget_shadow import ShadowBudget
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
from .routing import RequestDialect, RouteCall
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
    budget: TrialBudget | None = dataclasses.field(default=None, init=False)
    shadow: ShadowBudget | None = dataclasses.field(default=None, init=False)
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
    _close_task: asyncio.Task[RelayOutcome] | None = dataclasses.field(default=None, init=False)
    _proxy_stop: asyncio.Event | None = None
    _proxy_exit: dict[str, object] | None = None
    _bridged_tools: tuple[BridgedToolsSpec, ...] = ()
    _caller_keys: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        if self.port is None:
            self.port = self.config.port
        self._caller_keys.update((self.agent_key, self.verifier_key))
        if self.config.token_budget is not None and self.config.token_budget.enforce:
            self.budget = TrialBudget(self.config.token_budget,
                                      self.trial.directory / "sandbox-agent-proxy", self.trial.name,
                                      on_error=self._budget_error)
        elif self.config.token_budget is not None:
            self.shadow = ShadowBudget(self.config.token_budget,
                                       self.trial.directory / "sandbox-agent-proxy", self.trial.name,
                                       protected=self.protected_values)

    def _budget_error(self, error: Exception) -> None:
        if self.record_error is None:
            self.record_error = error

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
        if self.config.record == "usage":
            self.refusal_reason = "the conversation is not recorded in usage mode"
            self.verifier_conversation = {"refused": self.refusal_reason}
            return
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
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_once())
        return await asyncio.shield(self._close_task)

    async def _close_once(self) -> RelayOutcome:
        if self.budget is not None:
            try:
                await self.budget.close_out()
            except Exception as error:
                self._budget_error(error)
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
        if self.shadow is not None:
            await self.shadow.close()
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
            budget=self.budget.record() if self.budget is not None else None,
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
                                budget=self.budget,
                                usage_only=self.config.record == "usage",
                                shadow=self.shadow,
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
        if self.config.record == "usage":
            await self.sink.record(event)
            allowed = {
                inspect_ai.tool.ToolDef(tool).name
                for spec in self._bridged_tools
                for tool in spec.tools
            }
            self._append_usage_line(
                {
                    "call_id": uuid4().hex,
                    "call_id_source": "generated",
                    "trial": self.trial.name,
                    "phase": self.phase,
                    "kind": "tool_refused",
                    "tool": event.function if event.function in allowed else None,
                    "refused": "tool_outside_agent_phase",
                }
            )
            return
        self.events.append(event)
        await self.sink.record(event)

    def _append_usage_line(self, line: dict[str, object]) -> None:
        redacted = json.dumps(line, sort_keys=True)
        for secret in self.protected_values():
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        destination = self.trial.directory / "sandbox-agent-proxy" / "usage.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(redacted + "\n")

    async def _record_usage(self, call: RelayCall) -> None:
        """Count one call, pass its model event to the sink live, and append its token
        usage to usage.jsonl; keep no call or event in memory."""
        self._counts[f"{call.phase}_calls"] += 1
        self._dropped_betas.update(
            stripped.removeprefix("anthropic-beta:")
            for stripped in call.stripped
            if stripped.startswith("anthropic-beta:")
        )
        try:
            event = await model_events.model_event_from_call(call)
            usage = None
            if event is None:
                self._counts["refused_calls"] += 1
            else:
                if event.error is not None:
                    self._counts["failed_calls"] += 1
                if (
                    self.phase == "agent"
                    and call.phase == "agent"
                    and _answered(call, event)
                ):
                    self._answered_agent_calls += 1
                usage = None if event.output is None else event.output.usage
                if usage is None and event.error is not None:
                    usage = await _error_usage(call)
                await self.sink.record(event)
            line = {
                **_call_id(call),
                "dialect": _dialect(call),
                "trial": self.trial.name,
                "phase": call.phase,
                "ts_request": call.ts_request.isoformat(),
                "ts_response": call.ts_response.isoformat(),
                "model": call.model
                if call.model is not None
                and any(
                    declared in (call.model, f"{call.provider}/{call.model}")
                    for declared in self.config.declared_models
                )
                else None,
                "status": call.status,
                "refused": _refusal_code(call.refused),
                **_reported_counts(call, usage),
                "request_bytes": len(json.dumps(call.request).encode("utf-8")),
                "response_bytes": len(json.dumps(call.response).encode("utf-8")),
                "recorded_bytes": len(call.model_dump_json().encode("utf-8")),
            }
            if call.budget is not None:
                line["budget"] = call.budget
                line["call_id"] = call.budget["call_id"]
                line["refused"] = call.budget.get("refusal_code") or line["refused"]
            self._append_usage_line(line)
        except Exception as error:
            if self.record_error is None:
                self.record_error = error
            raise

    async def record(self, call: RelayCall) -> None:
        """Append one call and its Inspect events, preserving the first sink failure."""
        if self.config.record == "usage":
            await self._record_usage(call)
            return
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


# usage.jsonl carries no free text from a request or a refusal (a refusal message can
# echo the request's model field, which a caller may set to its key): only these codes.
USAGE_REFUSAL_CODES = frozenset(
    {
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
)


USAGE_DIALECTS = frozenset(dialect.value for dialect in RequestDialect)
_CALL_ID_HEADER = "x-bet-eval-call-id"


def _call_id(call: RelayCall) -> dict[str, str]:
    """The host's own call id when the call carried one (a bounded token), else a fresh id."""
    for key, value in call.headers.items():
        if (
            key.lower() == _CALL_ID_HEADER
            and 0 < len(value) <= 128
            and all(c.isalnum() or c in "-_.:" for c in value)
        ):
            return {"call_id": value, "call_id_source": "header"}
    return {"call_id": uuid4().hex, "call_id_source": "generated"}


def _dialect(call: RelayCall) -> str | None:
    value = str(getattr(call.dialect, "value", call.dialect))
    return value if value in USAGE_DIALECTS else None


_Path = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _UsageSource:
    """Every raw field a dialect's Inspect converter reads into the written counts.

    ``required`` maps each written count to its provider field; ``optional`` are the
    other count fields that feed the split (cache, thoughts); ``within`` says which of
    them a total must cover; ``iterations`` is Anthropic's per-iteration usage list,
    which Inspect sums instead of the top-level pair when it is present.
    """

    key: str
    required: dict[str, str]
    optional: tuple[_Path, ...] = ()
    within: tuple[tuple[str, tuple[_Path, ...]], ...] = ()
    iterations: bool = False


_CHAT_CACHE: tuple[_Path, ...] = (
    ("prompt_tokens_details", "cached_tokens"),
    ("prompt_tokens_details", "cache_write_tokens"),
)
_RESPONSES_CACHE: tuple[_Path, ...] = (
    ("input_tokens_details", "cached_tokens"),
    ("input_tokens_details", "cache_write_tokens"),
)
_ITERATION_COUNTS = ("input_tokens", "output_tokens")
_ITERATION_OPTIONAL = ("cache_creation_input_tokens", "cache_read_input_tokens")
_MAX_ITERATIONS = 1024

# A count the provider did not report as a non-negative integer (absent, a bool, a
# negative, a float or a string) is never written as a measured count: Inspect's
# converters default an absent count to 0 and do not check types, signs or totals.
_USAGE_SOURCES: dict[RequestDialect, _UsageSource] = {
    RequestDialect.MESSAGES: _UsageSource(
        "usage",
        {"input_tokens": "input_tokens", "output_tokens": "output_tokens"},
        optional=(("cache_creation_input_tokens",), ("cache_read_input_tokens",)),
        iterations=True,
    ),
    RequestDialect.CHAT_COMPLETIONS: _UsageSource(
        "usage",
        {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"},
        optional=_CHAT_CACHE,
        within=(("prompt_tokens", _CHAT_CACHE),),
    ),
    RequestDialect.RESPONSES: _UsageSource(
        "usage",
        {"input_tokens": "input_tokens", "output_tokens": "output_tokens"},
        optional=_RESPONSES_CACHE,
        within=(("input_tokens", _RESPONSES_CACHE),),
    ),
    RequestDialect.GEMINI: _UsageSource(
        "usageMetadata",
        {"input_tokens": "promptTokenCount", "output_tokens": "candidatesTokenCount"},
        optional=(("cachedContentTokenCount",), ("thoughtsTokenCount",)),
        within=(("promptTokenCount", (("cachedContentTokenCount",),)),),
    ),
}

# The smallest body each converter accepts around a usage object, so an error
# response's usage is normalised by the same Inspect converter as an answered call's.
_USAGE_ONLY_BODIES: dict[str, dict[str, pydantic.JsonValue]] = {
    "generate_anthropic": {
        "id": "usage",
        "type": "message",
        "role": "assistant",
        "model": "usage",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
    },
    "generate_completions": {
        "id": "usage",
        "object": "chat.completion",
        "created": 0,
        "model": "usage",
        "choices": [],
    },
    "generate_responses": {
        "id": "usage",
        "object": "response",
        "created_at": 0,
        "model": "usage",
        "output": [],
        "status": "incomplete",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
    "generate_google": {"candidates": []},
}

# Fields the SDK's usage schema requires but no written count is read from.
_USAGE_SCHEMA_DEFAULTS: dict[str, dict[str, pydantic.JsonValue]] = {
    "generate_completions": {"total_tokens": 0},
    "generate_responses": {
        "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
_COUNT_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
)
_MISSING = object()


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _at(raw: dict[str, pydantic.JsonValue], path: _Path) -> object:
    """The value at ``path``; ``_MISSING`` when absent, and ``raw`` itself (never a
    count) when a container on the way is present but not an object."""
    value: object = raw
    for part in path:
        if value is None:
            return _MISSING
        if not isinstance(value, dict):
            return raw
        value = value.get(part, _MISSING)  # pyright: ignore[reportUnknownMemberType]
        if value is _MISSING:
            return _MISSING
    return value


def _usage_source(
    call: RelayCall,
) -> tuple[_UsageSource, dict[str, pydantic.JsonValue]] | None:
    """The dialect's usage source and the response's raw usage object, if it has one."""
    try:
        source = _USAGE_SOURCES[RequestDialect(call.dialect)]
    except (KeyError, ValueError):
        return None
    raw = call.response.get(source.key) if isinstance(call.response, dict) else None
    return (source, raw) if isinstance(raw, dict) else None


def _iterations(raw: dict[str, pydantic.JsonValue]) -> list[pydantic.JsonValue] | None:
    """Anthropic's iteration list as Inspect reads it, checked; None when invalid.

    Inspect decides provenance first: when any entry is a ``fallback_message``, the
    written counts come from the top-level pair and the entries feed none of them (a
    declined attempt may have no output count), so only the list's shape is checked and
    nothing is kept. Otherwise the entries are compaction iterations, which Inspect sums
    instead of the top-level pair, so every entry's counts must be valid.
    """
    iterations = raw.get("iterations")
    if iterations is None:
        return []
    if not isinstance(iterations, list) or len(iterations) > _MAX_ITERATIONS:
        return None
    kinds: list[str | None] = []
    for item in iterations:
        if not isinstance(item, dict):
            return None
        kind = item.get("type")
        if kind is not None and not (isinstance(kind, str) and len(kind) <= 64):
            return None
        kinds.append(kind)
    if "fallback_message" in kinds:
        return []
    kept: list[pydantic.JsonValue] = []
    for item, kind in zip(iterations, kinds, strict=True):
        assert isinstance(item, dict)
        if not all(_is_count(item.get(f)) for f in _ITERATION_COUNTS):
            return None
        optional = {f: item[f] for f in _ITERATION_OPTIONAL if item.get(f) is not None}
        if not all(_is_count(v) for v in optional.values()):
            return None
        kept.append(
            {
                **({} if kind is None else {"type": kind}),
                **{f: item[f] for f in _ITERATION_COUNTS},
                **optional,
            }
        )
    return kept


def _checked_usage(
    source: _UsageSource, raw: dict[str, pydantic.JsonValue]
) -> tuple[dict[str, pydantic.JsonValue], set[str]] | None:
    """The raw usage reduced to the fields the converter reads, each checked, and the
    written counts whose required field is invalid; None when a contributor or a total
    is invalid, so the row's counts stay unread."""
    unread = {k for k, f in source.required.items() if not _is_count(raw.get(f))}
    usage: dict[str, pydantic.JsonValue] = {
        f: value if isinstance(value := raw.get(f), int) and _is_count(value) else 0
        for f in source.required.values()
    }
    for path in source.optional:
        value = _at(raw, path)
        if value is _MISSING or value is None:
            continue
        if not _is_count(value):
            return None
        node = usage
        for part in path[:-1]:
            child = node.setdefault(part, {})
            assert isinstance(child, dict)
            node = child
        node[path[-1]] = value  # pyright: ignore[reportArgumentType]
    for total, parts in source.within:
        values = [_at(raw, p) for p in parts]
        covered = sum(v for v in values if isinstance(v, int))
        if _is_count(raw.get(total)) and covered > usage[total]:  # pyright: ignore[reportOperatorIssue]
            return None
    if source.iterations:
        iterations = _iterations(raw)
        if iterations is None:
            return None
        if iterations:
            usage["iterations"] = iterations
    return usage, unread


async def _error_usage(call: RelayCall) -> inspect_ai.model.ModelUsage | None:
    """The usage an unanswered call's response reports, normalised as an answered call's.

    Full recording keeps an error response's body, and its reader counts the usage the
    body reports; usage mode must not drop those tokens. Only the checked fields reach
    the converter: no other provider data, however large, is passed to it.
    """
    found = _usage_source(call)
    body = _USAGE_ONLY_BODIES.get(call.method)
    if found is None or body is None:
        return None
    source, raw = found
    checked = _checked_usage(source, raw)
    if checked is None:
        return None
    usage = checked[0]
    for name, default in _USAGE_SCHEMA_DEFAULTS.get(call.method, {}).items():
        given = usage.get(name)
        if isinstance(default, dict):
            usage[name] = {**default, **(given if isinstance(given, dict) else {})}
        else:
            usage.setdefault(name, default)
    try:
        output = await model_events._convert_response(call, {**body, source.key: usage})  # pyright: ignore[reportPrivateUsage]
    except (ValueError, TypeError, KeyError):
        return None
    return output.usage


def _reported_counts(
    call: RelayCall, usage: inspect_ai.model.ModelUsage | None
) -> dict[str, int | None]:
    """The four written counts: the converted split, with every count the provider did
    not validly report, and every split built on an invalid source, written as null."""
    unread: dict[str, int | None] = dict.fromkeys(_COUNT_KEYS)
    if usage is None:
        return unread
    counts: dict[str, int | None] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": usage.input_tokens_cache_write,
        "cache_read_tokens": usage.input_tokens_cache_read,
    }
    found = _usage_source(call)
    if found is None:
        if call.dialect == RequestDialect.GEMINI:
            counts["input_tokens"] = counts["output_tokens"] = None
    else:
        checked = _checked_usage(*found)
        if checked is None:
            return unread
        for key in checked[1]:
            counts[key] = None
    if not all(value is None or _is_count(value) for value in counts.values()):
        return unread
    return counts


def _refusal_code(refused: str | None) -> str | None:
    if refused is None:
        return None
    if "caller is not authenticated" in refused:
        return "unauthenticated"
    if "call cap of" in refused:
        return "call_cap"
    if "recorded-byte cap of" in refused:
        return "recorded_bytes_cap"
    if "per-phase request cap" in refused:
        return "request_too_large"
    if "model" in refused and "declared" in refused:
        return "model_not_declared"
    return "request_rejected"


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
