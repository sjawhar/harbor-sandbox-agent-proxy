"""Per-trial relay lifecycle with injected routing, recording, tools, and secrets."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
from collections.abc import Collection, Mapping, Sequence

import httpx
from harbor.models.task.config import MCPServerConfig
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import (
    Tool,
    ToolDef,
    ToolError,
    mcp_connection,
    mcp_server_sandbox,
    mcp_tools,
)

from .collaborators import (
    BridgedMcpServer,
    DefaultSecretSource,
    DefaultUpstream,
    SecretSource,
    Upstream,
)
from .outcome import EventSink, RelayOutcome, TrialEvent, TrialIdentity
from .relay import (
    GATEWAY_TIMEOUT,
    RelayCall,
    RelayPhase,
    RelayPolicy,
    mint_caller_key,
    relay_methods,
)
from .sandbox_adapter import HarborEnvironment


_AGENT_ENVIRONMENT: Mapping[str, tuple[str, str]] = {
    "messages": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"),
    "chat-completions": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
    "responses": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
    "gemini": ("GOOGLE_GEMINI_BASE_URL", "GEMINI_API_KEY"),
}


@dataclasses.dataclass
class TrialRelay:
    """Own the mutable state of exactly one relay lifecycle."""

    trial: TrialIdentity
    upstream: Upstream
    sink: EventSink
    declared_models: frozenset[str]
    bridged_servers: Sequence[BridgedMcpServer] = ()
    secrets: SecretSource | None = None
    port: int = 13131
    phase: RelayPhase = "agent"
    agent_key: str = dataclasses.field(default_factory=mint_caller_key)
    verifier_key: str = dataclasses.field(default_factory=mint_caller_key)
    calls: list[RelayCall] = dataclasses.field(default_factory=list)
    events: list[TrialEvent] = dataclasses.field(default_factory=list)
    _mcp_server_configs: list[MCPServerConfig] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )
    _closed: bool = False
    environment: HarborEnvironment | None = None
    polling_interval: float | None = None
    _proxy_task: asyncio.Task[None] | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _proxy_stop: asyncio.Event | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.secrets is None:
            self.secrets = (
                DefaultSecretSource(self.upstream.environment)
                if isinstance(self.upstream, DefaultUpstream)
                else _EmptySecrets()
            )
        self._emit_key_minted("agent", self.agent_key)
        self._emit_key_minted("verifier", self.verifier_key)

    def protected_values(self) -> frozenset[str]:
        """Expose redaction values in memory only; sinks never receive their values."""
        assert self.secrets is not None
        return frozenset(
            {self.agent_key, self.verifier_key, *self.secrets.protected_values()}
        )

    def phase_env(self) -> dict[str, str]:
        """Return only loopback endpoints and the active sandbox caller key."""
        key = self.agent_key if self.phase == "agent" else self.verifier_key
        base = f"http://localhost:{self.port}"
        environment: dict[str, str] = {}
        for name, (base_name, key_name) in _AGENT_ENVIRONMENT.items():
            suffix = "/v1" if name in {"chat-completions", "responses"} else ""
            environment[base_name] = base + suffix
            environment[key_name] = key
        environment["ANTHROPIC_AUTH_TOKEN"] = key
        environment["OPENAI_API_BASE"] = base + "/v1"
        environment["GOOGLE_API_KEY"] = key
        environment["GOOGLE_GENERATIVE_AI_API_KEY"] = key
        return environment

    def current_policy(self) -> tuple[RelayPhase, RelayPolicy]:
        return self.phase, RelayPolicy(
            caller_key=self.agent_key if self.phase == "agent" else self.verifier_key,
            allowed_models=self.declared_models,
            bridged={server.name: frozenset() for server in self.bridged_servers}
            if self.phase == "agent"
            else {},
        )

    async def record(self, call: RelayCall) -> None:
        self.calls.append(call)
        event = TrialEvent(
            kind="relayed-call" if call.authenticated else "refused-call",
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

    async def open_agent_phase(self) -> None:
        if self._proxy_task is None and self.environment is not None:
            ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._proxy_stop = asyncio.Event()
            self._proxy_task = asyncio.create_task(
                self._hold_proxy(ready, self._proxy_stop)
            )
            await ready
        await self._record_phase("agent-opened")

    def mcp_server_configs(self) -> list[MCPServerConfig]:
        """Return the loopback MCP endpoints created with the proxy."""
        return list(self._mcp_server_configs)

    async def end_agent_phase(self) -> None:
        self.phase = "verifier"
        self.verifier_key = mint_caller_key()
        await self._record_key_minted("verifier", self.verifier_key)
        await self._record_phase("verifier-opened")

    async def resume_agent_phase(self) -> None:
        self.phase = "agent"
        await self._record_phase("agent-resumed")

    async def close(self) -> RelayOutcome:
        """Close the proxy and sink once, returning a redaction-safe outcome."""
        if self._closed:
            return self._outcome()
        self._closed = True
        if self._proxy_stop is not None:
            self._proxy_stop.set()
        if self._proxy_task is not None:
            await self._proxy_task
        await self._record_phase("closed")
        outcome = self._outcome()
        await self.sink.close(outcome)
        return outcome

    def _outcome(self) -> RelayOutcome:
        counts = {
            "agent_calls": sum(call.phase == "agent" for call in self.calls),
            "verifier_calls": sum(call.phase == "verifier" for call in self.calls),
            "refused_calls": sum(not call.authenticated for call in self.calls),
            "failed_calls": sum(
                call.status is not None and call.status >= 400 for call in self.calls
            ),
        }
        serialised_calls = tuple(
            {
                "phase": call.phase,
                "method": call.method,
                "model": call.model,
                "status": call.status,
                "refused": call.refused,
                "stripped": list(call.stripped),
            }
            for call in self.calls
        )
        return RelayOutcome(
            trial=self.trial,
            calls=serialised_calls,
            events=tuple(self.events),
            counts=counts,
            conversation=None,
            refusal_reason=None,
        )

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

    async def _hold_proxy(
        self, ready: asyncio.Future[None], stop: asyncio.Event
    ) -> None:
        from .proxy import sandbox_model_proxy
        from .sandbox_adapter import HarborSandboxEnvironment
        from .sandbox_binding import bind_sandbox_environment

        if self.environment is None:
            ready.set_result(None)
            return
        try:
            sandbox = HarborSandboxEnvironment(
                self.environment,
                default_user="root",
                polling_interval=self.polling_interval,
            )
            with bind_sandbox_environment(sandbox) as bound_sandbox:
                async with contextlib.AsyncExitStack() as stack:
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(timeout=GATEWAY_TIMEOUT)
                    )
                    bridged_tools = await self._create_bridged_tools(stack)
                    proxy = await stack.enter_async_context(
                        sandbox_model_proxy(
                            bound_sandbox,
                            methods=relay_methods(
                                self.current_policy,
                                self.upstream,
                                record=self.record,
                                client=client,
                            ),
                            port=self.port,
                            bridged_tools=bridged_tools,
                            polling_interval=self.polling_interval,
                        )
                    )
                    self.port = proxy.port
                    self._mcp_server_configs = [
                        MCPServerConfig(
                            name=config.name,
                            transport="streamable-http",
                            url=str(config.url),
                        )
                        for config in proxy.mcp_server_configs
                    ]
                    ready.set_result(None)
                    await stop.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
                return
            raise

    async def _create_bridged_tools(
        self, stack: contextlib.AsyncExitStack
    ) -> list[BridgedToolsSpec]:
        specs: list[BridgedToolsSpec] = []
        for server in self.bridged_servers:
            source = mcp_tools(
                mcp_server_sandbox(
                    name=server.name,
                    command=server.command,
                    args=list(server.args),
                )
            )
            await stack.enter_async_context(mcp_connection(source))
            tools = await source.tools()
            specs.append(
                BridgedToolsSpec(
                    name=server.name,
                    tools=[
                        self._phase_guarded_tool(tool, server.name)
                        for tool in tools
                        if ToolDef(tool).name not in server.exclude_tools
                    ],
                    require_proposal=server.require_proposal,
                )
            )
        return specs

    def _phase_guarded_tool(self, tool: Tool, server: str) -> Tool:
        inner = ToolDef(tool)

        async def guarded(**arguments: object) -> object:
            if self.phase == "agent":
                return await tool(**arguments)
            await self._record_refused_tool_call(server, inner.name)
            raise ToolError(
                f"{server}/{inner.name} is not served outside the agent phase"
            )

        return ToolDef(
            guarded,
            name=inner.name,
            description=inner.description,
            parameters=inner.parameters,
            parallel=inner.parallel,
            viewer=inner.viewer,
            model_input=inner.model_input,
            max_output=inner.max_output,
            options=inner.options,
        ).as_tool()

    async def _record_refused_tool_call(self, server: str, tool: str) -> None:
        event = TrialEvent(
            kind="refused-tool-call",
            phase=self.phase,
            data={"server": server, "tool": tool, "reason": "outside-agent-phase"},
        )
        self.events.append(event)
        await self.sink.record(event)


class _EmptySecrets:
    def protected_values(self) -> Collection[str]:
        return ()


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
