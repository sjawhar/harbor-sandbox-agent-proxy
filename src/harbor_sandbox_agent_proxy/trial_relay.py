"""Per-trial relay lifecycle with injected routing, recording, tools, and secrets."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Collection, Mapping, Sequence
from typing import Literal

import pydantic

from .collaborators import (
    BridgedMcpServer,
    DefaultSecretSource,
    DefaultUpstream,
    SecretSource,
    Upstream,
)
from .outcome import EventSink, RelayOutcome, TrialEvent, TrialIdentity
from .relay import RelayCall, RelayPhase, RelayPolicy, mint_caller_key

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
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.secrets is None:
            self.secrets = DefaultSecretSource(self.upstream.environment) if isinstance(self.upstream, DefaultUpstream) else _EmptySecrets()
        self._emit_key_minted("agent", self.agent_key)
        self._emit_key_minted("verifier", self.verifier_key)

    def protected_values(self) -> frozenset[str]:
        """Expose redaction values in memory only; sinks never receive their values."""
        assert self.secrets is not None
        return frozenset({self.agent_key, self.verifier_key, *self.secrets.protected_values()})

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
            bridged={server.name: frozenset() for server in self.bridged_servers} if self.phase == "agent" else {},
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
        await self._record_phase("agent-opened")

    async def end_agent_phase(self) -> None:
        self.phase = "verifier"
        self.verifier_key = mint_caller_key()
        await self._record_key_minted("verifier", self.verifier_key)
        await self._record_phase("verifier-opened")

    async def resume_agent_phase(self) -> None:
        self.phase = "agent"
        await self._record_phase("agent-resumed")

    async def close(self) -> RelayOutcome:
        """Close the sink once and return a redaction-safe outcome."""
        if self._closed:
            return self._outcome()
        self._closed = True
        await self._record_phase("closed")
        outcome = self._outcome()
        await self.sink.close(outcome)
        return outcome

    def _outcome(self) -> RelayOutcome:
        counts = {
            "agent_calls": sum(call.phase == "agent" for call in self.calls),
            "verifier_calls": sum(call.phase == "verifier" for call in self.calls),
            "refused_calls": sum(not call.authenticated for call in self.calls),
            "failed_calls": sum(call.status is not None and call.status >= 400 for call in self.calls),
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
            TrialEvent(kind="caller-key-minted", phase=phase, data={"fingerprint": _fingerprint(key)})
        )

    async def _record_key_minted(self, phase: RelayPhase, key: str) -> None:
        event = TrialEvent(kind="caller-key-minted", phase=phase, data={"fingerprint": _fingerprint(key)})
        self.events.append(event)
        await self.sink.record(event)

    async def _record_phase(self, kind: str) -> None:
        event = TrialEvent(kind=kind, phase=self.phase, data={})
        self.events.append(event)
        await self.sink.record(event)


class _EmptySecrets:
    def protected_values(self) -> Collection[str]:
        return ()


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
