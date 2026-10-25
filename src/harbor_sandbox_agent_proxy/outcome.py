"""Portable per-trial records and the default redacted JSON outcome sink."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import pydantic


@dataclasses.dataclass(frozen=True)
class TrialIdentity:
    """The trial identity known to package collaborators."""

    name: str
    directory: Path


@dataclasses.dataclass(frozen=True)
class TrialEvent:
    """A package-owned observation emitted during a trial lifecycle."""

    kind: str
    phase: str
    data: Mapping[str, pydantic.JsonValue]


@dataclasses.dataclass(frozen=True)
class RelayOutcome:
    """The redaction-safe terminal facts of a relay lifecycle."""

    trial: TrialIdentity
    calls: tuple[Mapping[str, object], ...]
    events: tuple[TrialEvent, ...]
    counts: Mapping[str, int]
    conversation: Mapping[str, object] | None
    refusal_reason: str | None


class EventSink(Protocol):
    """Receive package records for one trial without observing credential values."""

    async def record(self, event: TrialEvent) -> None: ...

    async def close(self, outcome: RelayOutcome) -> None: ...


@runtime_checkable
class EventSinkFactory(Protocol):
    """Open a fresh event sink for each trial."""

    async def open(self, trial: TrialIdentity) -> EventSink: ...


@dataclasses.dataclass
class JsonOutcomeSink:
    """Write a redacted, self-contained default outcome beneath the trial directory."""

    trial: TrialIdentity
    protected_values: Collection[str] = ()
    _events: list[TrialEvent] = dataclasses.field(default_factory=list)

    async def record(self, event: TrialEvent) -> None:
        self._events.append(event)

    async def close(self, outcome: RelayOutcome) -> None:
        destination = self.trial.directory / "sandbox-agent-proxy" / "outcome.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "trial": outcome.trial.name,
            "calls": [dict(call) for call in outcome.calls],
            "events": [
                {"kind": event.kind, "phase": event.phase, "data": dict(event.data)}
                for event in outcome.events
            ],
            "counts": dict(outcome.counts),
            "conversation": outcome.conversation,
            "refusal_reason": outcome.refusal_reason,
        }
        destination.write_text(
            json.dumps(_redact(payload, self.protected_values), sort_keys=True),
            encoding="utf-8",
        )


def _redact(value: object, secrets: Collection[str]) -> object:
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return result
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    return value
