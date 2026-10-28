"""Portable relay records and the default redacted JSON event sink."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import inspect_ai.event

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

    from .relay import RelayCall


@dataclasses.dataclass(frozen=True)
class TrialIdentity:
    """The trial identity known to package collaborators."""

    name: str
    directory: Path


@dataclasses.dataclass(frozen=True)
class RelayOutcome:
    """The complete result of one relay or runner-loop lifecycle.

    Relay mode owns the recorded sandbox proxy calls and selected conversation. A
    runner-loop agent has no sandbox proxy; its host-side trajectory is retained
    separately, before Harbor synchronizes the trial directory.
    """

    mode: Literal["relay", "runner-loop"]
    calls: list[RelayCall]
    events: list[inspect_ai.event.Event]
    conversation: list[ChatMessage] | None
    answered_agent_calls: int
    counts: Mapping[str, int]
    dropped_betas: frozenset[str]
    refusal_reason: str | None
    proxy_exit: Mapping[str, object] | None
    runner_atif: Path | None
    runner_atif_sha256: str | None
    record_error: str | None


class EventSink(Protocol):
    """Receive the Inspect events and completed outcome for one trial."""

    async def record(self, event: inspect_ai.event.Event) -> None: ...

    async def close(self, outcome: RelayOutcome) -> None: ...


@dataclasses.dataclass
class JsonOutcomeSink:
    """Write a redacted, self-contained package outcome beneath a trial directory."""

    trial: TrialIdentity
    protected: Callable[[], Collection[str]]

    async def record(self, event: inspect_ai.event.Event) -> None:
        """Accept a live event; the completed outcome supplies the persisted event list."""
        del event

    async def close(self, outcome: RelayOutcome) -> None:
        destination = self.trial.directory / "sandbox-agent-proxy" / "outcome.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "trial": self.trial.name,
            "mode": outcome.mode,
            "calls": [call.model_dump(mode="json") for call in outcome.calls],
            "events": [event.model_dump(mode="json") for event in outcome.events],
            "counts": dict(outcome.counts),
            "conversation": (
                None
                if outcome.conversation is None
                else [
                    message.model_dump(mode="json") for message in outcome.conversation
                ]
            ),
            "answered_agent_calls": outcome.answered_agent_calls,
            "dropped_betas": sorted(outcome.dropped_betas),
            "refusal_reason": outcome.refusal_reason,
            "proxy_exit": outcome.proxy_exit,
            "runner_atif": (
                None if outcome.runner_atif is None else str(outcome.runner_atif)
            ),
            "runner_atif_sha256": outcome.runner_atif_sha256,
            "record_error": outcome.record_error,
        }
        destination.write_text(
            json.dumps(_redact(payload, self.protected()), sort_keys=True),
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
