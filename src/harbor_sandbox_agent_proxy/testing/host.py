"""A RelayEnvironment host used by the Docker end-to-end proof."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import override

import inspect_ai.event

from ..environment import RelayEnvironment
from ..outcome import JsonOutcomeSink, RelayOutcome, TrialIdentity
from ..routing import ModelCall, Route, auth_headers, match_declared
from ..task_metadata import BridgedMcpServer
from .agents import HOST_KEY

_UPSTREAM_URL_ENV = "SANDBOX_AGENT_PROXY_E2E_UPSTREAM_URL"
_UPSTREAM_CAPTURE_ENV = "SANDBOX_AGENT_PROXY_E2E_UPSTREAM_CAPTURE"


@dataclasses.dataclass
class _E2EEventSink:
    """Write live host events and the package outcome for one trial."""

    trial: TrialIdentity
    outcome_sink: JsonOutcomeSink

    async def record(self, event: inspect_ai.event.Event) -> None:
        destination = self.trial.directory / "e2e-host" / "events.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(event.model_dump(mode="json")) + "\n")

    async def close(self, outcome: RelayOutcome) -> None:
        await self.outcome_sink.close(outcome)


class E2EHostEnvironment(RelayEnvironment):
    """Route E2E calls through the in-process upstream server."""

    @override
    async def route(self, call: ModelCall) -> Route:
        route = Route(
            os.environ[_UPSTREAM_URL_ENV],
            auth_headers(call.api, HOST_KEY),
            match_declared(call),
        )
        capture = Path(os.environ[_UPSTREAM_CAPTURE_ENV])
        with capture.open("a", encoding="utf-8") as destination:
            _ = destination.write(
                json.dumps({"dialect": call.api, "headers": route.headers}) + "\n"
            )
        return route

    @override
    def protected_values(self) -> Collection[str]:
        return frozenset({HOST_KEY})

    @override
    async def open_event_sink(self, trial: TrialIdentity) -> _E2EEventSink:
        return _E2EEventSink(
            trial,
            JsonOutcomeSink(
                trial,
                protected=lambda: {*self.protected_values(), *self.caller_keys},
            ),
        )

    @override
    async def bridged_tool_servers(self) -> Sequence[BridgedMcpServer]:
        return (
            BridgedMcpServer(
                name="e2e-tools",
                command="python3",
                args=["/opt/e2e_tools.py"],
                require_proposal=False,
            ),
        )
