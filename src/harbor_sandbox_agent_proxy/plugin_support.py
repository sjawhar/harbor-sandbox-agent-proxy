"""Default collaborator implementations used by the public Harbor plugin."""

from __future__ import annotations

from collections.abc import Collection

from .outcome import EventSink, EventSinkFactory, JsonOutcomeSink, TrialIdentity


class JsonOutcomeSinkFactory(EventSinkFactory):
    """Build an independent JSON sink for every trial."""

    def __init__(self, protected_values: Collection[str] = ()) -> None:
        self._protected_values = frozenset(protected_values)

    async def open(self, trial: TrialIdentity) -> EventSink:
        return JsonOutcomeSink(trial, protected_values=self._protected_values)
