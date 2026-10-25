"""Main-conversation selection for calls replayed outside an Inspect bridge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from inspect_ai.model import ChatMessage, ModelOutput

from ._inspect_compat import (
    Descent,
    MessageFingerprint,
    condensed_fingerprint,
    extends,
    message_fingerprint,
    position_descent,
)


class BridgeConversation(NamedTuple):
    """One call's full conversation, as tracked by ``BridgeConversationSelector``."""

    messages: list[ChatMessage]
    """The call's input messages plus its own reply, in order."""

    output: ModelOutput
    """The call's model output (its reply)."""


class BridgeConversationSelector:
    """Selects the agent's main-thread conversation among several bridged calls.

    ``AgentBridge`` uses this exact rule internally (see its ``_track_state``) to tell a
    scaffold's main agent loop apart from side calls -- a title-generation probe, a
    topic detector, a bash-path check -- made through the same model bridge. This
    class exposes the rule as a standalone, reusable tracker for a caller that
    replays bridged calls OUTSIDE a live ``AgentBridge``, for example to derive the
    main conversation from a recorded log of raw provider calls after the fact.

    Message counts alone can't do this: a side call longer than the main
    conversation would permanently displace the real one. Tracking instead follows
    thread identity:

    - A call whose messages extend the tracked thread (the tracked messages are a
      prefix of it, compared by role and text) always updates the selection.
    - Otherwise the call starts a new thread and descent from the initial input
      arbitrates (see ``_descends_from_initial``): a thread descends from the initial
      input if its non-system messages anchor on the initial input's, verbatim, as
      their condensed ``attachment://<hash>`` reference, or as decorated text
      containing it. A stronger-descending thread displaces the tracked thread when
      the tracked thread is a one-shot call or the stronger call is longer.
    - When descent can't discriminate, a length heuristic decides: adopt the new
      thread when it has more messages than the previous call (or, when both
      threads descend, than the tracked thread).
    - A new thread that isn't adopted is remembered as a candidate; if the next
      call extends the candidate, it is promoted (this recovers tracking after a
      scaffold compacts its history, replacing the conversation with a summary that
      neither extends the tracked thread nor descends from the initial input).

    Feed calls through ``observe()`` in the order they were made; ``selected`` holds
    the current main-thread verdict, or ``None`` before the first call.
    """

    def __init__(self, initial_input: Sequence[ChatMessage] | None = None) -> None:
        """Create a selector.

        Args:
          initial_input: The scaffold's original, pre-conversation prompt, for
            descent anchoring (see ``_descends_from_initial``). Pass ``None`` when it
            is unknown or unstable; descent then can't discriminate and selection
            falls back entirely to the length heuristic, as ``AgentBridge`` itself
            does before an initial prompt is recorded.
        """
        initial = [
            message for message in (initial_input or []) if message.role != "system"
        ]
        self._initial_fps: list[MessageFingerprint] = [
            message_fingerprint(message) for message in initial
        ]
        self._initial_fps_condensed: list[MessageFingerprint] = [
            condensed_fingerprint(fingerprint) for fingerprint in self._initial_fps
        ]
        self._initial_texts: list[str] = [message.text.strip() for message in initial]
        self._tracked_fps: list[MessageFingerprint] | None = None
        self._tracked_calls: int = 0
        self._tracked_descends: Any | None = None
        self._candidate_fps: list[MessageFingerprint] | None = None
        self._last_message_count: int = 0
        self._selected: BridgeConversation | None = None

    def observe(
        self, input: Sequence[ChatMessage], output: ModelOutput
    ) -> BridgeConversation | None:
        """Observe one call, in the order it was made.

        Args:
          input: The call's input messages.
          output: The call's model output.

        Returns:
          ``selected`` after this call (for convenience; also available as a property).
        """
        messages = list(input) + [output.message]
        fingerprints = [message_fingerprint(message) for message in messages]
        if self._tracked_fps is None:
            self._adopt(messages, output, fingerprints, calls=1)
        elif extends(self._tracked_fps, fingerprints):
            self._adopt(messages, output, fingerprints, calls=self._tracked_calls + 1)
        elif self._candidate_fps is not None and extends(
            self._candidate_fps, fingerprints
        ):
            self._adopt(messages, output, fingerprints, calls=2)
        else:
            descends = self._descends_from_initial(messages, fingerprints)
            if (
                descends is not None
                and self._tracked_descends is not None
                and descends > self._tracked_descends
                and (
                    self._tracked_calls == 1
                    or len(messages) > len(self._tracked_fps or [])
                )
            ):
                self._adopt(messages, output, fingerprints, calls=1)
            elif descends == self._tracked_descends and len(messages) > (
                len(self._tracked_fps or []) if descends else self._last_message_count
            ):
                self._adopt(messages, output, fingerprints, calls=1)
            else:
                self._candidate_fps = fingerprints
        self._last_message_count = len(messages)
        return self._selected

    @property
    def selected(self) -> BridgeConversation | None:
        """The currently tracked main-thread conversation, or ``None`` before any call."""
        return self._selected

    def _adopt(
        self,
        messages: list[ChatMessage],
        output: ModelOutput,
        fingerprints: list[MessageFingerprint],
        *,
        calls: int,
    ) -> None:
        self._selected = BridgeConversation(messages=messages, output=output)
        self._tracked_fps = fingerprints
        self._tracked_calls = calls
        self._tracked_descends = self._descends_from_initial(messages, fingerprints)
        self._candidate_fps = None

    def _descends_from_initial(
        self,
        messages: list[ChatMessage],
        fingerprints: list[MessageFingerprint],
    ) -> Any | None:
        """How a thread's non-system messages anchor on the initial input.

        See ``AgentBridge._descends_from_initial`` for the full rationale; this is
        the same grading rule applied to this selector's own initial-input state.
        """
        if not self._initial_fps:
            return None
        non_system = [
            (message, fingerprint)
            for message, fingerprint in zip(messages, fingerprints, strict=True)
            if fingerprint.role != "system"
        ]
        if len(non_system) < len(self._initial_fps):
            return Descent.NO
        quoted = False
        contained = False
        for (message, fingerprint), initial, condensed, initial_text in zip(
            non_system,
            self._initial_fps,
            self._initial_fps_condensed,
            self._initial_texts,
            strict=False,
        ):
            position = position_descent(
                message, fingerprint, initial, condensed, initial_text
            )
            if position is Descent.NO:
                return Descent.NO
            quoted = quoted or position is Descent.QUOTED
            contained = contained or position is Descent.CONTAINED
        if contained:
            return Descent.CONTAINED
        return Descent.QUOTED if quoted else Descent.EXACT
