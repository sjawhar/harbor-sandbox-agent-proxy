"""A bridged task tool is served only during the agent's turn."""

from __future__ import annotations

from collections.abc import Callable

import inspect_ai.tool
import pytest

from harbor_sandbox_agent_proxy.outcome import TrialEvent
from harbor_sandbox_agent_proxy.relay import RelayPhase
from harbor_sandbox_agent_proxy.trial_relay import phase_guarded_tool

_SERVER = "task-tools"
_TOOL = "ping"


@inspect_ai.tool.tool
def ping(ran: list[str]) -> inspect_ai.tool.Tool:
    async def execute(text: str) -> str:
        """Echo a task-tool request.

        Args:
            text: The text to echo back.
        """
        ran.append(text)
        return f"pong: {text}"

    return execute


def _guard(
    phase: Callable[[], RelayPhase], events: list[TrialEvent], ran: list[str]
) -> inspect_ai.tool.Tool:
    async def refused(event: TrialEvent) -> None:
        events.append(event)

    return phase_guarded_tool(ping(ran), server=_SERVER, phase=phase, refused=refused)


async def test_a_claude_code_trials_granted_call_is_refused_outside_the_agents_turn() -> (
    None
):
    """A verifier tool call runs nothing and reaches the trial event sink."""
    phase: RelayPhase = "verifier"
    events: list[TrialEvent] = []
    ran: list[str] = []
    guarded = _guard(lambda: phase, events, ran)

    with pytest.raises(inspect_ai.tool.ToolError, match="relay phase: verifier"):
        await guarded(text="hello")

    assert ran == []
    assert events == [
        TrialEvent(
            kind="refused-tool-call",
            phase="verifier",
            data={
                "server": _SERVER,
                "tool": _TOOL,
                "arguments": {"text": "hello"},
                "reason": "outside the agent's turn",
            },
        )
    ]


async def test_a_codex_trials_bare_call_is_served_in_the_agents_turn_and_refused_after() -> (
    None
):
    phase: RelayPhase = "agent"
    events: list[TrialEvent] = []
    ran: list[str] = []
    guarded = _guard(lambda: phase, events, ran)

    assert await guarded(text="hello") == "pong: hello"
    phase = "verifier"
    with pytest.raises(inspect_ai.tool.ToolError, match="relay phase: verifier"):
        await guarded(text="hello")

    assert ran == ["hello"]
    assert len(events) == 1


async def test_a_later_steps_agent_turn_is_served_between_two_refusals() -> None:
    phase: RelayPhase = "verifier"
    events: list[TrialEvent] = []
    ran: list[str] = []
    guarded = _guard(lambda: phase, events, ran)

    with pytest.raises(inspect_ai.tool.ToolError):
        await guarded(text="one")
    phase = "agent"
    assert await guarded(text="two") == "pong: two"
    phase = "verifier"
    with pytest.raises(inspect_ai.tool.ToolError):
        await guarded(text="three")

    assert ran == ["two"]
    assert [event.phase for event in events] == ["verifier", "verifier"]


async def test_the_judges_model_calls_are_answered_while_its_tool_calls_are_refused() -> (
    None
):
    """The guard only rejects bridged task tools; it does not block model forwarding."""
    phase: RelayPhase = "verifier"
    events: list[TrialEvent] = []
    ran: list[str] = []
    guarded = _guard(lambda: phase, events, ran)

    with pytest.raises(inspect_ai.tool.ToolError):
        await guarded(text="judge request")

    assert ran == []
    assert events[0].kind == "refused-tool-call"
    assert events[0].phase == "verifier"


async def test_the_guard_changes_nothing_the_bridge_serves_for_the_tool() -> None:
    ran: list[str] = []

    async def never_refused(event: TrialEvent) -> None:
        raise AssertionError(f"agent-phase tool call was refused: {event!r}")

    inner = ping(ran)
    guarded = phase_guarded_tool(
        inner, server=_SERVER, phase=lambda: "agent", refused=never_refused
    )
    expected = inspect_ai.tool.ToolDef(inner)
    actual = inspect_ai.tool.ToolDef(guarded)

    assert (
        actual.name,
        actual.description,
        actual.parameters.model_dump(),
        actual.parallel,
        actual.max_output,
        actual.options,
    ) == (
        expected.name,
        expected.description,
        expected.parameters.model_dump(),
        expected.parallel,
        expected.max_output,
        expected.options,
    )
    assert actual.viewer is expected.viewer
    assert actual.model_input is expected.model_input
