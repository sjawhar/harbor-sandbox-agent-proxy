"""Tests for ``BridgeConversationSelector``."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)

from harbor_sandbox_agent_proxy.conversation import BridgeConversationSelector

TASK = "In the year 2022, what castle did the Doctor spend 4.5 billion years in?"

TASK_SYSTEM = ChatMessageSystem(content="You are opencode, an agent that ...")


def output(completion: str) -> ModelOutput:
    return ModelOutput.from_content(model="mockllm/model", content=completion)


def title_generation_input() -> list[ChatMessage]:
    # mirrors opencode's session title generation request: its own system
    # prompt, a "Generate a title" preamble, then the first user message
    return [
        ChatMessageSystem(content="You are a title generator ..."),
        ChatMessageUser(content="Generate a title for this conversation:\n"),
        ChatMessageUser(content=TASK),
    ]


def task_selector() -> BridgeConversationSelector:
    return BridgeConversationSelector([ChatMessageUser(content=TASK)])


def test_side_call_arriving_first_does_not_displace_task_thread() -> None:
    """A longer one-shot side call landing first must not win (meridianlabs-ai/inspect_ai#140)."""
    selector = task_selector()

    selector.observe(title_generation_input(), output("Doctor Who Series 9 setting"))
    selector.observe(
        [TASK_SYSTEM, ChatMessageUser(content=TASK)],
        output("Castle"),
    )
    # a subsequent same-length one-shot task call must not displace the answer
    selector.observe(
        [TASK_SYSTEM, ChatMessageUser(content=TASK)],
        output("Castle TARDIS Console Room"),
    )

    assert selector.selected is not None
    assert selector.selected.output.completion == "Castle"
    assert [message.text for message in selector.selected.messages] == [
        TASK_SYSTEM.text,
        TASK,
        "Castle",
    ]


def test_main_loop_accumulation_is_tracked() -> None:
    selector = task_selector()

    turn1: list[ChatMessage] = [TASK_SYSTEM, ChatMessageUser(content=TASK)]
    out1 = output("checking")
    selected = selector.observe(turn1, out1)
    assert selected is not None and selected.output.completion == "checking"

    turn2 = turn1 + [out1.message, ChatMessageTool(content="tool result")]
    out2 = output("still checking")
    selected = selector.observe(turn2, out2)
    assert selected is not None and selected.output.completion == "still checking"

    turn3 = turn2 + [out2.message, ChatMessageTool(content="tool result 2")]
    selected = selector.observe(turn3, output("Castle"))
    assert selected is not None
    assert selected.output.completion == "Castle"
    assert len(selected.messages) == len(turn3) + 1


def test_shorter_side_call_is_ignored() -> None:
    # e.g. claude code's bash path detection side call
    selector = task_selector()

    turn1: list[ChatMessage] = [TASK_SYSTEM, ChatMessageUser(content=TASK)]
    out1 = output("working")
    selector.observe(turn1, out1)
    turn2 = turn1 + [out1.message, ChatMessageTool(content="tool result")]
    out2 = output("more work")
    selector.observe(turn2, out2)

    selected = selector.observe(
        [ChatMessageUser(content="Detect the paths in this bash command: ls /tmp")],
        output("/tmp"),
    )
    assert selected is not None and selected.output.completion == "more work"

    # main loop continues to be tracked afterwards
    turn3 = turn2 + [out2.message, ChatMessageTool(content="tool result 2")]
    selected = selector.observe(turn3, output("Castle"))
    assert selected is not None and selected.output.completion == "Castle"


def test_scaffold_compaction_recovery() -> None:
    """After the scaffold compacts its history the new (shorter) loop wins."""
    selector = task_selector()

    turn1: list[ChatMessage] = [TASK_SYSTEM, ChatMessageUser(content=TASK)]
    out1 = output("working")
    selector.observe(turn1, out1)
    turn2 = turn1 + [out1.message, ChatMessageTool(content="tool result")]
    out2 = output("more work")
    selector.observe(turn2, out2)
    turn3 = turn2 + [out2.message, ChatMessageTool(content="tool result 2")]
    selector.observe(turn3, output("even more work"))

    # compaction: history replaced by a summary (no longer shares the
    # original input prefix), then the loop keeps appending
    compact1: list[ChatMessage] = [
        TASK_SYSTEM,
        ChatMessageUser(content="Summary of the conversation so far: ..."),
    ]
    cout1 = output("compacted work")
    selector.observe(compact1, cout1)
    compact2 = compact1 + [cout1.message, ChatMessageTool(content="tool result 3")]
    selected = selector.observe(compact2, output("Castle"))

    assert selected is not None
    assert selected.output.completion == "Castle"
    assert len(selected.messages) == len(compact2) + 1


def test_length_heuristic_fallback_without_initial_input() -> None:
    # with no initial input to anchor descent, accumulation still tracks
    selector = BridgeConversationSelector(None)

    turn1: list[ChatMessage] = [TASK_SYSTEM, ChatMessageUser(content=TASK)]
    out1 = output("working")
    selector.observe(turn1, out1)
    turn2 = turn1 + [out1.message, ChatMessageTool(content="tool result")]
    selected = selector.observe(turn2, output("Castle"))
    assert selected is not None and selected.output.completion == "Castle"

    # shorter side call ignored
    selected = selector.observe(
        [ChatMessageUser(content="side call")], output("side answer")
    )
    assert selected is not None and selected.output.completion == "Castle"


def test_quote_wrapped_prompt_anchors_descent() -> None:
    """Opencode round-trips the prompt wrapped in literal double quotes."""
    selector = task_selector()
    quoted = f'"{TASK}"'

    selected = selector.observe(
        [TASK_SYSTEM, ChatMessageUser(content=quoted)], output("Castle")
    )
    assert selected is not None and selected.output.completion == "Castle"

    # a later verbatim (unquoted) side call must not beat the quoted main call
    selected = selector.observe(
        [ChatMessageSystem(content="..."), ChatMessageUser(content=TASK)],
        output("Doctor Who Series 9 setting"),
    )
    assert selected is not None and selected.output.completion == "Castle"


def test_selector_returns_none_before_first_call() -> None:
    assert task_selector().selected is None
