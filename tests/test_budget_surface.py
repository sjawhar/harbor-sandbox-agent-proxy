"""Provider envelope, payload and raw-usage boundary controls."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from .test_budget import api, body, budget, settings


@pytest.mark.parametrize("number,dialect,response,want", [
    (20, "anthropic", {"usage": {"input_tokens": 10, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 30, "output_tokens": 4}}, (60, 4)),
    (22, "anthropic", {"usage": {"input_tokens": 10, "output_tokens": 40, "output_tokens_details": {"thinking_tokens": 30}}}, (10, 40)),
    (23, "openai", {"usage": {"prompt_tokens": 10, "completion_tokens": 80, "prompt_tokens_details": {"cached_tokens": 5}}}, (10, 80)),
    (25, "openai-responses", {"status": "incomplete", "usage": {"input_tokens": 10, "output_tokens": 20}}, (10, 20)),
    (27, "google", {"candidates": [{"index": 0, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "thoughtsTokenCount": 5, "totalTokenCount": 40}}, (10, 30)),
    (28, "google", {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 10}}, (10, 0)),
])
def test_raw_provider_counts(number, dialect, response, want):
    meter = api().UsageMeter(dialect)
    meter.observe(response)
    m = meter.measurement()
    assert (m.n, m.o) == want
    assert m.terminal


@pytest.mark.parametrize("number,events,terminal", [
    (20, [{"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0}}}, {"type": "message_delta", "usage": {"output_tokens": 4}}, {"type": "message_stop"}], True),
    (30, [{"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0}}}, {"type": "error"}], False),
    (34, [{"type": "message_delta", "usage": {"output_tokens": 4}}], False),
])
def test_stream_usage_survives_assembly_failure(number, events, terminal):
    meter = api().UsageMeter("anthropic", streaming=True)
    for event in events:
        meter.feed(event)
    m = meter.measurement()
    assert m.terminal is terminal
    if number == 30:
        assert m.n == 10 and m.o is None


@pytest.mark.parametrize("number,value", [(27, True), (27, -1), (27, 1.2), (27, "3")])
def test_malformed_optional_counts_are_unknown(number, value):
    meter = api().UsageMeter("google")
    meter.observe({"candidates": [{"index": 0, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "thoughtsTokenCount": value}})
    assert meter.measurement().o is None


@pytest.mark.parametrize("number,left,right,equal", [
    (76, {"caller": "A"}, {"caller": "B"}, False),
    (76, {"cache_control": "A"}, {"cache_control": "B"}, False),
    (76, {"x": None}, {}, False),
    (76, {"a": 1, "b": 2}, {"b": 2, "a": 1}, False),
])
def test_76_payload_is_not_normalized(tmp_path, number, left, right, equal):
    b = budget(tmp_path)
    def request(value):
        return body(messages=[{"role": "assistant", "content": [{"type": "tool_use", "id": "id", "name": "f", "input": value}]}])
    a = b.prepare("anthropic", "m", request(left))
    c = b.prepare("anthropic", "m", request(right))
    assert (a.msg_hash == c.msg_hash) is equal
    b.release_writer()


def test_76_only_scoped_normalizations(tmp_path):
    b = budget(tmp_path)
    a = b.prepare("anthropic", "m", body("hello"))
    c = b.prepare("anthropic", "m", body(messages=[{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}]))
    assert a.msg_hash == c.msg_hash
    with pytest.raises(api().BudgetRefused):
        b.prepare("anthropic", "m", body(unknown_envelope=True))
    b.release_writer()


@pytest.mark.parametrize("number,field,value", [(78, "previous_response_id", "parent"), (78, "conversation", "id"), (78, "store", True), (29, "context_management", {})])
def test_server_state_refused(tmp_path, number, field, value):
    b = budget(tmp_path)
    request = {"model": "m", "input": "hi", "max_output_tokens": 10, field: value}
    with pytest.raises(api().BudgetRefused):
        b.prepare("openai-responses", "m", request)
    b.release_writer()


@pytest.mark.parametrize("number,dialect,payload", [
    (43, "anthropic", body(messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}]}])),
    (44, "openai", body(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])),
    (77, "google", {"contents": [{"role": "user", "parts": [{"fileData": {"fileUri": "x"}}]}], "generationConfig": {"maxOutputTokens": 10}}),
])
def test_modalities_refused(tmp_path, number, dialect, payload):
    b = budget(tmp_path)
    with pytest.raises(api().BudgetRefused):
        b.prepare(dialect, "m", payload)
    b.release_writer()


@pytest.mark.parametrize("fixture", ["codex_responses_code_mode.json", "codex_responses_tool_search.json"])
def test_77_complete_codex_control_workflows(tmp_path, fixture):
    data = json.loads((Path(__file__).parent / "fixtures" / fixture).read_text())
    request = data.get("body", data.get("request", data))
    b = budget(tmp_path, inject_output_limit=True)
    p = b.prepare("openai-responses", "m", request)
    assert p.reserved > 0
    b.release_writer()


def test_21_87_iterations_are_outside_enabled_contract():
    meter = api().UsageMeter("anthropic")
    meter.observe({"usage": {"input_tokens": 10, "output_tokens": 2, "iterations": [{"input_tokens": 10, "output_tokens": 2}, {"input_tokens": 20, "output_tokens": 3}]}})
    m = meter.measurement()
    assert (m.n, m.o, m.w) == (10, 5, 20)
    assert m.outside_contract


def test_28_outside_thinking_is_per_candidate(tmp_path):
    cfg = settings(thinking_ceiling={"m": 2000}, max_branches={"m": 8})
    cfg.profiles["m"].outside_output_thinking = True
    b = api().TrialBudget(cfg, tmp_path, "t")
    p = b.prepare("google", "m", {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"candidateCount": 8, "maxOutputTokens": 1000, "thinkingConfig": {"thinkingBudget": 2000}}})
    assert p.r_out == 24_000
    b.release_writer()


def test_77_opaque_reasoning_needs_independent_bound(tmp_path):
    b = budget(tmp_path)
    request = {"input": [{"type": "reasoning", "encrypted_content": "short"}], "max_output_tokens": 10}
    with pytest.raises(api().BudgetRefused):
        b.prepare("openai-responses", "m", request)
    b.config.profiles["m"].effective_input_tokens_max = 200_000
    p = b.prepare("openai-responses", "m", request)
    assert p.r_in >= 200_000
    b.release_writer()


@pytest.mark.parametrize("dialect,path", [
    ("anthropic", ("max_tokens",)),
    ("openai", ("max_tokens",)),
    ("openai", ("max_completion_tokens",)),
    ("openai", ("n",)),
    ("openai-responses", ("max_output_tokens",)),
    ("google", ("generationConfig", "maxOutputTokens")),
    ("google", ("generationConfig", "candidateCount")),
    ("anthropic", ("thinking", "budget_tokens")),
    ("google", ("generationConfig", "thinkingConfig", "thinkingBudget")),
])
@pytest.mark.parametrize("value", [-10, True, 1.5, "10", 2**63, 100001])
def test_73_every_nested_spelling_is_strict(tmp_path, dialect, path, value):
    b = budget(tmp_path, thinking_ceiling={"m": 100})
    if dialect == "google":
        payload = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {"maxOutputTokens": 1000}}
    elif dialect == "openai-responses":
        payload = {"input": "hello", "max_output_tokens": 1000}
    else:
        payload = body(max_tokens=1000)
        if path == ("max_completion_tokens",):
            payload.pop("max_tokens")
    target = payload
    for key in path[:-1]:
        target = target.setdefault(key, {})
    if path[0] == "thinking":
        payload["thinking"]["type"] = "enabled"
    target[path[-1]] = value
    before = b.journal_path.read_bytes()
    with pytest.raises(api().BudgetRefused):
        b.prepare(dialect, "m", payload)
    assert b.held == b.spent == 0
    assert b.journal_path.read_bytes() == before
    b.release_writer()


def test_73_dynamic_thinking_requires_bound_and_conflicting_chat_limits_refuse(tmp_path):
    b = budget(tmp_path)
    with pytest.raises(api().BudgetRefused):
        b.prepare("openai", "m", body(max_completion_tokens=10))
    with pytest.raises(api().BudgetRefused):
        b.prepare("google", "m", {"contents": [], "generationConfig": {"maxOutputTokens": 10, "thinkingConfig": {"thinkingBudget": -1}}})
    b.release_writer()


@pytest.mark.parametrize("value", [True, -1, 1.2, "5"])
def test_27_malformed_cached_subset_is_unknown(value):
    meter = api().UsageMeter("google")
    meter.observe({"candidates": [{"index": 0, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "cachedContentTokenCount": value}})
    assert meter.measurement().n is None


@pytest.mark.parametrize("dialect,key", [("openai", "prompt_tokens_details"), ("openai-responses", "input_tokens_details"), ("anthropic", "output_tokens_details")])
@pytest.mark.parametrize("value", [True, -1, 1.2, "5"])
def test_27_optional_usage_details_do_not_invent_zero(dialect, key, value):
    counts = {"prompt_tokens": 10, "completion_tokens": 20} if dialect == "openai" else {"input_tokens": 10, "output_tokens": 20}
    counts[key] = {"thinking_tokens" if dialect == "anthropic" else "cached_tokens": value}
    meter = api().UsageMeter(dialect)
    meter.observe({"usage": counts})
    assert not meter.measurement().complete


@pytest.mark.parametrize("field,value", [("presencePenalty", True), ("frequencyPenalty", "1"), ("logprobs", 1.5), ("responseLogprobs", 1)])
def test_73_other_google_generation_scalars(tmp_path, field, value):
    b = budget(tmp_path)
    payload = {"contents": [], "generationConfig": {"maxOutputTokens": 10, field: value}}
    with pytest.raises(api().BudgetRefused):
        b.prepare("google", "m", payload)
    b.release_writer()


@pytest.mark.parametrize("dialect", ["openai", "google"])
def test_43_nested_schema_size_is_bounded(tmp_path, dialect):
    b = budget(tmp_path)
    schema = {"type": "object", "description": "x" * 65536}
    if dialect == "openai":
        payload = body(tools=[{"type": "function", "function": {"name": "f", "parameters": schema}}])
    else:
        payload = {"contents": [], "generationConfig": {"maxOutputTokens": 10, "responseSchema": schema}}
    with pytest.raises(api().BudgetRefused):
        b.prepare(dialect, "m", payload)
    b.release_writer()
