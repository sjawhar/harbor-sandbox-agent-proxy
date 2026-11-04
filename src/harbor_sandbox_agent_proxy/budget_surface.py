"""Enumerated provider envelopes; arbitrary tool/schema payloads remain untouched."""
from __future__ import annotations
import hashlib
from .budget_usage import dialect_name

_COMMON = {"model", "stream", "tools", "tool_choice", "temperature", "top_p", "top_k", "seed", "stop", "metadata"}
_TOP = {
    "anthropic": _COMMON | {"messages", "system", "max_tokens", "thinking", "output_config", "service_tier", "stop_sequences"},
    "openai": _COMMON | {"messages", "max_tokens", "max_completion_tokens", "n", "stream_options", "reasoning_effort", "reasoning", "response_format", "parallel_tool_calls", "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs", "user", "store", "service_tier"},
    "openai-responses": _COMMON | {"input", "instructions", "max_output_tokens", "reasoning", "text", "parallel_tool_calls", "include", "store", "prompt_cache_key", "client_metadata", "service_tier", "truncation"},
    "google": {"model", "contents", "systemInstruction", "tools", "toolConfig", "generationConfig", "safetySettings", "labels"},
}
_BLOCKS = {
    "text": {"type", "text", "cache_control", "citations"},
    "tool_use": {"type", "id", "name", "input", "caller", "cache_control"},
    "tool_result": {"type", "tool_use_id", "content", "is_error", "cache_control"},
    "thinking": {"type", "thinking", "signature", "cache_control"},
    "redacted_thinking": {"type", "data", "cache_control"},
    "input_text": {"type", "text"}, "output_text": {"type", "text", "annotations", "logprobs"},
}
_ITEMS = {
    "message": {"type", "id", "role", "content", "status", "internal_chat_message_metadata_passthrough"},
    "function_call": {"type", "id", "call_id", "name", "namespace", "arguments", "status"},
    "function_call_output": {"type", "id", "call_id", "output", "status"},
    "custom_tool_call": {"type", "id", "call_id", "name", "input", "status"},
    "custom_tool_call_output": {"type", "id", "call_id", "output", "status"},
    "reasoning": {"type", "id", "summary", "content", "encrypted_content", "status"},
    "additional_tools": {"type", "id", "role", "tools"},
    "tool_search_call": {"type", "id", "call_id", "execution", "arguments", "status", "internal_chat_message_metadata_passthrough"},
    "tool_search_output": {"type", "id", "call_id", "execution", "tools", "status", "internal_chat_message_metadata_passthrough"},
}
_GENERATION = {"maxOutputTokens", "candidateCount", "thinkingConfig", "temperature", "topP", "topK", "seed", "stopSequences", "responseMimeType", "responseSchema", "responseJsonSchema", "presencePenalty", "frequencyPenalty", "responseLogprobs", "logprobs"}
_INERT = {"stream", "stream_options", "max_tokens", "max_completion_tokens", "max_output_tokens", "n", "temperature", "top_p", "top_k", "seed", "stop", "stop_sequences", "metadata"}


def reject(code: str = "unsupported_input") -> None:
    from .budget import BudgetRefused
    raise BudgetRefused(code)


def envelope(value: object, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - keys:
        reject()
    return value


def _blocks(value: object, dialect: str) -> None:
    if isinstance(value, str) or value is None:
        return
    if not isinstance(value, list):
        reject()
    for block in value:
        if not isinstance(block, dict):
            reject()
        kind = block.get("type")
        allowed = _BLOCKS.get(kind)
        if allowed is None or (dialect != "anthropic" and kind not in ("text", "input_text", "output_text")):
            reject()
        envelope(block, allowed)
        if kind == "tool_result":
            _blocks(block.get("content"), "anthropic")


def _tools(tools: object, dialect: str, max_schema: int) -> None:
    from .budget import serialize
    if not isinstance(tools, list):
        reject()
    for tool in tools:
        if not isinstance(tool, dict):
            reject()
        # Existing confinement removes provider-executed declarations. They are not input references.
        kind = tool.get("type", "function")
        if kind == "namespace":
            envelope(tool, {"type", "name", "description", "tools"})
            _tools(tool.get("tools", []), dialect, max_schema)
        elif dialect == "anthropic" and "name" in tool:
            envelope(tool, {"name", "description", "input_schema", "cache_control", "strict", "type", "defer_loading", "allowed_callers", "input_examples"})
        elif dialect == "openai" and kind == "function":
            envelope(tool, {"type", "function"})
            envelope(tool.get("function"), {"name", "description", "parameters", "strict"})
            if "parameters" in tool["function"] and len(serialize(tool["function"]["parameters"])) > max_schema:
                reject()
        elif dialect == "openai-responses" and kind in ("function", "custom", "tool_search"):
            envelope(tool, {"type", "name", "description", "parameters", "strict", "defer_loading", "format", "execution"})
        elif dialect == "google":
            envelope(tool, {"functionDeclarations", "googleSearch", "codeExecution", "googleSearchRetrieval", "urlContext"})
            for declaration in tool.get("functionDeclarations", []):
                envelope(declaration, {"name", "description", "parameters", "parametersJsonSchema", "response", "responseJsonSchema", "behavior"})
                for key in ("parameters", "parametersJsonSchema", "response", "responseJsonSchema"):
                    if key in declaration and len(serialize(declaration[key])) > max_schema:
                        reject()
        for key in ("input_schema", "parameters", "parametersJsonSchema"):
            if key in tool and len(serialize(tool[key])) > max_schema:
                reject()


def validate_surface(dialect: str, body: dict, max_schema: int = 65536) -> None:
    from .budget import serialize
    dialect = dialect_name(dialect)
    if dialect not in _TOP:
        reject()
    envelope(body, _TOP[dialect])
    if dialect == "anthropic":
        if "system" in body:
            _blocks(body["system"], dialect)
        for message in body.get("messages", []):
            envelope(message, {"role", "content"})
            _blocks(message.get("content"), dialect)
        if "thinking" in body:
            envelope(body["thinking"], {"type", "budget_tokens", "display"})
        if "output_config" in body:
            envelope(body["output_config"], {"effort", "format"})
    elif dialect == "openai":
        for message in body.get("messages", []):
            envelope(message, {"role", "content", "name", "tool_calls", "tool_call_id", "refusal"})
            _blocks(message.get("content"), dialect)
            for tool in message.get("tool_calls", []):
                envelope(tool, {"id", "type", "function"})
                if tool.get("type") != "function":
                    reject()
                envelope(tool.get("function"), {"name", "arguments"})
        if "stream_options" in body:
            envelope(body["stream_options"], {"include_usage"})
    elif dialect == "openai-responses":
        if body.get("store") is True or body.get("truncation", "disabled") != "disabled":
            reject()
        items = body.get("input", [])
        if isinstance(items, str):
            items = []
        if not isinstance(items, list):
            reject()
        for item in items:
            kind = item.get("type", "message") if isinstance(item, dict) else None
            if kind not in _ITEMS:
                reject()
            envelope(item, _ITEMS[kind])
            if kind == "message":
                _blocks(item.get("content"), dialect)
            if kind.startswith("tool_search") and item.get("execution") != "client":
                reject()
            if "tools" in item:
                _tools(item["tools"], dialect, max_schema)
    else:
        envelope(body.get("generationConfig", {}), _GENERATION)
        for key in ("responseSchema", "responseJsonSchema"):
            if key in body.get("generationConfig", {}) and len(serialize(body["generationConfig"][key])) > max_schema:
                reject()
        for message in [*body.get("contents", []), *([body["systemInstruction"]] if "systemInstruction" in body else [])]:
            envelope(message, {"role", "parts"})
            for part in message.get("parts", []):
                envelope(part, {"text", "functionCall", "functionResponse", "thought", "thoughtSignature"})
                if "functionCall" in part:
                    envelope(part["functionCall"], {"name", "args", "id"})
                if "functionResponse" in part:
                    envelope(part["functionResponse"], {"name", "response", "id", "willContinue", "scheduling"})
    if "tools" in body:
        _tools(body["tools"], dialect, max_schema)
    for key in ("response_format", "text", "output_config"):
        if key in body and len(serialize(body[key])) > max_schema:
            reject()
    if "reasoning" in body:
        envelope(body["reasoning"], {"effort", "summary", "generate_summary", "context"})


def _integer(value: object, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        reject("invalid_request_value")
    return value


def _normalized_message(message: dict, dialect: str) -> dict:
    result = dict(message)
    content = result.get("content")
    if dialect in ("anthropic", "openai") and isinstance(content, str):
        result["content"] = [{"type": "text", "text": content}]
    elif dialect == "anthropic" and isinstance(content, list):
        result["content"] = [{k: v for k, v in block.items() if k != "cache_control"} for block in content]
    return result


def prepare(config, dialect: str, model: str, body: dict):
    from .budget import Plan, serialize
    dialect = dialect_name(dialect)
    validate_surface(dialect, body, config.max_schema_bytes)
    if model not in config.profiles:
        reject("unsupported_model_profile")
    p = config.profiles[model]
    ceiling = config.output_ceiling[model]
    branches = 1
    if dialect == "openai":
        branches = _integer(body.get("n", 1), 1, config.max_branches.get(model, 1))
        if "max_tokens" in body and "max_completion_tokens" in body:
            reject("invalid_request_value")
        field = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        target = body
    elif dialect == "google":
        target = body.setdefault("generationConfig", {})
        branches = _integer(target.get("candidateCount", 1), 1, config.max_branches.get(model, 1))
        field = "maxOutputTokens"
    else:
        target = body
        field = "max_output_tokens" if dialect == "openai-responses" else "max_tokens"
    if field not in target:
        if not config.inject_output_limit:
            reject("output_limit_required")
        target[field] = ceiling
    output = _integer(target[field], 1, ceiling)
    for key in ("stream", "parallel_tool_calls", "store", "logprobs"):
        if key in body and type(body[key]) is not bool:
            reject("invalid_request_value")
    if "stream_options" in body and "include_usage" in body["stream_options"] and type(body["stream_options"]["include_usage"]) is not bool:
        reject("invalid_request_value")
    for source, fields in ((body, {"temperature": (0, 2), "top_p": (0, 1), "top_k": (0, 2**31 - 1), "seed": (0, 2**63 - 1), "frequency_penalty": (-2, 2), "presence_penalty": (-2, 2), "top_logprobs": (0, 20)}),
                           (target if dialect == "google" else {}, {"temperature": (0, 2), "topP": (0, 1), "topK": (0, 2**31-1), "seed": (0, 2**63-1), "presencePenalty": (-2, 2), "frequencyPenalty": (-2, 2), "logprobs": (0, 20)})):
        for key, (low, high) in fields.items():
            if key in source and (type(source[key]) not in (int, float) or not low <= source[key] <= high):
                reject("invalid_request_value")
            if key in source and key in ("top_k", "topK", "seed", "top_logprobs", "logprobs"):
                _integer(source[key], low, high)
    if dialect == "google" and "responseLogprobs" in target and type(target["responseLogprobs"]) is not bool:
        reject("invalid_request_value")
    thinking = body.get("thinking", {}) if dialect == "anthropic" else target.get("thinkingConfig", {}) if dialect == "google" else {}
    if thinking:
        envelope(thinking, {"type", "budget_tokens", "display"} if dialect == "anthropic" else {"thinkingBudget", "includeThoughts", "thinkingLevel"})
        mode = thinking.get("type", "enabled")
        if mode not in ("enabled", "disabled", "adaptive"):
            reject("invalid_request_value")
        key = "budget_tokens" if dialect == "anthropic" else "thinkingBudget"
        if key in thinking:
            value = thinking[key]
            if model not in config.thinking_ceiling:
                reject("unbounded_thinking")
            if not (dialect == "google" and type(value) is int and value == -1):
                _integer(value, 0, config.thinking_ceiling[model])
            if dialect == "anthropic" and value >= output:
                reject("invalid_request_value")
        elif dialect == "google" and model not in config.thinking_ceiling:
            reject("unbounded_thinking")
    effort = body.get("reasoning_effort", body.get("reasoning", {}).get("effort"))
    if effort is not None and effort not in p.reasoning_efforts:
        reject("invalid_request_value")
    if dialect == "openai" and body.get("stream"):
        body.setdefault("stream_options", {})["include_usage"] = True
    if dialect != "google":
        body["model"] = model
    wire = serialize(body)
    opaque = dialect == "openai-responses" and isinstance(body.get("input"), list) and any(isinstance(x, dict) and x.get("encrypted_content") for x in body["input"])
    if opaque and p.effective_input_tokens_max is None:
        reject("unsupported_input")
    r_in = max(len(wire), p.effective_input_tokens_max or 0) + p.route_injected_tokens_max
    thinking_bound = config.thinking_ceiling.get(model, 0) if p.outside_output_thinking else 0
    r_out = branches * output + (branches * thinking_bound if p.thinking_scope == "per_branch" else thinking_bound)
    reserved = r_in + r_out
    if type(reserved) is not int or reserved <= 0:
        reject("invalid_reservation")
    if reserved > (config.max_reservation or config.limit):
        reject("reservation_too_large")
    sequence_key = "contents" if dialect == "google" else "input" if dialect == "openai-responses" else "messages"
    messages = body.get(sequence_key, [])
    if isinstance(messages, str):
        messages = [{"role": "user", "content": [{"type": "input_text", "text": messages}]}]
    root = {k: v for k, v in body.items() if k != sequence_key and k not in _INERT}
    if dialect == "anthropic":
        for key in ("system", "tools"):
            if isinstance(root.get(key), list):
                root[key] = [{k: v for k, v in item.items() if k != "cache_control"} for item in root[key]]
    if dialect == "google" and "generationConfig" in root:
        root["generationConfig"] = {k: v for k, v in root["generationConfig"].items() if k not in {"maxOutputTokens", "candidateCount", "temperature", "topP", "topK", "seed", "stopSequences"}}
    root_id = hashlib.sha256(serialize([dialect, model, root])).hexdigest()
    h = hashlib.sha256(root_id.encode()).digest()
    prefixes = []
    for message in messages:
        h = hashlib.sha256(h + serialize(_normalized_message(message, dialect))).digest()
        prefixes.append(h.hex())
    msg_hash = prefixes[-1] if prefixes else h.hex()
    return Plan(wire, root_id, tuple(prefixes), msg_hash, r_in, r_out, branches, model, p.id, p.credit == "eligible")
