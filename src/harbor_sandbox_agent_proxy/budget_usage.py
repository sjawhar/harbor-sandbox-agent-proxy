"""Raw provider usage, independent of Inspect's cache-adjusted ModelUsage."""
from __future__ import annotations
import dataclasses


@dataclasses.dataclass(frozen=True)
class Measurement:
    n: int | None = None
    o: int | None = None
    w: int = 0
    terminal: bool = False
    seen: bool = False
    outside_contract: bool = False

    @property
    def complete(self) -> bool:
        return self.terminal and self.n is not None and self.o is not None


def count(data: dict, key: str, optional: bool = False) -> int | None:
    value = data.get(key, 0 if optional else None)
    return value if type(value) is int and value >= 0 else None


def total(*values: int | None) -> int | None:
    return None if any(v is None for v in values) else sum(values)


def dialect_name(value: object) -> str:
    value = str(getattr(value, "value", value))
    return {"messages": "anthropic", "chat-completions": "openai", "chat_completions": "openai",
            "responses": "openai-responses", "gemini": "google"}.get(value, value)


def _valid_details(value: object) -> bool:
    return isinstance(value, dict) and all(
        _valid_details(v) if isinstance(v, dict) else type(v) is int and v >= 0
        for v in value.values()
    )


class UsageMeter:
    def __init__(self, dialect: str, branches: int = 1, streaming: bool = False):
        self.dialect = dialect_name(dialect)
        self.branches, self.streaming = branches, streaming
        self.n = self.o = None
        self.w = 0
        self.seen = self.terminal = self.outside_contract = False
        self._start = self._delta = False
        self._finished: set[int] = set()

    def _anthropic(self, usage: dict) -> None:
        self.seen = True
        iterations = usage.get("iterations")
        if isinstance(iterations, list) and iterations and not any(x.get("type") == "fallback_message" for x in iterations if isinstance(x, dict)):
            self.outside_contract = True
            first = iterations[0]
            self.n = total(count(first, "input_tokens"), count(first, "cache_read_input_tokens", True), count(first, "cache_creation_input_tokens", True))
            work = [total(count(x, "input_tokens"), count(x, "cache_read_input_tokens", True), count(x, "cache_creation_input_tokens", True)) for x in iterations[1:] if isinstance(x, dict)]
            self.w = total(*work) or 0
            self.o = total(*(count(x, "output_tokens") for x in iterations if isinstance(x, dict)))
        else:
            self.n = total(count(usage, "input_tokens"), count(usage, "cache_read_input_tokens", True), count(usage, "cache_creation_input_tokens", True))
            self.o = count(usage, "output_tokens")
        if "output_tokens_details" in usage and not _valid_details(usage["output_tokens_details"]):
            self.o = None

    def observe(self, body: object) -> None:
        if not isinstance(body, dict):
            return
        if self.dialect == "anthropic":
            usage = body.get("usage")
            if isinstance(usage, dict):
                self._anthropic(usage)
                self.terminal = True
        elif self.dialect in ("openai", "openai-responses"):
            usage = body.get("usage")
            if isinstance(usage, dict):
                self.seen = True
                prompt, output = ("prompt_tokens", "completion_tokens") if self.dialect == "openai" else ("input_tokens", "output_tokens")
                self.n, self.o = count(usage, prompt), count(usage, output)
                for field, component in (("prompt_tokens_details" if self.dialect == "openai" else "input_tokens_details", "n"),
                                         ("completion_tokens_details" if self.dialect == "openai" else "output_tokens_details", "o")):
                    if field in usage and not _valid_details(usage[field]):
                        setattr(self, component, None)
                self.terminal = True
        elif self.dialect == "google":
            candidates = body.get("candidates", [])
            if isinstance(candidates, list):
                for index, candidate in enumerate(candidates):
                    if isinstance(candidate, dict) and candidate.get("finishReason"):
                        self._finished.add(candidate.get("index", index))
            usage = body.get("usageMetadata")
            if isinstance(usage, dict):
                self.seen = True
                self.n = total(count(usage, "promptTokenCount"), count(usage, "toolUsePromptTokenCount", True))
                if "cachedContentTokenCount" in usage and count(usage, "cachedContentTokenCount") is None:
                    self.n = None
                blocked = isinstance(body.get("promptFeedback"), dict) and body["promptFeedback"].get("blockReason") and not candidates
                self.o = total(count(usage, "candidatesTokenCount", bool(blocked)), count(usage, "thoughtsTokenCount", True))
                reported = count(usage, "totalTokenCount", True)
                if "totalTokenCount" in usage and reported is None:
                    self.o = None
                elif self.n is not None and self.o is not None and reported is not None:
                    self.o += max(0, reported - self.n - self.o)
                self.terminal = bool(blocked) or len(self._finished) >= self.branches

    def feed(self, event: dict) -> None:
        kind = event.get("type")
        if self.dialect == "anthropic":
            if kind == "message_start":
                self.observe(event.get("message"))
                self.terminal = False
                self._start = self.seen
                self.o = None
            elif kind == "message_delta" and isinstance(event.get("usage"), dict):
                self.seen = True
                self.o = count(event["usage"], "output_tokens")
                self._delta = True
            elif kind == "message_stop":
                self.terminal = self._start and self._delta
        elif self.dialect == "openai-responses":
            if kind in ("response.completed", "response.incomplete", "response.failed"):
                self.observe(event.get("response"))
        else:
            self.observe(event)

    def measurement(self) -> Measurement:
        # A cumulative output seen before the terminal marker is not final output.
        return Measurement(self.n, self.o if self.terminal else None, self.w,
                           self.terminal, self.seen, self.outside_contract)
