"""Per-trial harbor-growth/1 accounting.

The journal is authoritative; snapshots are caches. Bounds are conditional on the
trusted route contract, not inferred from successful samples. No request bodies,
credentials or provider text are retained by the ledger.
"""
from __future__ import annotations

import asyncio
import dataclasses
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Annotated, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt, model_validator

from .budget_usage import Measurement, UsageMeter

RULE = "harbor-growth/1"
SCHEMA = 1
CAUSES = frozenset({"limit", "insufficient", "bound_violated", "usage_unknown", "persistence_failed"})
STAGES = frozenset({"A", "B", "C", "D", "E", "F", "G", "forced_close"})
CHARGE_KINDS = frozenset({"zero", "measured", "partial", "bound"})
REFUSAL_CODES = frozenset({
    "token_budget", "token_budget_insufficient", "token_budget_wait", "budget_closed",
    "usage_unknown", "unsupported_input", "unsupported_model_profile", "output_limit_required",
    "invalid_request_value", "invalid_reservation", "reservation_too_large", "unbounded_thinking", "bound_violated",
})
Positive = Annotated[StrictInt, Field(gt=0)]
Nonnegative = Annotated[StrictInt, Field(ge=0)]


def serialize(value: object) -> bytes:
    """The compact typed JSON representation also used by the budgeted sender."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def event_hash(event: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in event.items() if k != "h"}, sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class BudgetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: str = Field(min_length=1)
    credit: Literal["none", "eligible"] = "none"
    credit_evidence: str | None = None
    bound_source: str = Field(min_length=1)
    bound_version: str = Field(min_length=1)
    route_injected_tokens_max: Nonnegative
    route_bound_source: str = Field(min_length=1)
    route_bound_version: str = Field(min_length=1)
    effective_input_tokens_max: Positive | None = None
    outside_output_thinking: StrictBool = False
    thinking_scope: Literal["per_branch", "aggregate"] = "per_branch"
    reasoning_efforts: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh")

    @model_validator(mode="after")
    def eligible_evidence(self):
        if self.credit == "eligible" and not self.credit_evidence:
            raise ValueError("eligible credit requires model/mode-specific M2 evidence")
        if self.thinking_scope == "aggregate" and not self.bound_source:
            raise ValueError("aggregate thinking requires an explicit bound contract")
        return self


class TokenBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Positive
    rule: Literal["harbor-growth/1"]
    overshoot: Literal["none", "one_call"]
    unknown_usage: Literal["bound", "stop"]
    enforce: StrictBool
    hsap_commit: str = Field(min_length=1)
    output_ceiling: dict[str, Positive]
    profiles: dict[str, BudgetProfile]
    thinking_ceiling: dict[str, Nonnegative] = Field(default_factory=dict)
    max_branches: dict[str, Positive] = Field(default_factory=dict)
    max_reservation: Positive | None = None
    max_schema_bytes: Positive = 65536
    inject_output_limit: StrictBool = False
    wait_seconds: float = Field(default=900, ge=0, allow_inf_nan=False)
    close_grace_seconds: float = Field(default=5, ge=0, allow_inf_nan=False)
    zero_charge_statuses: tuple[StrictInt, ...] = (400, 401, 403, 404, 413, 422, 429)

    @model_validator(mode="after")
    def bounds_complete(self):
        if not self.enforce:
            return self
        if set(self.output_ceiling) != set(self.profiles):
            raise ValueError("every routed model needs an output ceiling and bound profile")
        if not set(self.thinking_ceiling) <= set(self.profiles) or not set(self.max_branches) <= set(self.profiles):
            raise ValueError("bound keys must name routed models")
        for model, p in self.profiles.items():
            if p.outside_output_thinking and model not in self.thinking_ceiling:
                raise ValueError("outside-output thinking requires a ceiling")
        return self


class ShadowBudgetProposal(BaseModel):
    """An explicitly disabled policy may be invalid; it is diagnostic input only."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    enforce: StrictBool

    @model_validator(mode="after")
    def disabled_only(self):
        if self.enforce:
            raise ValueError("an unvalidated proposal requires enforce:false")
        return self


class BudgetRefused(Exception):
    def __init__(self, code: str, message: str | None = None, status: int = 400):
        self.code, self.status = code, status
        super().__init__(message or code)


@dataclasses.dataclass(frozen=True)
class Plan:
    wire: bytes
    root_id: str
    prefixes: tuple[str, ...]
    msg_hash: str
    r_in: int
    r_out: int
    branches: int
    model: str
    profile: str
    credit: bool

    @property
    def reserved(self) -> int:
        return self.r_in + self.r_out


@dataclasses.dataclass
class Attempt:
    seq: int
    call_id: str
    attempt: int
    root_id: str
    prefixes: tuple[str, ...]
    msg_hash: str
    r_in: int
    r_out: int
    branches: int
    model: str
    profile: str
    credit: bool
    forwarded_bytes: int
    task: asyncio.Task | None = None
    settled: bool = False
    usage: Measurement | None = None
    stage: str | None = None
    facts: dict | None = None

    @property
    def reserved(self) -> int:
        return self.r_in + self.r_out


def native_refusal_code(message: object) -> str | None:
    """Decode hsap's closed terminal message suffix, not an authentication witness."""
    if not isinstance(message, str):
        return None
    _, marker, tail = message.rpartition(" [hsap-budget-code:")
    if marker and tail.endswith("]") and tail[:-1] in REFUSAL_CODES:
        return tail[:-1]
    return None


def encode_refusal_code(message: str, code: str) -> str:
    if code not in REFUSAL_CODES:
        raise ValueError("unknown budget refusal code")
    if native_refusal_code(message) == code:
        return message
    return f"{message} [hsap-budget-code:{code}]"


def _refusal_message(limit: int, spent: int, cause: str, reserved: int) -> str:
    if cause == "insufficient":
        text = f"token budget of {limit:,} cannot fit the next request ({limit - spent:,} remaining, {reserved:,} reserved, rule {RULE})"
        code = "token_budget_insufficient"
    elif cause == "limit":
        text = f"token budget of {limit:,} spent ({spent:,} tokens counted, rule {RULE})"
        code = "token_budget"
    else:
        text = f"token budget admission stopped ({cause}, rule {RULE})"
        code = "token_budget"
    return encode_refusal_code(text, code)


class TrialBudget:
    def __init__(self, config: TokenBudgetConfig, directory: Path, trial: str, *,
                 new_ledger: bool = False, on_error: Callable[[Exception], None] | None = None):
        self.config, self.directory, self.trial = config, Path(directory), trial
        self.config_hash = event_hash(config.model_dump(mode="json"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / "budget.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock.close()
            raise RuntimeError("budget journal already has an active writer") from error
        self.snapshot_path = self.directory / "budget.json"
        self.previous_ledger_id = None
        first = self.directory / "budget.jsonl"
        if first.exists():
            prior = read_budget(self.directory)
            if not new_ledger:
                self.release_writer()
                raise RuntimeError("a trial ledger already exists; explicit new_ledger required")
            self.previous_ledger_id = prior.get("ledger_id")
        self.ledger_id = str(uuid4())
        self.journal_path = (self.directory / f"budget.{self.ledger_id}.jsonl") if first.exists() else first
        self.spent = self.held = 0
        self.lifecycle = "open"
        self.latch_cause = None
        self.admission_fence = None
        self.record_error = None
        self.on_error = on_error
        self._seq = -1
        self._next = 1
        self._attempts: dict[int, Attempt] = {}
        self._admitted = 0
        self._waiters: list[asyncio.Future] = []
        self._changed = asyncio.Event()
        self._close_task: asyncio.Task[dict] | None = None
        self._close_callers: set[asyncio.Task] = set()
        self._checkpoints: dict[tuple[str, str], tuple[int, int]] = {}
        self._roots: dict[str, int] = {}
        self._last_root_id: str | None = None
        self._contexts: dict[str, dict] = {}
        self._closed_peaks = 0
        self._peaks: dict[str, int] = {}
        self._stats = dict(bound_charged=0, partial_charged=0, billed_tokens=0,
                           checkpoint_conflicts=0, credit_invalidations=0, forks=0,
                           root_first_use_charged=0, late_completion=0, iteration_work=0,
                           repeated_root_contexts=0)
        self._refusals: dict[str, int] = {}
        self._witnesses: list[dict] = []
        self._forced: list[int] = []
        self.crossing_allowance = 0
        try:
            self._append("open", trial=trial, config_hash=self.config_hash,
                         hsap_commit=config.hsap_commit, config=config.model_dump(mode="json"),
                         previous_ledger_id=self.previous_ledger_id)
            self._snapshot()
        except BaseException:
            self.release_writer()
            raise

    def recorded_bytes(self) -> int:
        path = self.directory / "usage.jsonl"
        return path.stat().st_size if path.exists() else 0

    def prepare(self, dialect: str, model: str, body: dict) -> Plan:
        from .budget_surface import prepare
        return prepare(self.config, dialect, model, body)

    def validate_surface(self, dialect: str, body: dict) -> None:
        from .budget_surface import validate_surface
        validate_surface(dialect, body, self.config.max_schema_bytes)

    def _append(self, kind: str, **fields) -> None:
        self._seq += 1
        event = dict(version=SCHEMA, kind=kind, ledger_id=self.ledger_id, seq=self._seq, **fields)
        event["h"] = event_hash(event)
        created = not self.journal_path.exists()
        with self.journal_path.open("ab") as stream:
            stream.write(serialize(event) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if created:
            self._fsync_directory()

    def _fsync_directory(self) -> None:
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _persist(self, events: list[tuple[str, dict]]) -> None:
        try:
            for kind, fields in events:
                self._append(kind, **fields)
            self._snapshot()
        except Exception as error:
            self.admission_fence = "persistence_failed"
            self.latch_cause = self.latch_cause or "persistence_failed"
            self.record_error = self.record_error or error
            if self.on_error is not None:
                self.on_error(error)
            self._wake()
            raise

    def _snapshot(self) -> None:
        temporary = self.snapshot_path.with_suffix(".json.tmp")
        data = self.record()
        data["journal"] = self.journal_path.name
        with temporary.open("wb") as stream:
            stream.write(serialize(data))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.snapshot_path)
        self._fsync_directory()

    def _wake(self) -> None:
        self._changed.set()
        for future in self._waiters:
            if not future.done():
                future.set_result(None)

    async def admit(self, plan: Plan, call_id: str, attempt: int = 1) -> Attempt:
        deadline = time.monotonic() + self.config.wait_seconds
        while True:
            if self.lifecycle != "open":
                raise BudgetRefused("budget_closed")
            if self.config.enforce:
                if self.admission_fence or self.latch_cause:
                    self.refuse(call_id, self.latch_cause or self.admission_fence, plan.reserved)
                fits = self.spent + self.held + plan.reserved <= self.config.limit
                crossing = self.config.overshoot == "one_call" and self.held == 0 and self.spent < self.config.limit
                if not fits and not crossing:
                    if self.held == 0:
                        self.latch_cause = self.latch_cause or "insufficient"
                        self._persist([("latch", dict(cause=self.latch_cause, at=self.spent))])
                        self.refuse(call_id, "insufficient", plan.reserved)
                    future = asyncio.get_running_loop().create_future()
                    self._waiters.append(future)
                    try:
                        await asyncio.wait_for(future, max(0, deadline - time.monotonic()))
                    except TimeoutError:
                        raise BudgetRefused("token_budget_wait", status=429) from None
                    finally:
                        self._waiters.remove(future)
                    continue
                if not fits:
                    self.crossing_allowance = plan.reserved
            self._admitted += 1
            item = Attempt(self._admitted, call_id, attempt, plan.root_id, plan.prefixes,
                           plan.msg_hash, plan.r_in, plan.r_out, plan.branches, plan.model,
                           plan.profile, plan.credit, len(plan.wire), asyncio.current_task())
            self._attempts[item.seq] = item
            self.held += item.reserved
            self._persist([("admit", dict(admit_seq=item.seq, call_id=call_id, attempt=attempt,
                         root_id=item.root_id, msg_hash=item.msg_hash, R=item.reserved,
                         r_in=item.r_in, r_out=item.r_out, profile=item.profile,
                         forwarded_bytes=item.forwarded_bytes, crossing_allowance=self.crossing_allowance))])
            return item

    def refuse(self, call_id: str, cause: str, reserved: int) -> None:
        if cause == "insufficient" and reserved == 0 and self._witnesses:
            reserved = self._witnesses[-1]["reserved"]
        code = "token_budget_insufficient" if cause == "insufficient" else "token_budget"
        message = _refusal_message(self.config.limit, self.spent, cause, reserved)
        if self._witnesses and self._witnesses[-1]["message"] == message:
            raise BudgetRefused(code, message)
        event = dict(call_id=call_id, code=code, cause=cause, message=message,
                     spent=self.spent, remaining=self.config.limit - self.spent, reserved=reserved)
        self._refusals[code] = self._refusals.get(code, 0) + 1
        self._witnesses.append(event)
        self._persist([("budget_refused", event)])
        raise BudgetRefused(code, message)

    def settle(self, item: Attempt, usage: Measurement, *, sent: bool = True,
               status: int | None = 200, connect_failed: bool = False, forced: bool = False,
               terminal_error: bool = False) -> dict:
        if item.settled:
            self._stats["late_completion"] += 1
            return dict(state="after_close" if self.lifecycle == "closed" else "late_completion",
                        charge=None, n=usage.n, o=usage.o, admit_seq=item.seq,
                        call_id=item.call_id, attempt=item.attempt, ledger_id=self.ledger_id)
        item.settled, item.usage, item.task = True, usage, None
        zero = not sent or connect_failed or (status in self.config.zero_charge_statuses and not usage.seen and usage.n is None and usage.o is None)
        if not sent:
            item.stage = "A"
        elif connect_failed:
            item.stage = "B"
        elif zero:
            item.stage = "C"
        elif usage.complete:
            item.stage = "G" if terminal_error else "D"
        elif usage.n is not None or usage.o is not None:
            item.stage = "E"
        else:
            item.stage = "F"
        if forced:
            item.stage = "forced_close"
            self._forced.append(item.seq)
        unsafe = "bound_violated" if (usage.outside_contract or
            (usage.n is not None and usage.n > item.r_in) or
            (usage.o is not None and usage.o > item.r_out)) else (
            "usage_unknown" if not zero and not usage.complete and self.config.unknown_usage == "stop" else None)
        events = [("settle", dict(admit_seq=item.seq, stage=item.stage, n=usage.n, o=usage.o,
                                   w=usage.w, terminal=usage.terminal, zero=zero, unsafe=unsafe))]
        if unsafe and self.admission_fence is None:
            self.admission_fence = unsafe
            events.append(("fence", dict(cause=unsafe, admit_seq=item.seq)))
        item.facts = dict(state="settled_pending_commit", ledger_id=self.ledger_id,
                          admit_seq=item.seq, call_id=item.call_id, attempt=item.attempt,
                          n=usage.n, o=usage.o, reserved=item.reserved, r_in=item.r_in,
                          r_out=item.r_out, forwarded_bytes=item.forwarded_bytes,
                          charge=None, spent_after=None, prefix_tokens=None, latched_after=None,
                          stage=item.stage, profile=item.profile, rule=RULE,
                          root_changed=None, iteration_work=usage.w)
        item._zero = zero
        item._unsafe = unsafe
        while (current := self._attempts.get(self._next)) is not None and current.settled:
            facts, commit_events = self._commit(current)
            events.extend(commit_events)
            current.facts.update(facts)
            self._attempts.pop(self._next)
            self._next += 1
        self._persist(events)
        self._wake()
        return dict(item.facts)

    def _commit(self, item: Attempt) -> tuple[dict, list]:
        usage = item.usage
        assert usage is not None
        matches = [(prefix, self._checkpoints[(item.root_id, prefix)]) for prefix in item.prefixes
                   if (item.root_id, prefix) in self._checkpoints]
        structural = matches[-1][1] if matches else None
        credit, credited_from, invalidated = 0, None, False
        if structural and item.credit and usage.n is not None:
            if usage.n >= structural[0]:
                credit, credited_from = structural
            else:
                invalidated = True
                self._stats["credit_invalidations"] += 1
        if item._zero:
            charge, kind = 0, "zero"
        else:
            charge = (item.r_in if usage.n is None else max(0, usage.n - credit)) + usage.w + (item.r_out if usage.o is None else usage.o)
            kind = "measured" if usage.complete else "partial" if usage.n is not None or usage.o is not None else "bound"
            if kind in ("partial", "bound"):
                self._stats[kind + "_charged"] += charge
        first = item.root_id not in self._roots
        root_changed = self._last_root_id is not None and item.root_id != self._last_root_id
        self._last_root_id = item.root_id
        if first:
            self._stats["root_first_use_charged"] += charge
        self._roots[item.root_id] = self._roots.get(item.root_id, 0) + charge
        self.spent += charge
        self.held -= item.reserved
        self._stats["iteration_work"] += usage.w
        if usage.n is not None and usage.o is not None:
            self._stats["billed_tokens"] += usage.n + usage.o + usage.w
            peak = usage.n + usage.o
            self._peaks[item.root_id] = max(self._peaks.get(item.root_id, 0), peak)
            ctx = self._contexts.get(item.root_id)
            if ctx and structural is None:
                self._closed_peaks += ctx["peak"]
                self._stats["repeated_root_contexts"] += 1
                ctx = None
            elif ctx and matches and matches[-1][0] != ctx["msg_hash"]:
                self._stats["forks"] += 1
            self._contexts[item.root_id] = dict(peak=max(peak, ctx["peak"] if ctx else 0), msg_hash=item.msg_hash)
        checkpoint = None
        key = (item.root_id, item.msg_hash)
        if kind == "measured" and usage.n is not None:
            if key in self._checkpoints:
                if self._checkpoints[key][0] != usage.n:
                    self._stats["checkpoint_conflicts"] += 1
            else:
                self._checkpoints[key] = (usage.n, item.seq)
                checkpoint = dict(key=item.msg_hash, length=usage.n)
        cause = item._unsafe or ("limit" if self.spent >= self.config.limit else None)
        new_latch = cause if self.latch_cause is None else None
        if new_latch:
            self.latch_cause = new_latch
        facts = dict(state="committed", charge=charge, charge_kind=kind, prefix_tokens=credit,
                     credited_from=credited_from, credit_invalidated=invalidated,
                     spent_after=self.spent, latched_after=self.latch_cause,
                     root_id=item.root_id, root_changed=root_changed, root_first_use=first,
                     branches=item.branches, iteration_work=usage.w)
        event = dict(admit_seq=item.seq, call_id=item.call_id, attempt=item.attempt,
                     K=credit, credited_from=credited_from, checkpoint_added=checkpoint,
                     charge=charge, charge_kind=kind, spent_after=self.spent,
                     latch_cause_after=self.latch_cause, diagnostics=self._diagnostics())
        events = [("commit", event)]
        if new_latch:
            events.append(("latch", dict(cause=new_latch, at=self.spent)))
        return facts, events

    def _diagnostics(self) -> dict:
        return dict(**self._stats, per_root=dict(self._roots), peak_context_per_root=sum(self._peaks.values()),
                    carried_context=self._closed_peaks + sum(c["peak"] for c in self._contexts.values()),
                    carried_context_approximate=bool(self._stats["forks"] or self._stats["credit_invalidations"]),
                    branch_work="unknown")

    def record(self) -> dict:
        return dict(version=SCHEMA, ledger_id=self.ledger_id, trial=self.trial,
                    config_hash=self.config_hash, hsap_commit=self.config.hsap_commit,
                    rule=RULE, enforce=self.config.enforce, limit=self.config.limit,
                    profiles={k: p.model_dump(mode="json") for k, p in self.config.profiles.items()},
                    spent=self.spent, held=self.held, lifecycle=self.lifecycle,
                    latch_cause=self.latch_cause, admission_fence=self.admission_fence,
                    complete=self.lifecycle == "closed" and not self._attempts and self.record_error is None,
                    incomplete_reason=None if self.lifecycle == "closed" else "missing_close",
                    refusals=dict(self._refusals), refusal_witnesses=list(self._witnesses),
                    forced_settled=list(self._forced), crossing_allowance=self.crossing_allowance,
                    **self._diagnostics())

    async def close_out(self) -> dict:
        caller = asyncio.current_task()
        if caller is not None:
            self._close_callers.add(caller)
        try:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_once())
            return await asyncio.shield(self._close_task)
        finally:
            if caller is not None:
                self._close_callers.discard(caller)

    async def _close_once(self) -> dict:
        if self.lifecycle == "closed":
            return self.record()
        self.lifecycle = "closing"
        self._persist([("close_begin", {})])
        self._wake()
        deadline = time.monotonic() + self.config.close_grace_seconds
        while self._attempts and time.monotonic() < deadline:
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), max(0, deadline - time.monotonic()))
            except TimeoutError:
                break
        for item in list(self._attempts.values()):
            if not item.settled:
                # A task awaiting this shutdown must not cancel its own cleanup wait.
                if item.task is not None and item.task is not asyncio.current_task() and item.task not in self._close_callers:
                    item.task.cancel()
                self.settle(item, Measurement(), forced=True)
        self.lifecycle = "closed"
        try:
            self._persist([("close", dict(spent=self.spent, forced_settled=self._forced))])
        finally:
            self.release_writer()
        return self.record()

    def release_writer(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()


def _journal_lines(path: Path):
    with path.open("rb") as stream:
        yield from stream


def _generation_path(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    generations = {}
    parents = set()
    for path in paths:
        with path.open("rb") as stream:
            line = stream.readline()
        event = json.loads(line)
        if not isinstance(event, dict) or not line.endswith(b"\n") or event.get("kind") != "open" or event.get("h") != event_hash(event):
            raise ValueError("invalid generation header")
        ledger = event["ledger_id"]
        if ledger in generations:
            raise ValueError("duplicate ledger generation")
        generations[ledger] = path
        if event.get("previous_ledger_id"):
            parents.add(event["previous_ledger_id"])
    tips = set(generations) - parents
    if len(tips) != 1 or not parents <= set(generations):
        raise ValueError("ambiguous or missing generation lineage")
    return generations[tips.pop()]


def read_budget(directory: Path) -> dict:
    """Replay validated durable events. A snapshot alone never supplies a witness."""
    directory = Path(directory)
    if (directory / "sandbox-agent-proxy").is_dir():
        directory = directory / "sandbox-agent-proxy"
    paths = [directory / "budget.jsonl", *directory.glob("budget.*.jsonl")]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return dict(complete=False, incomplete_reason="missing_journal", lifecycle="open")
    try:
        path = _generation_path(paths)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(complete=False, incomplete_reason="invalid_generation", lifecycle="open")
    result = dict(complete=False, incomplete_reason="missing_close", lifecycle="open", spent=0,
                  held=0, refusals={}, refusal_witnesses=[], forced_settled=[], latch_cause=None,
                  admission_fence=None, bound_charged=0, partial_charged=0, crossing_allowance=0)
    result["attempts"] = {}
    admits, settled, committed = {}, {}, set()
    ledger = None
    closed = False
    reason = None
    last_committed_latch = None
    forced = []
    previous_root = None
    try:
        for expected_seq, line in enumerate(_journal_lines(path)):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not line.endswith(b"\n") or event.get("h") != event_hash(event):
                    raise ValueError
            except (ValueError, TypeError, UnicodeError):
                reason = "torn_tail"
                break
            if type(event.get("version")) is not int or event["version"] != SCHEMA or type(event.get("seq")) is not int or event["seq"] != expected_seq:
                reason = "event_schema"
                break
            kind = event.get("kind")
            if closed:
                reason = "event_after_close"
                break
            if expected_seq == 0:
                if kind != "open":
                    reason = "missing_open"
                    break
                ledger = event["ledger_id"]
                cfg = TokenBudgetConfig.model_validate(event["config"])
                if event["config_hash"] != event_hash(cfg.model_dump(mode="json")) or event["hsap_commit"] != cfg.hsap_commit:
                    reason = "config_hash"
                    break
                result.update(ledger_id=ledger, trial=event["trial"], config_hash=event["config_hash"],
                              hsap_commit=event["hsap_commit"], rule=cfg.rule, enforce=cfg.enforce,
                              limit=cfg.limit, profiles={k: v.model_dump(mode="json") for k, v in cfg.profiles.items()})
                continue
            if event.get("ledger_id") != ledger:
                reason = "ledger_id_mismatch"
                break
            seq = event.get("admit_seq")
            if kind == "admit":
                if result["lifecycle"] != "open" or seq != len(admits) + 1 or type(event["R"]) is not int or event["R"] < 1:
                    reason = "invalid_admit"
                    break
                if any(type(event[k]) is not int or event[k] <= 0 for k in ("r_in", "r_out")) or event["R"] != event["r_in"] + event["r_out"]:
                    reason = "invalid_reservation"
                    break
                allowance = event["crossing_allowance"]
                fits = result["spent"] + result["held"] + event["R"] <= cfg.limit
                crossing = cfg.enforce and not fits
                if crossing and (cfg.overshoot != "one_call" or result["held"] != 0 or result["spent"] >= cfg.limit):
                    reason = "invalid_crossing"
                    break
                expected_allowance = event["R"] if crossing else result["crossing_allowance"]
                if type(allowance) is not int or allowance != expected_allowance:
                    reason = "crossing_allowance"
                    break
                result["crossing_allowance"] = allowance
                admits[seq] = event
                result["held"] += event["R"]
            elif kind == "settle":
                if seq not in admits or seq in settled:
                    reason = "duplicate_settle" if seq in settled else "unadmitted_settle"
                    break
                if event["stage"] not in STAGES or type(event["terminal"]) is not bool or type(event["zero"]) is not bool or event["unsafe"] not in (None, *CAUSES):
                    reason = "settle_schema"
                    break
                if any(v is not None and (type(v) is not int or v < 0) for v in (event["n"], event["o"], event["w"])):
                    reason = "settle_counts"
                    break
                if event["stage"] == "forced_close":
                    forced.append(seq)
                settled[seq] = event
            elif kind == "commit":
                if seq not in settled or seq != len(committed) + 1:
                    reason = "commit_order"
                    break
                charge = event["charge"]
                if type(charge) is not int or charge < 0 or event["spent_after"] != result["spent"] + charge:
                    reason = "charge_sum"
                    break
                committed.add(seq)
                result["spent"] += charge
                result["held"] -= admits[seq]["R"]
                result["attempts"][f"{event['call_id']}:{event['attempt']}"] = {
                    "ledger_id": ledger, "admit_seq": seq, "charge": charge,
                    "prefix_tokens": event["K"], "spent_after": event["spent_after"],
                    "charge_kind": event["charge_kind"], "latched_after": event["latch_cause_after"],
                    "root_id": admits[seq]["root_id"],
                    "root_changed": previous_root is not None and previous_root != admits[seq]["root_id"],
                    "iteration_work": settled[seq]["w"], "stage": settled[seq]["stage"],
                }
                previous_root = admits[seq]["root_id"]
                if event["latch_cause_after"] is not None:
                    if result["latch_cause"] not in (None, event["latch_cause_after"]):
                        reason = "latch_changed"
                        break
                if event["charge_kind"] not in CHARGE_KINDS or event["latch_cause_after"] not in (None, *CAUSES) or event["call_id"] != admits[seq]["call_id"] or event["attempt"] != admits[seq]["attempt"]:
                    reason = "commit_schema"
                    break
                last_committed_latch = event["latch_cause_after"]
                if sum(event["diagnostics"]["per_root"].values()) != result["spent"]:
                    reason = "per_root_sum"
                    break
                result.update(event.get("diagnostics", {}))
            elif kind == "latch":
                if event["cause"] not in CAUSES:
                    reason = "latch_schema"
                    break
                if result["latch_cause"] is not None:
                    reason = "duplicate_latch"
                    break
                result["latch_cause"] = event["cause"]
                if event["at"] != result["spent"]:
                    reason = "latch_spent"
                    break
            elif kind == "fence":
                if event["cause"] not in CAUSES:
                    reason = "fence_schema"
                    break
                if result["admission_fence"] is not None:
                    reason = "duplicate_fence"
                    break
                result["admission_fence"] = event["cause"]
            elif kind == "budget_refused":
                if event["cause"] != result["latch_cause"] and event["cause"] != result["admission_fence"]:
                    reason = "refusal_cause"
                    break
                if event["spent"] != result["spent"] or event["remaining"] != result["limit"] - result["spent"]:
                    reason = "refusal_spent"
                    break
                code = "token_budget_insufficient" if event["cause"] == "insufficient" else "token_budget"
                if event["code"] != code or event["message"] != _refusal_message(result["limit"], result["spent"], event["cause"], event["reserved"]):
                    reason = "refusal_message"
                    break
                result["refusals"][event["code"]] = result["refusals"].get(event["code"], 0) + 1
                result["refusal_witnesses"].append({k: event[k] for k in ("call_id", "code", "cause", "message", "spent", "remaining", "reserved")})
            elif kind == "close_begin":
                if result["lifecycle"] != "open":
                    reason = "duplicate_close_begin"
                    break
                result["lifecycle"] = "closing"
            elif kind == "close":
                if result["lifecycle"] != "closing" or len(committed) != len(admits) or event["spent"] != result["spent"]:
                    reason = "invalid_close"
                    break
                if event["forced_settled"] != forced or last_committed_latch not in (None, result["latch_cause"]):
                    reason = "close_consistency"
                    break
                result.update(lifecycle="closed", forced_settled=event["forced_settled"])
                closed = True
            else:
                reason = "unknown_event"
                break
    except (KeyError, ValueError, TypeError, OSError) as error:
        reason = "invalid_journal"
    result["complete"] = closed and reason is None
    result["incomplete_reason"] = reason or (None if closed else "unsettled_admits" if len(committed) != len(admits) else "missing_close")
    try:
        snapshot = json.loads((directory / "budget.json").read_bytes())
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot is not an object")
        result["snapshot_stale"] = any(snapshot.get(k) != v for k, v in result.items() if k not in ("snapshot_stale", "attempts"))
    except (OSError, ValueError):
        result["snapshot_stale"] = True
    return result


@dataclasses.dataclass(frozen=True)
class Ending:
    ending: str
    accepted: bool
    grade: object
    cause: str | None = None


def classify_ending(record: dict | None, *, trial: str, config_hash: str, hsap_commit: str,
                    limit: int, grade: object = None, agent_error: str | None = None,
                    runner_error: str | None = None, native_terminal: str | None = None,
                    record_error: object = None, proxy_exit: object = None,
                    shadow_experiment: bool = False, profiles: dict | None = None) -> Ending:
    """One ordered classifier; grades survive even missing/crashed accounting."""
    def end(name, cause=None):
        return Ending(name, name in ("natural", "budget_exhausted"), grade, cause)
    expected = dict(trial=trial, config_hash=config_hash, hsap_commit=hsap_commit, limit=limit, rule=RULE)
    if record is not None and any(k in record and record[k] != v for k, v in expected.items()):
        return end("harness_failure", "identity_mismatch")
    if record is None or not record.get("complete") or record.get("lifecycle") != "closed" or any(k not in record for k in expected):
        return end("accounting_censored", "incomplete")
    if not record.get("enforce") and not shadow_experiment:
        return end("harness_failure", "shadow_record")
    if profiles is not None and record.get("profiles") != profiles:
        return end("harness_failure", "profile_mismatch")
    cause = record.get("latch_cause")
    if cause in ("bound_violated", "usage_unknown", "persistence_failed") or record.get("admission_fence") in ("bound_violated", "usage_unknown", "persistence_failed"):
        return end("accounting_censored", cause or record.get("admission_fence"))
    if record.get("bound_charged", 0) + record.get("partial_charged", 0) > 0.05 * record.get("spent", 0):
        return end("accounting_censored", "material_uncertainty")
    if runner_error or record_error or proxy_exit:
        return end("harness_failure", "runner_error")
    if cause in ("limit", "insufficient"):
        for witness in record.get("refusal_witnesses", []):
            message = witness.get("message")
            valid = (cause == "limit" and record["spent"] >= limit) or (cause == "insufficient" and witness.get("reserved", 0) > witness.get("remaining", limit))
            if valid and witness.get("cause") == cause and message and native_terminal and message in native_terminal:
                return end("budget_exhausted", cause)
        return end("harness_failure", "missing_budget_witness")
    if cause is not None:
        return end("harness_failure", cause)
    return end("natural", "agent_error" if agent_error else None)
