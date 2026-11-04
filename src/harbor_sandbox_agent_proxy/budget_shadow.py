"""Fail-open observation with one ordered, bounded, host-owned writer.

Only the enforcing TrialBudget gates a request. Shadow senders enqueue without
waiting for storage; queue overflow loses observations, never model calls.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import queue
import threading
from concurrent.futures import Future
from pathlib import Path

from .budget import Attempt, BudgetRefused, Plan, ShadowBudgetProposal, TokenBudgetConfig, TrialBudget, serialize
from .budget_usage import Measurement, UsageMeter
from .outcome import _redact

logger = logging.getLogger(__name__)
_QUEUE_CAPACITY = 256
_CLOSE_TIMEOUT = 1.0


@dataclasses.dataclass
class _Observation:
    plan: Plan
    call_id: str
    attempt: int
    item: Attempt | None = None


class ShadowMeter(UsageMeter):
    """A broken/malformed measurement must not change the upstream response."""

    def __init__(self, observer, dialect, branches, streaming, call_id, attempt):
        super().__init__(dialect, branches, streaming)
        self.observer, self.call_id, self.attempt = observer, call_id, attempt

    def observe(self, body):
        try:
            super().observe(body)
        except Exception as error:
            self.observer.failure("meter", error, self.call_id, self.attempt)

    def feed(self, event):
        try:
            super().feed(event)
        except Exception as error:
            self.observer.failure("meter", error, self.call_id, self.attempt)

    def measurement(self):
        try:
            return super().measurement()
        except Exception as error:
            self.observer.failure("meter", error, self.call_id, self.attempt)
            return Measurement()


class ShadowBudget:
    def __init__(self, config: TokenBudgetConfig | ShadowBudgetProposal, directory: Path, trial: str, *, ledger=None, protected=lambda: ()):
        self.config = config.model_copy(deep=True, update={"enforce": False})
        self.trial = trial
        self.path = Path(directory) / "budget-shadow.jsonl"
        self.protected = protected
        self.last_event = None
        self.last_refusal = None
        self.log_error = None
        self.ledger = ledger
        self.configuration_error = False
        self.dropped_observations = 0
        self.observation_errors = 0
        self.close_timed_out = False
        self._queue = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self._ready = Future()
        self._done = Future()
        self._writer = threading.Thread(target=self._run, name="hsap-shadow-writer", daemon=True)
        try:
            self.config = TokenBudgetConfig.model_validate(self.config.model_dump(mode="json"))
            TokenBudgetConfig.model_validate({**self.config.model_dump(mode="json"), "enforce": True})
        except Exception as error:
            self.configuration_error = True
            self.log("would_refuse", "configuration", reason="invalid_budget_config", error_type=type(error).__name__)
        if ledger is not None:
            ledger.on_error = None
            ledger.config = self.config
        try:
            self._writer.start()
        except Exception as error:
            self.observation_errors += 1
            self.log_error = type(error).__name__
            self._queue.shutdown(immediate=True)
            self._ready.set_result(None)
            self._done.set_result(None)

    def _enqueue(self, operation, *args, **kwargs):
        try:
            self._queue.put_nowait((operation, args, kwargs))
            return True
        except (queue.Full, queue.ShutDown):
            # No logging, filesystem access or wait on the producer path.
            self.dropped_observations += 1
            return False

    def observation_status(self):
        """In-memory coverage; a pending writer is not a durable complete record."""
        return dict(complete=self._done.done() and not (
                        self.configuration_error or self.dropped_observations or
                        self.observation_errors or self.log_error or self.close_timed_out),
                    dropped_observations=self.dropped_observations,
                    observation_errors=self.observation_errors, log_error=self.log_error,
                    writer_pending=not self._done.done(), close_timed_out=self.close_timed_out)

    def _run(self):
        try:
            with asyncio.Runner() as runner:
                if self.ledger is None and not self.configuration_error:
                    try:
                        self.ledger = TrialBudget(self.config, self.path.parent, self.trial)
                    except Exception as error:
                        self.failure("open", error)
                self._ready.set_result(None)
                while True:
                    try:
                        operation, args, kwargs = self._queue.get()
                    except queue.ShutDown:
                        break
                    try:
                        if operation == "admit":
                            runner.run(self._admit(*args))
                        elif operation == "settle":
                            self._settle(*args, **kwargs)
                        else:
                            self._write_log(*args)
                    except Exception as error:
                        self.failure(operation, error)
                runner.run(self._close_ledger())
        except Exception as error:
            self.failure("writer", error)
        finally:
            self._queue.shutdown(immediate=True)
            if not self._ready.done():
                self._ready.set_result(None)
            status = self.observation_status()
            status["writer_pending"] = False
            status["complete"] = not (self.configuration_error or self.dropped_observations or
                                      self.observation_errors or self.log_error or self.close_timed_out)
            self.log("observation_status", "close", **status)
            self._done.set_result(None)

    def log(self, kind, stage, call_id=None, attempt=None, **fields):
        event = dict(version=1, kind=kind, stage=stage, trial=self.trial,
                     call_id=call_id, attempt=attempt, **fields)
        self.last_event = event
        if kind == "would_refuse":
            self.last_refusal = event
        if threading.current_thread() is self._writer:
            self._write_log(event)
        else:
            self._enqueue("log", event)

    def _write_log(self, event):
        try:
            safe = _redact(event, self.protected())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as stream:
                stream.write(serialize(safe) + b"\n")
        except Exception as error:
            # Host logging also stays on the writer: a slow handler is storage.
            self.log_error = type(error).__name__
            try:
                logger.warning("Shadow budget log unavailable: stage=%s reason=%s error_type=%s",
                               event["stage"], event.get("reason", event["kind"]), self.log_error)
            except Exception:
                pass

    def failure(self, stage, error, call_id=None, attempt=None):
        self.observation_errors += 1
        if isinstance(error, BudgetRefused):
            reason = error.code
        elif isinstance(error, OSError) or stage in ("open", "admit", "settle", "close"):
            reason = "persistence_failed"
        else:
            reason = "shadow_error"
        self.log("would_refuse", stage, call_id, attempt, reason=reason,
                 error_type=type(error).__name__)

    def surface(self, dialect, body, call_id):
        if self.configuration_error:
            self.log("would_refuse", "configuration", call_id, reason="invalid_budget_config")
            return False
        try:
            from .budget_surface import validate_surface
            validate_surface(dialect, body, self.config.max_schema_bytes)
            return True
        except Exception as error:
            self.failure("surface", error, call_id)
            return False

    def prepare(self, dialect, model, body, call_id) -> Plan | None:
        if self.configuration_error:
            self.log("would_refuse", "prepare", call_id, reason="invalid_budget_config")
            return None
        try:
            from .budget_surface import prepare
            plan = prepare(self.config, dialect, model, copy.deepcopy(body))
            # Only this hypothetical copy receives output/usage normalization.
            wire = serialize(body)
            profile = self.config.profiles[model]
            r_in = max(len(wire), profile.effective_input_tokens_max or 0) + profile.route_injected_tokens_max
            return dataclasses.replace(plan, wire=wire, r_in=r_in)
        except Exception as error:
            self.failure("prepare", error, call_id)
            return None

    def _would_refuse(self, plan):
        b = self.ledger
        if b.lifecycle != "open":
            return "budget_closed", "closed"
        cause = b.latch_cause or b.admission_fence
        if cause:
            return "token_budget", cause
        if b.spent + b.held + plan.reserved <= self.config.limit:
            return None
        if self.config.overshoot == "one_call" and b.held == 0 and b.spent < self.config.limit:
            return None
        if b.held:
            return "token_budget_wait", "in_flight_reservations"
        return "token_budget_insufficient", "insufficient"

    def admit(self, plan, call_id, attempt):
        if plan is None:
            return None
        observation = _Observation(plan, call_id, attempt)
        return observation if self._enqueue("admit", observation) else None

    async def _admit(self, observation):
        plan, call_id, attempt = observation.plan, observation.call_id, observation.attempt
        if self.ledger is None:
            self.log("would_refuse", "admit", call_id, attempt, reason="persistence_failed", cause="ledger_unavailable")
            return
        try:
            refusal = self._would_refuse(plan)
            if refusal is not None:
                self.log("would_refuse", "admit", call_id, attempt, reason=refusal[0], cause=refusal[1],
                         spent=self.ledger.spent, held=self.ledger.held, limit=self.config.limit, reserved=plan.reserved)
                if refusal[0] == "budget_closed":
                    return
            observation.item = await self.ledger.admit(plan, call_id, attempt)
        except Exception as error:
            self.failure("admit", error, call_id, attempt)
        finally:
            # The worker, never a real request handler, owns bookkeeping tasks.
            for pending in self.ledger._attempts.values():
                if pending.call_id == call_id and pending.attempt == attempt:
                    pending.task = None

    def meter(self, dialect, branches, streaming, call_id, attempt):
        try:
            return ShadowMeter(self, dialect, branches, streaming, call_id, attempt)
        except Exception as error:
            self.failure("meter", error, call_id, attempt)
            return None

    def settle(self, observation, usage, **fields):
        self._enqueue("settle", observation, usage, **fields)

    def _settle(self, observation, usage, *, plan, call_id, attempt, sent, status, response_status, connect_failed, terminal_error):
        try:
            item = observation.item if observation is not None else None
            if item is not None:
                try:
                    self.ledger.settle(item, usage, sent=sent, status=status,
                                       connect_failed=connect_failed, terminal_error=terminal_error)
                except Exception as error:
                    self.failure("settle", error, call_id, attempt)
                unsafe = getattr(item, "_unsafe", None)
                if unsafe is not None:
                    self.log("would_refuse", "settle", call_id, attempt, reason=unsafe)
            self.log("usage", "settle", call_id, attempt, n=usage.n, o=usage.o, w=usage.w,
                     terminal=usage.terminal, successful=response_status is not None and 200 <= response_status < 300,
                     upstream_status=status, response_status=response_status,
                     forwarded_bytes=None if plan is None else len(plan.wire),
                     r_in=None if plan is None else plan.r_in, r_out=None if plan is None else plan.r_out,
                     profile=None if plan is None else plan.profile,
                     model=None if plan is None else plan.model,
                     root_id=None if plan is None else plan.root_id)
        except Exception as error:
            self.failure("settle", error, call_id, attempt)

    async def close(self):
        # Drain only after baseline proxy shutdown, with a bounded wait. A stuck
        # filesystem may outlive the trial on this daemon; it cannot hold it open.
        self._queue.shutdown()
        try:
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(self._done)), _CLOSE_TIMEOUT)
        except TimeoutError:
            self.close_timed_out = True

    async def _close_ledger(self):
        if self.ledger is None:
            return
        try:
            for item in list(self.ledger._attempts.values()):
                item.task = None
                if not item.settled:
                    try:
                        self.ledger.settle(item, Measurement(), forced=True)
                    except Exception as error:
                        self.failure("close", error, item.call_id, item.attempt)
            await self.ledger.close_out()
        except Exception as error:
            self.failure("close", error)
        finally:
            try:
                self.ledger.release_writer()
            except Exception as error:
                self.failure("close", error)
