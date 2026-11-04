"""Fail-open budget observation, separate from forwarding and ordinary recordings.

Only the enforcing TrialBudget can gate a request. This observer plans on copies,
never sends its planned bytes, never waits for quota and never cancels handlers.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path

from .budget import BudgetRefused, Plan, ShadowBudgetProposal, TokenBudgetConfig, TrialBudget, serialize
from .budget_usage import Measurement, UsageMeter
from .outcome import _redact

logger = logging.getLogger(__name__)


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
        try:
            self.config = TokenBudgetConfig.model_validate(self.config.model_dump(mode="json"))
            TokenBudgetConfig.model_validate({**self.config.model_dump(mode="json"), "enforce": True})
        except Exception as error:
            self.configuration_error = True
            self.log("would_refuse", "configuration", reason="invalid_budget_config", error_type=type(error).__name__)
        if ledger is not None:
            # Fatal recording callbacks belong to enforcing budgets only.
            ledger.on_error = None
            ledger.config = self.config
        elif not self.configuration_error:
            try:
                self.ledger = TrialBudget(self.config, directory, trial)
            except Exception as error:
                self.failure("open", error)

    def log(self, kind, stage, call_id=None, attempt=None, **fields):
        event = dict(version=1, kind=kind, stage=stage, trial=self.trial,
                     call_id=call_id, attempt=attempt, **fields)
        self.last_event = event
        if kind == "would_refuse":
            self.last_refusal = event
        try:
            safe = _redact(event, self.protected())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as stream:
                stream.write(serialize(safe) + b"\n")
        except Exception as error:
            # No second durable destination can be guaranteed when storage fails.
            self.log_error = type(error).__name__
            try:
                logger.warning("Shadow budget log unavailable: stage=%s reason=%s error_type=%s",
                               stage, fields.get("reason", kind), self.log_error)
            except Exception:
                pass

    def failure(self, stage, error, call_id=None, attempt=None):
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
            # Output injection / include_usage normalization is hypothetical only.
            # The actual sender retains the baseline body and JSON encoding.
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

    async def admit(self, plan, call_id, attempt):
        if plan is None:
            return None
        if self.ledger is None:
            self.log("would_refuse", "admit", call_id, attempt, reason="persistence_failed", cause="ledger_unavailable")
            return None
        item = None
        try:
            refusal = self._would_refuse(plan)
            if refusal is not None:
                self.log("would_refuse", "admit", call_id, attempt, reason=refusal[0], cause=refusal[1],
                         spent=self.ledger.spent, held=self.ledger.held, limit=self.config.limit, reserved=plan.reserved)
                if refusal[0] == "budget_closed":
                    return None
            # The observational ledger has enforce:false: this never waits for quota.
            item = await self.ledger.admit(plan, call_id, attempt)
            return item
        except Exception as error:
            self.failure("admit", error, call_id, attempt)
            return None
        finally:
            # An append can fail after creating an attempt; do not retain/cancel
            # the real request task in either the successful or failed case.
            for pending in self.ledger._attempts.values():
                if pending.call_id == call_id and pending.attempt == attempt:
                    pending.task = None

    def meter(self, dialect, branches, streaming, call_id, attempt):
        try:
            return ShadowMeter(self, dialect, branches, streaming, call_id, attempt)
        except Exception as error:
            self.failure("meter", error, call_id, attempt)
            return None

    def settle(self, item, usage, *, plan, call_id, attempt, sent, status, response_status, connect_failed, terminal_error):
        try:
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
        if self.ledger is None:
            return
        try:
            # Called after baseline proxy shutdown. Finish only shadow bookkeeping;
            # never wait for or signal an actual request task.
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
