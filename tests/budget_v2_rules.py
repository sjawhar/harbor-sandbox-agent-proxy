"""Executable counterexamples to the superseded v2 admission/commit/reader rules.

This is deliberately not a transport, persistence, or provider implementation.
Only the four rules exercised by regressions 71, 72, 74 and 79 live here.
"""
from __future__ import annotations

import json
from types import SimpleNamespace


class BudgetRefused(Exception):
    def __init__(self, code):
        self.code = code


def config(**overrides):
    return SimpleNamespace(**{"limit": 1_000_000, **overrides})


class TrialBudget:
    def __init__(self, config, directory, trial, **kwargs):
        self.config = config
        self.spent = 0
        self.held = 0
        self.checkpoints = {}
        self.paid_roots = set()
        self.attempts = []
        self.commits = []

    def prepare(self, dialect, model, body):
        # v2 subtracts already-paid root bytes, and checks only the upper n bound.
        root = body.get("system", "")
        root_bytes = len(json.dumps(root).encode()) if root in self.paid_roots else 0
        n = body.get("n", 1)
        output = body.get("max_tokens", body.get("max_completion_tokens", 10))
        return SimpleNamespace(body=body, root=root,
                               messages=body.get("messages", []),
                               reserved=len(json.dumps(body).encode()) - root_bytes + n * output)

    async def admit(self, plan, call_id, attempt=1):
        if self.spent + self.held + plan.reserved > self.config.limit:
            raise BudgetRefused("token_budget")
        self.held += plan.reserved
        item = SimpleNamespace(plan=plan, seq=len(self.attempts) + 1)
        self.attempts.append(item)
        return item

    def settle(self, attempt, measurement, **kwargs):
        key = (attempt.plan.root, json.dumps(attempt.plan.messages))
        self.paid_roots.add(attempt.plan.root)
        credit = self.checkpoints.get(key, 0)
        charge = max(0, measurement.n - credit) + measurement.o
        self.spent += charge
        self.held -= attempt.plan.reserved
        self.checkpoints[key] = min(self.checkpoints.get(key, measurement.n), measurement.n)
        self.commits.append({"admit_seq": attempt.seq, "charge": charge, "spent_after": self.spent})
        return {"charge": charge}

    async def close_out(self):
        return None


def Measurement(n, o, **kwargs):
    return SimpleNamespace(n=n, o=o)


def classify_ending(record, **kwargs):
    # The v2 common witness requires limit exhaustion even for natural success.
    return SimpleNamespace(ending="harness_failure", grade=kwargs.get("grade"), accepted=False)
