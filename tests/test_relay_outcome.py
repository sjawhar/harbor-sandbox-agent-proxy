from __future__ import annotations

import json
from pathlib import Path

from harbor_sandbox_agent_proxy.outcome import (
    JsonOutcomeSink,
    RelayOutcome,
    TrialEvent,
    TrialIdentity,
)


def _outcome(
    *, mode: str, runner_atif: Path | None, sha256: str | None
) -> RelayOutcome:
    return RelayOutcome(
        mode=mode,  # type: ignore[arg-type]
        calls=[],
        events=[],
        conversation=None,
        answered_agent_calls=0,
        counts={
            "agent_calls": 0,
            "verifier_calls": 0,
            "failed_calls": 0,
            "refused_calls": 0,
        },
        dropped_betas=frozenset({"experimental"}) if mode == "relay" else frozenset(),
        refusal_reason=(
            "the agent made no answered model call through the relay"
            if mode == "relay"
            else None
        ),
        proxy_exit=None,
        runner_atif=runner_atif,
        runner_atif_sha256=sha256,
        record_error=None,
    )


def test_relay_outcome_keeps_relay_and_runner_loop_results_distinct() -> None:
    relay = _outcome(mode="relay", runner_atif=None, sha256=None)
    runner_loop = _outcome(
        mode="runner-loop",
        runner_atif=Path("/tmp/runner/trajectory.json"),
        sha256="a" * 64,
    )

    assert relay.mode == "relay"
    assert relay.refusal_reason is not None
    assert runner_loop.mode == "runner-loop"
    assert runner_loop.runner_atif == Path("/tmp/runner/trajectory.json")
    assert runner_loop.runner_atif_sha256 == "a" * 64


async def test_json_outcome_sink_writes_redacted_package_records(
    tmp_path: Path,
) -> None:
    secret = "host-key-not-for-sandbox"
    trial = TrialIdentity(name="trial-1", directory=tmp_path)
    sink = JsonOutcomeSink(trial, protected_values=(secret,))
    event = TrialEvent(
        kind="caller-key-minted",
        phase="agent",
        data={"fingerprint": "not-a-secret"},
    )
    await sink.record(event)

    outcome = _outcome(mode="relay", runner_atif=None, sha256=None)
    await sink.close(
        RelayOutcome(
            **{
                **outcome.__dict__,
                "events": [event],
            }
        )
    )

    written = json.loads(
        (tmp_path / "sandbox-agent-proxy" / "outcome.json").read_text(encoding="utf-8")
    )
    assert written["trial"] == "trial-1"
    assert written["mode"] == "relay"
    assert written["events"] == [
        {
            "kind": "caller-key-minted",
            "phase": "agent",
            "data": {"fingerprint": "not-a-secret"},
        }
    ]
