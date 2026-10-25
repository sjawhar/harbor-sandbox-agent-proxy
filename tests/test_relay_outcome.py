from __future__ import annotations

import json

from harbor_sandbox_agent_proxy.outcome import (
    JsonOutcomeSink,
    RelayOutcome,
    TrialEvent,
    TrialIdentity,
)


async def test_json_outcome_sink_writes_redacted_package_records(tmp_path) -> None:
    secret = "host-key-not-for-sandbox"
    trial = TrialIdentity(name="trial-1", directory=tmp_path)
    sink = JsonOutcomeSink(trial, protected_values=(secret,))
    event = TrialEvent(
        kind="caller-key-minted",
        phase="agent",
        data={"fingerprint": "not-a-secret"},
    )
    await sink.record(event)

    await sink.close(
        RelayOutcome(
            trial=trial,
            calls=(
                {
                    "phase": "agent",
                    "model": "claude",
                    "detail": f"Bearer {secret}",
                },
            ),
            events=(event,),
            counts={"agent_calls": 1},
            conversation=None,
            refusal_reason=None,
        )
    )

    outcome = json.loads(
        (tmp_path / "sandbox-agent-proxy" / "outcome.json").read_text(encoding="utf-8")
    )
    assert outcome["trial"] == "trial-1"
    assert outcome["calls"][0]["detail"] == "Bearer [REDACTED]"
    assert outcome["events"] == [
        {
            "kind": "caller-key-minted",
            "phase": "agent",
            "data": {"fingerprint": "not-a-secret"},
        }
    ]
