from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import skip_if_no_docker


@pytest.mark.e2e
@pytest.mark.docker
def test_stock_harbor_run_writes_a_redacted_outcome(tmp_path: Path) -> None:
    skip_if_no_docker()
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nRUN useradd --create-home --uid 1000 agent\n",
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Write the marker.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n[agent]\nuser = 1000\n', encoding="utf-8"
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\necho '{\"reward\": 1}' > /logs/verifier/reward.json\n",
        encoding="utf-8",
    )
    job = {
        "job_name": "sandbox-agent-proxy-e2e",
        "jobs_dir": str(tmp_path / "jobs"),
        "agents": [
            {
                "import_path": "harbor_sandbox_agent_proxy.agent:RelayAgent",
                "kwargs": {
                    "agent": {
                        "import_path": "harbor_sandbox_agent_proxy.testing.agents:SilentAgent"
                    },
                    "relay": {"declared_models": ["anthropic/claude-e2e"]},
                },
            }
        ],
        "environment": {
            "import_path": "harbor_sandbox_agent_proxy.environment:RelayEnvironment",
            "kwargs": {
                "environment": {
                    "import_path": "harbor.environments.docker.docker:DockerEnvironment"
                },
                "relay": {"declared_models": ["anthropic/claude-e2e"]},
            },
        },
        "tasks": [{"path": str(task)}],
        "n_concurrent_trials": 1,
        "quiet": True,
    }
    config = tmp_path / "job.json"
    config.write_text(json.dumps(job), encoding="utf-8")

    result = subprocess.run(
        [
            "harbor",
            "run",
            "--config",
            str(config),
            "--yes",
            "--plugin",
            "sandbox-agent-proxy",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    [outcome] = (tmp_path / "jobs").glob("**/sandbox-agent-proxy/outcome.json")
    payload = json.loads(outcome.read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "agent_calls": 0,
        "failed_calls": 0,
        "refused_calls": 0,
        "verifier_calls": 0,
    }
