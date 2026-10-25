"""Collection rules for tests that require the optional Harbor dependency."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from typing import Final

import pytest

HARBOR_TEST_MODULES: Final = (
    "test_trial_relay.py",
    "test_trial_relay_tool_phase.py",
    "test_relay_agent.py",
    "test_relay_environment.py",
    "test_plugin.py",
    "test_environment.py",
    "test_wrappers.py",
    "e2e",
)
RUNNER_VENV_TEST_MODULES: Final = ("test_runner_venv.py",)

collect_ignore: list[str] = (
    []
    if importlib.util.find_spec("harbor")
    else [*HARBOR_TEST_MODULES, *RUNNER_VENV_TEST_MODULES]
)

def skip_if_no_docker() -> None:
    """Skip a Docker-dependent test when Docker cannot be contacted."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")

    result = subprocess.run(
        ["docker", "version"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        pytest.skip("Docker is not available")
