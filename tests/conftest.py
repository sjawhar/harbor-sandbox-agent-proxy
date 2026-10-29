"""Collection rules for tests that require the optional Harbor dependency."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from typing import Final

import pytest

HARBOR_TEST_MODULES: Final = (
    "test_connection.py",
    "test_environment_host.py",
    "test_trial_relay.py",
    "test_trial_relay_tool_phase.py",
    "test_relay_agent.py",
    "test_relay_environment.py",
    "test_wrappers.py",
    "e2e",
)
RUNNER_VENV_TEST_MODULES: Final = ("test_runner_venv.py",)

collect_ignore: list[str] = (
    []
    if importlib.util.find_spec("harbor")
    else [*HARBOR_TEST_MODULES, *RUNNER_VENV_TEST_MODULES]
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit opt-in required for Docker end-to-end tests."""
    parser.addoption("--e2e", action="store_true", help="run end-to-end tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip Docker end-to-end tests unless the caller explicitly requested them."""
    if config.getoption("--e2e"):
        return
    skip = pytest.mark.skip(reason="pass --e2e to run end-to-end tests")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


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
