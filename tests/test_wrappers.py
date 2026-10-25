from __future__ import annotations

import pytest

harbor = pytest.importorskip("harbor")

from harbor_sandbox_agent_proxy.agent import RelayAgent
from harbor_sandbox_agent_proxy.environment import RelayEnvironment


def test_wrappers_have_package_neutral_harbor_identifiers() -> None:
    assert RelayAgent.name() == "sandbox-agent-proxy"
    assert RelayEnvironment.type() == "sandbox-agent-proxy"
