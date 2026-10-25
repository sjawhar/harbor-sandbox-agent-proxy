from __future__ import annotations

import pytest

pytest.importorskip("harbor")

from harbor_sandbox_agent_proxy.environment import RelayEnvironment


def test_environment_uses_the_public_wrapper_identifier() -> None:
    assert RelayEnvironment.type() == "sandbox-agent-proxy"
