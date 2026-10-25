"""Compatibility contracts with the supported Inspect releases."""

from __future__ import annotations

import inspect

from harbor_sandbox_agent_proxy import (
    _inspect_compat as compat,
)
from harbor_sandbox_agent_proxy import (
    conversation,
    proxy,
    sandbox_binding,
)


def test_inspect_proxy_seams_accept_the_package_calls() -> None:
    """Fail when an Inspect upgrade moves a private seam this package needs."""
    inspect.signature(compat.sandbox_service).bind(
        name=compat.MODEL_SERVICE,
        methods={},
        until=lambda: False,
        sandbox=object(),
        instance="proxy-test",
        polling_interval=None,
        started=None,
        requires_python=False,
    )
    inspect.signature(compat.SandboxAgentBridge.__init__).bind(
        object(),
        state=object(),
        filter=None,
        retry_refusals=None,
        compaction=None,
        port=13131,
        model=None,
    )
    inspect.signature(compat.ExecRemoteStreamingOptions).bind(
        concurrency=False,
        user="root",
        env={},
        poll_timeout=600,
        poll_timeout_recovery=None,
    )
    inspect.signature(compat.sandbox_with_injected_tools).bind(sandbox=object())


def test_inspect_side_public_api_is_explicit() -> None:
    assert proxy.__all__ == [
        "sandbox_model_proxy",
        "ModelProxy",
        "ModelProxyError",
        "ModelProxyMethod",
    ]
    assert conversation.__all__ == [
        "BridgeConversation",
        "BridgeConversationSelector",
    ]
    assert sandbox_binding.__all__ == ["bind_sandbox_environment"]
