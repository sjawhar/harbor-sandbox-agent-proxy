"""Bind caller-managed sandboxes to Inspect's current-context contract."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from typing import final, override

from inspect_ai.util import SandboxEnvironment

from ._inspect_compat import (
    SandboxEnvironmentProxy,
    sandbox_default_context_var,
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)

__all__ = ["bind_sandbox_environment"]


@final
class _SingleSandboxEnvironments(Mapping[str, SandboxEnvironment]):
    """Keep the one-sandbox lookup behavior used by Inspect at the floor version."""

    def __init__(self, name: str, environment: SandboxEnvironment) -> None:
        self._name = name
        self._environment = environment

    @override
    def __getitem__(self, key: str) -> SandboxEnvironment:
        return self._environment

    @override
    def __iter__(self) -> Iterator[str]:
        yield self._name

    @override
    def __len__(self) -> int:
        return 1


@contextmanager
def bind_sandbox_environment(
    environment: SandboxEnvironment, *, name: str = "default"
) -> Generator[SandboxEnvironment]:
    """Bind ``environment`` as the current context's sandbox.

    Makes ``environment`` resolvable via :func:`inspect_ai.util.sandbox` and
    :func:`inspect_ai.util.sandbox_with`, and usable by anything that reads the
    current sandbox context (for example ``mcp_tools()``, or an agent that calls
    ``sandbox()`` internally), for the duration of the ``with`` block -- without
    running inside Inspect's own eval loop and sandbox provisioning (the
    ``@sandboxenv`` registry, ``sample_init``/``sample_cleanup``). This is for a
    caller that constructs and lifecycle-manages its own ``SandboxEnvironment`` (for
    example a harness driving a sandbox through a provider Inspect does not provision)
    and needs it to work as the current sandbox for code written against Inspect's own
    sample-context contract.

    ``environment`` is wrapped to record its commands and file transfers as Inspect
    ``SandboxEvent``s, exactly as a task's own sandboxes are wrapped when a sample
    starts (``init_sandbox_environments_sample``); within an active sample transcript
    this adds the same events a task-declared sandbox would produce, and recording is
    a no-op with no active transcript.

    This always binds exactly one environment, as the sole bound name and the block's
    default. A caller that needs more than one named sandbox resolvable at once should
    construct and bind a single environment of its own that dispatches by name, rather
    than nesting calls (nesting replaces, not merges, the prior binding).

    Args:
      environment: The sandbox environment to bind.
      name: The name ``environment`` is resolvable under via ``sandbox(name)``. It is
        also the block's default, so ``sandbox()`` with no argument resolves it too.

    Yields:
      ``environment``, wrapped for event recording.
    """
    proxy = SandboxEnvironmentProxy(environment)
    environments_token = sandbox_environments_context_var.set(
        _SingleSandboxEnvironments(name, proxy)
    )
    with_environments_token = sandbox_with_environments_context_var.set({})
    default_token = sandbox_default_context_var.set(name)
    try:
        yield proxy
    finally:
        sandbox_default_context_var.reset(default_token)
        sandbox_with_environments_context_var.reset(with_environments_token)
        sandbox_environments_context_var.reset(environments_token)
