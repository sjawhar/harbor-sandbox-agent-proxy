"""Injected collaborators for routing, recording, secrets, and bridged tools."""

from __future__ import annotations

import dataclasses
import enum
import os
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, runtime_checkable
from urllib.parse import quote


class RequestDialect(enum.StrEnum):
    """The model API addressed by a sandbox proxy request."""

    MESSAGES = "messages"
    CHAT_COMPLETIONS = "chat-completions"
    RESPONSES = "responses"
    GEMINI = "gemini"


@dataclasses.dataclass(frozen=True)
class UpstreamTarget:
    """A fresh destination and authentication headers for one forwarded request."""

    url: str
    headers: Mapping[str, str]


@runtime_checkable
class Upstream(Protocol):
    """Resolve the destination for one model request."""

    async def resolve(self, *, dialect: RequestDialect, model: str) -> UpstreamTarget: ...


@runtime_checkable
class SecretSource(Protocol):
    """Values that must never be copied from host state into a sandbox."""

    def protected_values(self) -> Collection[str]: ...


@dataclasses.dataclass(frozen=True)
class BridgedMcpServer:
    """One sandbox command exposed to the model proxy as a bridged MCP server."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    exclude_tools: frozenset[str] = frozenset()
    require_proposal: bool = True


class BridgedToolServerProvider(Protocol):
    """Supplies task-specific bridged tool servers for a single trial."""

    async def servers_for(self, trial_name: str) -> Sequence[BridgedMcpServer]: ...


@dataclasses.dataclass(frozen=True)
class DefaultSecretSource:
    """Expose the public-provider host credentials only for redaction and confinement."""

    environment: Mapping[str, str]

    def protected_values(self) -> frozenset[str]:
        return frozenset(
            value
            for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
            if (value := self.environment.get(name))
        )


@dataclasses.dataclass(frozen=True)
class DefaultUpstream:
    """Route to public provider endpoints with a host key read at request time."""

    environment: Mapping[str, str] = dataclasses.field(default_factory=lambda: os.environ)

    async def resolve(self, *, dialect: RequestDialect, model: str) -> UpstreamTarget:
        match dialect:
            case RequestDialect.MESSAGES:
                return UpstreamTarget(
                    "https://api.anthropic.com/v1/messages",
                    {"x-api-key": self._required("ANTHROPIC_API_KEY")},
                )
            case RequestDialect.CHAT_COMPLETIONS:
                return UpstreamTarget(
                    "https://api.openai.com/v1/chat/completions",
                    {"authorization": f"Bearer {self._required('OPENAI_API_KEY')}"},
                )
            case RequestDialect.RESPONSES:
                return UpstreamTarget(
                    "https://api.openai.com/v1/responses",
                    {"authorization": f"Bearer {self._required('OPENAI_API_KEY')}"},
                )
            case RequestDialect.GEMINI:
                identifier = _bare_model(model)
                return UpstreamTarget(
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    + f"{quote(identifier, safe='._-')}:streamGenerateContent?alt=sse",
                    {"x-goog-api-key": self._required("GEMINI_API_KEY")},
                )

    def _required(self, name: str) -> str:
        value = self.environment.get(name)
        if not value:
            raise ValueError(f"{name} is not set in the host environment")
        return value


def _bare_model(model: str) -> str:
    """Remove an optional provider prefix from a model identifier."""
    provider, separator, identifier = model.partition("/")
    return identifier if separator and provider in {"anthropic", "openai", "google"} else model
