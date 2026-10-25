"""Package-neutral configuration for a sandbox-agent relay."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RelayConfig(BaseModel):
    """Configuration shared by the agent and environment wrappers."""

    model_config = ConfigDict(extra="forbid")

    declared_models: tuple[str, ...] = Field(min_length=1)
    port: int = Field(default=13131, ge=1, le=65535)
    polling_interval: float | None = Field(default=None, gt=0)
    tools: Literal["any", "bridged-only"] = "any"
    verifier_tools: Literal["any", "bridged-only"] | None = None
    permitted_tool_names: tuple[str, ...] = ()
    permitted_tool_prefixes: tuple[str, ...] = ()
    allowed_betas: tuple[str, ...] | None = None
    verifier_allowed_betas: tuple[str, ...] | None = None
    max_request_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    max_calls: int = Field(default=10_000, gt=0)
    max_recorded_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    conversation_path: PurePosixPath = PurePosixPath(
        "/tests/sandbox-agent-proxy/conversation.json"
    )
