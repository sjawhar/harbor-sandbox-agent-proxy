"""Task metadata for bridged sandbox MCP servers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import ClassVar, Final

import pydantic

TABLE: Final = "sandbox_agent_proxy"


class BridgedMcpServer(pydantic.BaseModel):
    """One task-provided MCP server exposed by the relay proxy."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")

    name: str
    command: str
    args: list[str] = pydantic.Field(default_factory=list)
    exclude_tools: list[str] = pydantic.Field(default_factory=list)
    require_proposal: pydantic.StrictBool = True


class _RelayMetadata(pydantic.BaseModel):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")

    bridged_mcp_servers: list[BridgedMcpServer] = pydantic.Field(default_factory=list)
    token_budget_required: pydantic.StrictBool = False
    token_budget_limit: pydantic.StrictInt | None = pydantic.Field(default=None, gt=0)


class _Metadata(pydantic.BaseModel):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="ignore")

    sandbox_agent_proxy: _RelayMetadata = pydantic.Field(default_factory=_RelayMetadata)


class _TaskToml(pydantic.BaseModel):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="ignore")

    metadata: _Metadata = pydantic.Field(default_factory=_Metadata)


def read_bridged_servers(task_dir: Path) -> tuple[BridgedMcpServer, ...]:
    """Read optional ``[metadata.sandbox_agent_proxy]`` bridged MCP server records."""
    task_toml = task_dir / "task.toml"
    parsed = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    return tuple(
        _TaskToml.model_validate(
            parsed
        ).metadata.sandbox_agent_proxy.bridged_mcp_servers
    )


def check_token_budget(task_dir: Path, budget) -> None:
    """A trusted task can require an enforcing, exactly matching token allocation."""
    if not (task_dir / "task.toml").is_file():
        return
    parsed = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    metadata = _TaskToml.model_validate(parsed).metadata.sandbox_agent_proxy
    if metadata.token_budget_required and (
        budget is None or not budget.enforce or metadata.token_budget_limit is None
        or budget.limit != metadata.token_budget_limit
    ):
        raise ValueError("task requires an enforcing token budget with its declared limit")
