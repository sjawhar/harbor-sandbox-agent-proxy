"""Contracts for task-provided bridged MCP server metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from harbor_sandbox_agent_proxy.task_metadata import read_bridged_servers


def _task_dir(tmp_path: Path, task_toml: str) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    return task_dir


def test_read_bridged_servers_reads_the_sandbox_agent_proxy_table(
    tmp_path: Path,
) -> None:
    """A task-declared MCP server must reach the relay with its full configuration."""
    task_dir = _task_dir(
        tmp_path,
        "\n".join(
            (
                "[[metadata.sandbox_agent_proxy.bridged_mcp_servers]]",
                'name = "task-tools"',
                'command = "python3"',
                'args = ["/opt/task_tools.py"]',
                'exclude_tools = ["admin"]',
                "require_proposal = false",
                "",
            )
        ),
    )

    servers = read_bridged_servers(task_dir)

    assert [
        (
            server.name,
            server.command,
            server.args,
            server.exclude_tools,
            server.require_proposal,
        )
        for server in servers
    ] == [("task-tools", "python3", ["/opt/task_tools.py"], ["admin"], False)]


def test_read_bridged_servers_returns_no_servers_without_the_metadata_table(
    tmp_path: Path,
) -> None:
    """Tasks without relay metadata should retain Harbor's no-bridged-tools default."""
    assert read_bridged_servers(_task_dir(tmp_path, "[metadata]\n")) == ()


def test_read_bridged_servers_refuses_an_unknown_server_key(tmp_path: Path) -> None:
    """A misspelled bridge setting must not silently leave a task differently configured."""
    task_dir = _task_dir(
        tmp_path,
        "\n".join(
            (
                "[[metadata.sandbox_agent_proxy.bridged_mcp_servers]]",
                'name = "task-tools"',
                'command = "python3"',
                "require_proposals = false",
                "",
            )
        ),
    )

    with pytest.raises(ValueError):
        _ = read_bridged_servers(task_dir)
