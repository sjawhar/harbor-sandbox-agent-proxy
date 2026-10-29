"""Stock Harbor Docker proof for the default host's Messages API relay."""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Self, override

import pytest
from harbor.agents.installed.base import BaseInstalledAgent

from harbor_sandbox_agent_proxy.testing.agents import (
    HOST_KEY,
    MessagesAgent,
    ResponsesAgent,
)
from tests.conftest import skip_if_no_docker


def test_e2e_model_agents_run_in_the_sandbox() -> None:
    assert issubclass(MessagesAgent, BaseInstalledAgent)
    assert issubclass(ResponsesAgent, BaseInstalledAgent)


_AGENTS = "harbor_sandbox_agent_proxy.testing.agents"


class _FakeUpstreamServer:
    """Host HTTP server that answers the shapes each sandbox agent requires."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, object]]] = []
        self._lock = threading.Lock()
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), self._handler()
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-upstream", daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exception: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    def _handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["content-length"])
                body = json.loads(self.rfile.read(length))
                assert isinstance(body, dict)
                with server._lock:
                    server.requests.append(
                        (
                            self.path,
                            {
                                name.lower(): value
                                for name, value in self.headers.items()
                            },
                            body,
                        )
                    )
                self._send_json(_response_for(self.path, body))

            def _send_json(self, payload: dict[str, object]) -> None:
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                _ = self.wfile.write(encoded)

            @override
            def log_message(self, format: str, *args: object) -> None:
                del format, args

        return Handler


def _response_for(path: str, body: dict[str, object]) -> dict[str, object]:
    if path == "/v1/messages":
        messages = body.get("messages")
        has_tool_result = isinstance(messages, list) and any(
            isinstance(message, dict)
            and isinstance(content := message.get("content"), list)
            and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
            for message in messages
        )
        if body.get("tools") and not has_tool_result:
            return {
                "id": "msg_tool",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_call",
                        "name": "mcp__e2e-tools__echo",
                        "input": {"message": "during-agent"},
                    }
                ],
                "model": body["model"],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        return {
            "id": "msg_final",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "relay complete"}],
            "model": body["model"],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    if path == "/responses":
        inputs = body.get("input")
        has_tool_result = isinstance(inputs, list) and any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in inputs
        )
        output = (
            [
                {
                    "type": "function_call",
                    "id": "function_id",
                    "call_id": "call_id",
                    "name": "echo",
                    "arguments": json.dumps({"message": "during-agent"}),
                    "status": "completed",
                }
            ]
            if body.get("tools") and not has_tool_result
            else [
                {
                    "type": "message",
                    "id": "message_id",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "relay complete",
                            "annotations": [],
                        }
                    ],
                }
            ]
        )
        return {
            "id": "response_tool" if not has_tool_result else "response_final",
            "object": "response",
            "created_at": 1_760_000_000,
            "status": "completed",
            "model": body["model"],
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "temperature": 1.0,
            "top_p": 1.0,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    raise AssertionError(f"unexpected upstream path {path}")


def _task(tmp_path: Path, *, host: str) -> Path:
    task = tmp_path / "task"
    environment = task / "environment"
    tests = task / "tests"
    environment.mkdir(parents=True)
    tests.mkdir()
    (environment / "Dockerfile").write_text(
        """\
FROM python:3.13-slim
RUN apt-get update \\
    && apt-get install -y --no-install-recommends util-linux \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mcp==1.27.0
RUN useradd --create-home --uid 1000 agent
COPY e2e_tools.py /opt/e2e_tools.py
""",
        encoding="utf-8",
    )
    (environment / "e2e_tools.py").write_text(
        """\
from mcp.server.fastmcp import FastMCP

server = FastMCP("e2e-tools")


@server.tool()
def echo(message: str) -> str:
    '''Return the supplied message.'''
    return message


if __name__ == "__main__":
    server.run()
""",
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Use the tool.\n", encoding="utf-8")
    metadata = (
        """

[metadata.sandbox_agent_proxy]
bridged_mcp_servers = [
  { name = "e2e-tools", command = "python3", args = ["/opt/e2e_tools.py"], require_proposal = false }
]
"""
        if host == "default"
        else ""
    )
    (task / "task.toml").write_text(
        f"""\
schema_version = "1.4"

[agent]
user = 1000{metadata}
""",
        encoding="utf-8",
    )
    (tests / "test.sh").write_text(
        """\
#!/bin/sh
set -eu
python3 - <<'PY'
import asyncio
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def tool_description() -> str:
    async with streamablehttp_client("http://localhost:13131/mcp/e2e-tools") as streams:
        read, write, _ = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools.tools[0].description


description = asyncio.run(tool_description())


request = Request(
    os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
    data=json.dumps(
        {
            "model": "anthropic/test-messages",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "propose the tool"}],
            "tools": [
                {
                    "name": "mcp__e2e-tools__echo",
                    "description": description,
                    "input_schema": {"type": "object"},
                }
            ],
        }
    ).encode(),
    headers={
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    },
    method="POST",
)
with urlopen(request, timeout=30) as response:
    assert response.status == 200


def expected_refusal(error):
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        return False
    if isinstance(error, BaseExceptionGroup):
        return all(expected_refusal(member) for member in error.exceptions)
    return (
        isinstance(error, Exception)
        and "not served outside the agent's turn" in str(error)
    )


async def main() -> None:
    try:
        async with streamablehttp_client(
            "http://localhost:13131/mcp/e2e-tools"
        ) as streams:
            read, write, _ = streams
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("echo", {"message": "after-agent"})
                if result.isError:
                    raise RuntimeError(result.content)
                raise SystemExit("the bridged tool was served after the agent phase")
    except BaseException as error:
        if not expected_refusal(error):
            raise
        Path("/logs/verifier/tool-error.txt").write_text(str(error))


asyncio.run(main())
PY
setpriv --reuid=1000 --regid=1000 --clear-groups cat /tests/test.sh >/dev/null 2>&1 && exit 1
setpriv --reuid=1000 --regid=1000 --clear-groups sh -c 'printf denied >/tests/agent-write' && exit 1
printf 'permission-and-tool-checks-passed\n' >/logs/verifier/e2e-checks.txt
printf '{"reward": 1}' >/logs/verifier/reward.json
""",
        encoding="utf-8",
    )
    return task


def _capture_rows(path: Path) -> Iterator[dict[str, object]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        assert isinstance(value, dict)
        yield value


def _job(task: Path, *, host: str, agent: str, model: str) -> dict[str, object]:
    relay = {"declared_models": [model, "anthropic/test-messages"]}
    environment_import_path = {
        "default": "harbor_sandbox_agent_proxy.environment:RelayEnvironment",
        "subclass": "harbor_sandbox_agent_proxy.testing.host:E2EHostEnvironment",
    }[host]
    return {
        "job_name": "sandbox-agent-proxy-e2e",
        "jobs_dir": str(task.parent / "jobs"),
        "agents": [
            {
                "import_path": "harbor_sandbox_agent_proxy.agent:RelayAgent",
                "kwargs": {
                    "agent": {"import_path": f"{_AGENTS}:{agent}"},
                    "relay": relay,
                },
            }
        ],
        "environment": {
            "import_path": environment_import_path,
            "kwargs": {
                "environment": {
                    "import_path": "harbor.environments.docker.docker:DockerEnvironment"
                },
                "relay": relay,
            },
        },
        "tasks": [{"path": str(task)}],
        "n_concurrent_trials": 1,
        "quiet": True,
    }


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize("host", ["default", "subclass"])
@pytest.mark.parametrize(
    ("agent", "model", "dialect"),
    [
        pytest.param("MessagesAgent", "anthropic/test-messages", "messages"),
        pytest.param("ResponsesAgent", "openai/test-responses", "responses"),
    ],
)
def test_stock_harbor_relays_model_calls_without_exposing_host_key(
    tmp_path: Path, host: str, agent: str, model: str, dialect: str
) -> None:
    skip_if_no_docker()
    task = _task(tmp_path, host=host)
    capture = tmp_path / "upstream.jsonl"
    job = _job(task, host=host, agent=agent, model=model)
    config = tmp_path / "job.json"
    config.write_text(json.dumps(job), encoding="utf-8")

    expected_paths = (
        ["/v1/messages"] * 3
        if dialect == "messages"
        else ["/responses", "/responses", "/v1/messages"]
    )
    expected_headers = (
        [{"x-api-key": HOST_KEY}] * 3
        if dialect == "messages"
        else [
            {"authorization": f"Bearer {HOST_KEY}"},
            {"authorization": f"Bearer {HOST_KEY}"},
            {"x-api-key": HOST_KEY},
        ]
    )
    with _FakeUpstreamServer() as upstream:
        subclass_env = (
            {
                "SANDBOX_AGENT_PROXY_E2E_UPSTREAM_URL": upstream.url,
                "SANDBOX_AGENT_PROXY_E2E_UPSTREAM_CAPTURE": str(capture),
            }
            if host == "subclass"
            else {}
        )
        result = subprocess.run(
            [
                "harbor",
                "run",
                "--config",
                str(config),
                "--yes",
                "--plugin",
                "sandbox-agent-proxy",
            ],
            cwd=tmp_path,
            env=os.environ
            | {
                "ANTHROPIC_API_KEY": HOST_KEY,
                "ANTHROPIC_BASE_URL": upstream.url,
                "OPENAI_API_KEY": HOST_KEY,
                "OPENAI_BASE_URL": upstream.url,
            }
            | subclass_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )

    assert result.returncode == 0, result.stderr
    assert [path for path, _, _ in upstream.requests] == expected_paths
    assert [
        {
            key: value
            for key, value in headers.items()
            if key in {"authorization", "x-api-key"}
        }
        for _, headers, _ in upstream.requests
    ] == expected_headers
    if host == "subclass":
        rows = list(_capture_rows(capture))
        assert [row["dialect"] for row in rows] == [dialect, dialect, "messages"]
        assert [row["headers"] for row in rows] == expected_headers

    if dialect == "responses":
        response_requests = [
            body for path, _, body in upstream.requests if path == "/responses"
        ]
        assert len(response_requests) == 2
        for request in response_requests:
            tools = request["tools"]
            assert isinstance(tools, list)
            assert {"type": "tool_search", "execution": "client"} in tools
            [namespace] = [tool for tool in tools if tool["type"] == "namespace"]
            assert namespace["name"] == "mcp__e2e_tools"
            assert [tool["name"] for tool in namespace["tools"]] == ["echo"]

    [outcome] = (tmp_path / "jobs").glob("**/sandbox-agent-proxy/outcome.json")
    payload = json.loads(outcome.read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "agent_calls": 2,
        "failed_calls": 0,
        "refused_calls": 0,
        "verifier_calls": 1,
    }
    assert any(
        event["event"] == "tool"
        and event["metadata"].get("sandbox_agent_proxy")
        == {
            "phase": "verifier",
            "server": "e2e-tools",
            "refused": "outside the agent's turn",
        }
        for event in payload["events"]
    )
    assert [
        event["metadata"]["sandbox_agent_proxy"]["phase"]
        for event in payload["events"]
        if event["event"] == "model"
    ] == ["agent", "agent", "verifier"]
    assert HOST_KEY not in outcome.read_text(encoding="utf-8")

    if host == "subclass":
        [events] = (tmp_path / "jobs").glob("**/e2e-host/events.jsonl")
        assert [event["event"] for event in _capture_rows(events)] == [
            "model",
            "tool",
            "model",
            "model",
            "tool",
        ]
        assert HOST_KEY not in events.read_text(encoding="utf-8")

    [sandbox_environment] = (tmp_path / "jobs").glob("**/agent/environment.json")
    assert HOST_KEY not in sandbox_environment.read_text(encoding="utf-8")
    [sandbox_env_command] = (tmp_path / "jobs").glob("**/agent/environment.txt")
    assert HOST_KEY not in sandbox_env_command.read_text(encoding="utf-8")
    [bridged_result] = (tmp_path / "jobs").glob("**/agent/bridged-tool.json")
    assert (
        "during-agent"
        in json.loads(bridged_result.read_text(encoding="utf-8"))["result"]
    )
    [checks] = (tmp_path / "jobs").glob("**/verifier/e2e-checks.txt")
    assert checks.read_text(encoding="utf-8") == "permission-and-tool-checks-passed\n"
