"""Stock Harbor Docker proof for both supported model request dialects."""

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

from harbor_sandbox_agent_proxy.testing.agents import HOST_KEY
from tests.conftest import skip_if_no_docker

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
    if path == "/messages":
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
        if body.get("tools") and not has_tool_result:
            return {
                "id": "response_tool",
                "object": "response",
                "model": body["model"],
                "output": [
                    {
                        "type": "function_call",
                        "id": "function_id",
                        "call_id": "call_id",
                        "name": "echo",
                        "arguments": json.dumps({"message": "during-agent"}),
                        "status": "completed",
                    }
                ],
            }
        return {
            "id": "response_final",
            "object": "response",
            "model": body["model"],
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "relay complete"}],
                }
            ],
        }
    raise AssertionError(f"unexpected upstream path {path}")


def _task(tmp_path: Path, *, leak_host_environment: bool = False) -> Path:
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
    if leak_host_environment:
        (environment / "docker-compose.yaml").write_text(
            """\
services:
  main:
    environment:
      - OPENAI_API_KEY
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
    (task / "task.toml").write_text(
        """\
schema_version = "1.4"

[agent]
user = 1000
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


async def main() -> None:
    async with streamablehttp_client("http://localhost:13131/mcp/e2e-tools") as streams:
        read, write, _ = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                result = await session.call_tool("echo", {"message": "after-agent"})
            except Exception as error:
                Path("/logs/verifier/tool-error.txt").write_text(str(error))
                if "not served outside the agent phase" not in str(error):
                    raise
                return
            if result.isError:
                if "not served outside the agent phase" in str(result.content):
                    return
                raise RuntimeError(f"unexpected post-agent tool error: {result.content}")
            raise SystemExit("the bridged tool was served after the agent phase")


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


def _job(task: Path, *, agent: str, model: str) -> dict[str, object]:
    return {
        "job_name": "sandbox-agent-proxy-e2e",
        "jobs_dir": str(task.parent / "jobs"),
        "agents": [
            {
                "import_path": "harbor_sandbox_agent_proxy.agent:RelayAgent",
                "kwargs": {
                    "agent": {"import_path": f"{_AGENTS}:{agent}"},
                    "relay": {"declared_models": [model, "anthropic/test-messages"]},
                    "upstream": f"{_AGENTS}:FakeUpstream",
                    "secrets": f"{_AGENTS}:FakeSecretSource",
                    "bridged_tool_server_provider": (
                        f"{_AGENTS}:E2EBridgedToolServerProvider"
                    ),
                },
            }
        ],
        "environment": {
            "import_path": "harbor_sandbox_agent_proxy.environment:RelayEnvironment",
            "kwargs": {
                "environment": {
                    "import_path": "harbor.environments.docker.docker:DockerEnvironment"
                },
                "relay": {"declared_models": [model, "anthropic/test-messages"]},
                "secrets": f"{_AGENTS}:FakeSecretSource",
            },
        },
        "tasks": [{"path": str(task)}],
        "n_concurrent_trials": 1,
        "quiet": True,
    }


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize(
    ("agent", "model", "dialect"),
    [
        pytest.param("MessagesAgent", "anthropic/test-messages", "messages"),
        pytest.param("ResponsesAgent", "openai/test-responses", "responses"),
    ],
)
def test_stock_harbor_relays_model_calls_without_exposing_host_key(
    tmp_path: Path, agent: str, model: str, dialect: str
) -> None:
    skip_if_no_docker()
    task = _task(tmp_path)
    capture = tmp_path / "upstream.jsonl"
    job = _job(task, agent=agent, model=model)
    config = tmp_path / "job.json"
    config.write_text(json.dumps(job), encoding="utf-8")

    with _FakeUpstreamServer() as upstream:
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
                "OPENAI_API_KEY": HOST_KEY,
                "SANDBOX_AGENT_PROXY_E2E_UPSTREAM_URL": upstream.url,
                "SANDBOX_AGENT_PROXY_E2E_UPSTREAM_CAPTURE": str(capture),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )

    assert result.returncode == 0, result.stderr
    rows = list(_capture_rows(capture))
    assert [row["dialect"] for row in rows] == [dialect, dialect, "messages"]
    assert [row["headers"] for row in rows] == [{"x-api-key": HOST_KEY}] * 3
    assert [headers["x-api-key"] for _, headers, _ in upstream.requests] == [
        HOST_KEY
    ] * 3
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
        event["kind"] == "refused-tool-call"
        and event["phase"] == "verifier"
        and event["data"]["reason"] == "outside-agent-phase"
        for event in payload["events"]
    )
    assert HOST_KEY not in outcome.read_text(encoding="utf-8")

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


@pytest.mark.e2e
@pytest.mark.docker
def test_docker_compose_host_environment_reference_is_refused(tmp_path: Path) -> None:
    """A task compose variable can inherit a host key outside wrapper-owned env maps."""
    skip_if_no_docker()
    task = _task(tmp_path, leak_host_environment=True)
    config = tmp_path / "job.json"
    config.write_text(
        json.dumps(
            _job(
                task,
                agent="MessagesAgent",
                model="anthropic/test-messages",
            )
        ),
        encoding="utf-8",
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
        env=os.environ | {"OPENAI_API_KEY": HOST_KEY},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    [trial_result] = (tmp_path / "jobs").glob("**/task__*/result.json")
    exception = json.loads(trial_result.read_text(encoding="utf-8"))["exception_info"]
    assert exception["exception_type"] == "RunnerSecretExposureError"
    assert "OPENAI_API_KEY" in exception["exception_message"]
    assert HOST_KEY not in exception["exception_message"]
