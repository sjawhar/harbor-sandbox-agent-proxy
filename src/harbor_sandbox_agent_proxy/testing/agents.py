"""Small Harbor agents used by the package Docker end-to-end test."""

from __future__ import annotations

import shlex
from typing import Final, override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HOST_KEY: Final = "host-key-not-for-sandbox"
_MCP_URL: Final = "http://localhost:13131/mcp/e2e-tools"

_AGENT_PROGRAM: Final = r"""
import asyncio
import json
import os
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def post(url, headers, body):
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


async def call_tool():
    async with streamablehttp_client(os.environ["E2E_MCP_URL"]) as streams:
        read, write, _ = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("echo", {"message": "during-agent"})
            if result.isError:
                raise RuntimeError(result.content)
            return result.content[0].text


async def tool_description():
    async with streamablehttp_client(os.environ["E2E_MCP_URL"]) as streams:
        read, write, _ = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools.tools[0].description


async def messages():
    description = await tool_description()
    base = os.environ["ANTHROPIC_BASE_URL"]
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }
    first = {
        "model": "anthropic/test-messages",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "use the tool"}],
        "tools": [
            {
                "name": "mcp__e2e-tools__echo",
                "description": description,
                "input_schema": {"type": "object"},
            }
        ],
    }
    reply = post(f"{base}/v1/messages", headers, first)
    call = next(item for item in reply["content"] if item["type"] == "tool_use")
    output = await call_tool()
    second = {
        **first,
        "messages": [
            *first["messages"],
            {"role": "assistant", "content": reply["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": output,
                    }
                ],
            },
        ],
    }
    post(f"{base}/v1/messages", headers, second)
    return output


async def responses():
    description = await tool_description()
    base = os.environ["OPENAI_BASE_URL"]
    headers = {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    first = {
        "model": "openai/test-responses",
        "input": [{"role": "user", "content": "use the tool"}],
        "tools": [
            {"type": "tool_search", "execution": "client"},
            {
                "type": "namespace",
                "name": "mcp__e2e_tools",
                "description": "",
                "tools": [
                    {
                        "type": "function",
                        "name": "echo",
                        "description": description,
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ],
    }
    reply = post(f"{base}/responses", headers, first)
    call = next(item for item in reply["output"] if item["type"] == "function_call")
    output = await call_tool()
    second = {
        **first,
        "input": [
            *first["input"],
            *reply["output"],
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": output,
            },
        ],
    }
    post(f"{base}/responses", headers, second)
    return output


async def main():
    output = await (messages() if os.environ["E2E_DIALECT"] == "messages" else responses())
    Path("/logs/agent").mkdir(parents=True, exist_ok=True)
    Path("/logs/agent/environment.json").write_text(json.dumps(dict(os.environ), sort_keys=True))
    Path("/logs/agent/environment.txt").write_text(
        subprocess.run(["env"], capture_output=True, check=True, text=True).stdout
    )
    Path("/logs/agent/bridged-tool.json").write_text(json.dumps({"result": output}))


asyncio.run(main())
"""


class _E2EAgent(BaseInstalledAgent):
    dialect: str

    @override
    def version(self) -> str | None:
        return "1"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        del environment

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        del instruction, context
        result = await self.exec_as_agent(
            environment,
            f"python3 -c {shlex.quote(_AGENT_PROGRAM)}",
            env={"E2E_DIALECT": self.dialect, "E2E_MCP_URL": _MCP_URL},
        )
        if result.return_code != 0:
            detail = result.stderr or result.stdout or "e2e agent failed"
            _ = (self.logs_dir / "e2e-agent-error.txt").write_text(
                detail, encoding="utf-8"
            )
            raise RuntimeError(detail)


class MessagesAgent(_E2EAgent):
    """A Claude Code-shaped agent that uses the Messages API and MCP tool names."""

    MODEL_CONNECTION = ModelConnectionSpec(
        default_provider="anthropic",
        api_key_envs=("ANTHROPIC_API_KEY",),
        base_url_envs=("ANTHROPIC_BASE_URL",),
    )
    dialect = "messages"

    @staticmethod
    @override
    def name() -> str:
        return "sandbox-agent-proxy-e2e-messages"


class ResponsesAgent(_E2EAgent):
    """A Codex-shaped agent that uses the Responses API and MCP tool names."""

    MODEL_CONNECTION = ModelConnectionSpec(default_provider="openai")
    dialect = "responses"

    @staticmethod
    @override
    def name() -> str:
        return "sandbox-agent-proxy-e2e-responses"
