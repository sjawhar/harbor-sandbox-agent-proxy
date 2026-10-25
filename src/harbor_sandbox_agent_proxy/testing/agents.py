"""Small Harbor agents used by the package's stock-Docker end-to-end test."""

from __future__ import annotations

from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SilentAgent(BaseAgent):
    """An installed-shaped test agent that exercises wrapper lifecycle without a model call."""

    @staticmethod
    @override
    def name() -> str:
        return "sandbox-agent-proxy-silent"

    @override
    def version(self) -> str | None:
        return "1"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        del instruction, environment, context
