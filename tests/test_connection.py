"""Contracts for deriving relay connection variables from Harbor agents."""

from __future__ import annotations

import pytest
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.agents.model_connection import PROVIDERS, ModelConnectionSpec

from harbor_sandbox_agent_proxy.connection import (
    AgentConnection,
    agent_connection,
    served_connection_env,
)
from harbor_sandbox_agent_proxy.routing import BASE_URL_VARIABLES, KEY_VARIABLES


class _NoConnection(BaseInstalledAgent):
    MODEL_CONNECTION = None


class _UnsupportedProvider(BaseInstalledAgent):
    MODEL_CONNECTION = ModelConnectionSpec(default_provider="mistral")


@pytest.mark.parametrize(
    ("agent", "model_name", "expected"),
    [
        pytest.param(
            ClaudeCode,
            "anthropic/claude-x",
            AgentConnection(
                "anthropic",
                ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                ("ANTHROPIC_BASE_URL",),
            ),
            id="claude-code",
        ),
        pytest.param(
            Codex,
            "openai/gpt-x",
            AgentConnection(
                "openai",
                ("OPENAI_API_KEY",),
                ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            ),
            id="codex",
        ),
        pytest.param(
            GeminiCli,
            "google/gemini-2.0",
            AgentConnection(
                "google",
                ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
                ("GOOGLE_GEMINI_BASE_URL", "GOOGLE_BASE_URL"),
            ),
            id="gemini-cli",
        ),
    ],
)
def test_agent_connection_uses_an_agents_declared_connection_variables(
    agent: type[BaseInstalledAgent], model_name: str, expected: AgentConnection
) -> None:
    """A changed Harbor declaration must change the proxy variables it receives."""
    assert agent_connection(agent, model_name) == expected


@pytest.mark.parametrize(
    ("connection", "expected"),
    [
        pytest.param(
            AgentConnection(
                "anthropic",
                ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                ("ANTHROPIC_BASE_URL",),
            ),
            {
                "ANTHROPIC_API_KEY": "k",
                "ANTHROPIC_AUTH_TOKEN": "k",
                "ANTHROPIC_BASE_URL": "http://localhost:13131",
            },
            id="anthropic",
        ),
        pytest.param(
            AgentConnection(
                "openai",
                ("OPENAI_API_KEY",),
                ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            ),
            {
                "OPENAI_API_KEY": "k",
                "OPENAI_BASE_URL": "http://localhost:13131/v1",
                "OPENAI_API_BASE": "http://localhost:13131/v1",
            },
            id="openai",
        ),
        pytest.param(
            AgentConnection(
                "google",
                ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
                ("GOOGLE_GEMINI_BASE_URL", "GOOGLE_BASE_URL"),
            ),
            {
                "GEMINI_API_KEY": "k",
                "GOOGLE_API_KEY": "k",
                "GOOGLE_GENERATIVE_AI_API_KEY": "k",
                "GOOGLE_GEMINI_BASE_URL": "http://localhost:13131",
                "GOOGLE_BASE_URL": "http://localhost:13131",
            },
            id="google",
        ),
    ],
)
def test_proxy_env_sets_every_agent_connection_alias(
    connection: AgentConnection, expected: dict[str, str]
) -> None:
    """An alias omitted from the sandbox environment can make its agent bypass the relay."""
    assert connection.proxy_env(key="k", port=13131) == expected


def test_agent_connection_refuses_agents_without_a_model_connection() -> None:
    """An opaque agent declaration must fail before it can run outside the relay."""
    with pytest.raises(ValueError, match="declares no model connection"):
        _ = agent_connection(_NoConnection, None)


def test_agent_connection_refuses_unserved_providers() -> None:
    """A provider with no matching proxy dialect must not be offered a proxy URL."""
    with pytest.raises(ValueError, match="serves anthropic, google and openai"):
        _ = agent_connection(_UnsupportedProvider, "mistral/model")


def test_served_connection_env_sets_only_the_public_provider_variables() -> None:
    """Verifier routing needs exactly the variables the relay serves for all providers."""
    assert served_connection_env(key="k", port=13131) == {
        "ANTHROPIC_API_KEY": "k",
        "OPENAI_API_KEY": "k",
        "GOOGLE_API_KEY": "k",
        "GEMINI_API_KEY": "k",
        "GOOGLE_GENERATIVE_AI_API_KEY": "k",
        "ANTHROPIC_BASE_URL": "http://localhost:13131",
        "OPENAI_BASE_URL": "http://localhost:13131/v1",
        "OPENAI_API_BASE": "http://localhost:13131/v1",
        "GOOGLE_BASE_URL": "http://localhost:13131",
    }


def test_default_routing_variables_match_harbors_provider_table() -> None:
    """A Harbor provider-table update must not make the default host silently drift."""
    for provider in ("anthropic", "google", "openai"):
        assert KEY_VARIABLES[provider] == PROVIDERS[provider].api_key_envs
        assert BASE_URL_VARIABLES[provider] == PROVIDERS[provider].base_url_envs
