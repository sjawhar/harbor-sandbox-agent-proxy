"""Derive proxy connection variables from Harbor agent declarations."""

from __future__ import annotations

import dataclasses
from typing import Final

from harbor.agents.base import BaseAgent
from harbor.agents.model_connection import (
    PROVIDERS,
    ModelConnectionSpec,
    resolve_model_connection,
)

PROXY_BASE_PATHS: Final = {
    "anthropic": "",
    "google": "",
    "openai": "/v1",
    "openai-responses": "/v1",
}


@dataclasses.dataclass(frozen=True)
class AgentConnection:
    """The model connection variables one Harbor agent declares."""

    provider: str
    api_key_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...]

    def proxy_env(self, *, key: str, port: int) -> dict[str, str]:
        """Point every declared key and base-URL alias to the sandbox relay."""
        base_url = f"http://localhost:{port}{PROXY_BASE_PATHS[self.provider]}"
        return {
            **{name: key for name in self.api_key_envs},
            **{name: base_url for name in self.base_url_envs},
        }


def agent_connection(
    agent_class: type[BaseAgent], model_name: str | None
) -> AgentConnection:
    """Return the relay environment required by the Harbor ``agent_class`` declaration."""
    spec = agent_class.MODEL_CONNECTION
    if spec is None:
        raise ValueError(f"{agent_class.__name__} declares no model connection")
    provider = _resolved_provider(spec, model_name)
    if provider not in PROXY_BASE_PATHS:
        raise ValueError(
            f"the relay serves anthropic, google and openai providers, not {provider!r}"
        )
    provider_access = PROVIDERS.get(provider)
    if provider_access is None:
        raise ValueError(
            f"the relay serves anthropic, google and openai providers, not {provider!r}"
        )
    return AgentConnection(
        provider,
        _names(spec.api_key_envs, provider_access.api_key_envs),
        _names(spec.base_url_envs, provider_access.base_url_envs),
    )


def served_connection_env(*, key: str, port: int) -> dict[str, str]:
    """Point every public-provider verifier variable at the sandbox relay."""
    environment: dict[str, str] = {}
    for provider in ("anthropic", "google", "openai"):
        access = PROVIDERS[provider]
        environment.update(
            AgentConnection(
                provider, access.api_key_envs, access.base_url_envs
            ).proxy_env(key=key, port=port)
        )
    return environment


def _resolved_provider(spec: ModelConnectionSpec, model_name: str | None) -> str | None:
    connection = resolve_model_connection(model_name, spec, lambda *_names: None)
    return connection.provider


def _names(
    agent_names: tuple[str, ...], provider_names: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*agent_names, *provider_names)))
