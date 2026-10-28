"""Contracts for host-independent model routing."""

from __future__ import annotations

import pytest

from harbor_sandbox_agent_proxy.routing import (
    ModelCall,
    RequestDialect,
    Route,
    RouteRefused,
    default_protected_values,
    default_route,
    endpoint_url,
    match_declared,
)


def _call(
    dialect: RequestDialect,
    model: str | None,
    *declared_models: str,
) -> ModelCall:
    return ModelCall(dialect, model, declared_models)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param(
            _call(RequestDialect.MESSAGES, "claude-x", "anthropic/claude-x"),
            "claude-x",
            id="bare-name-matches-provider-declaration",
        ),
        pytest.param(
            _call(RequestDialect.CHAT_COMPLETIONS, "gpt-x", "openai/gpt-x"),
            "gpt-x",
            id="bare-name-matches-openai-declaration",
        ),
    ],
)
def test_match_declared_returns_the_wire_model_for_one_matching_declaration(
    call: ModelCall, expected: str
) -> None:
    """A matching declaration permits the model name the provider endpoint expects."""
    assert match_declared(call) == expected


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            _call(RequestDialect.MESSAGES, "openai/claude-x", "anthropic/claude-x"),
            "is not one the launch declared for the anthropic endpoint",
            id="other-provider-prefix",
        ),
        pytest.param(
            _call(RequestDialect.MESSAGES, "claude-x"),
            "is not one the launch declared",
            id="no-declarations",
        ),
        pytest.param(
            _call(
                RequestDialect.MESSAGES,
                "claude-x",
                "claude-x",
                "anthropic/claude-x",
            ),
            "matches multiple launch declarations",
            id="multiple-declarations",
        ),
        pytest.param(
            _call(RequestDialect.MESSAGES, None, "anthropic/claude-x"),
            "is not one the launch declared",
            id="missing-model",
        ),
    ],
)
def test_match_declared_refuses_ambiguous_or_undeclared_models(
    call: ModelCall, message: str
) -> None:
    """A changed declaration check must not forward an unapproved model."""
    with pytest.raises(RouteRefused, match=message):
        _ = match_declared(call)


@pytest.mark.parametrize(
    ("base_url", "dialect", "model", "expected"),
    [
        pytest.param(
            "https://gw.test/anthropic",
            RequestDialect.MESSAGES,
            "claude-x",
            "https://gw.test/anthropic/v1/messages",
            id="messages",
        ),
        pytest.param(
            "https://gw.test/openai",
            RequestDialect.CHAT_COMPLETIONS,
            "gpt-x",
            "https://gw.test/openai/chat/completions",
            id="chat-completions",
        ),
        pytest.param(
            "https://gw.test/openai/v1",
            RequestDialect.RESPONSES,
            "gpt-x",
            "https://gw.test/openai/v1/responses",
            id="responses",
        ),
        pytest.param(
            "https://gw.test/google-ai",
            RequestDialect.GEMINI,
            "gemini-2.0",
            "https://gw.test/google-ai/v1beta/models/gemini-2.0:streamGenerateContent?alt=sse",
            id="gemini",
        ),
    ],
)
def test_endpoint_url_uses_the_provider_dialect_path(
    base_url: str, dialect: RequestDialect, model: str, expected: str
) -> None:
    """A dialect-specific path must reach the matching provider API."""
    assert endpoint_url(base_url, dialect, model) == expected


def test_endpoint_url_refuses_an_invalid_gemini_model_identifier() -> None:
    """A malformed Gemini model must not become a path controlled by the sandbox."""
    with pytest.raises(RouteRefused, match="is not a valid model id"):
        _ = endpoint_url(
            "https://gw.test/google-ai", RequestDialect.GEMINI, "gemini/2.0"
        )


async def test_default_route_reads_anthropic_key_and_default_base() -> None:
    """The default host needs only an Anthropic key to route an allowed call."""
    call = _call(RequestDialect.MESSAGES, "claude-x", "anthropic/claude-x")

    assert await default_route(call, {"ANTHROPIC_API_KEY": "k"}) == Route(
        "https://api.anthropic.com", {"x-api-key": "k"}, "claude-x"
    )


async def test_default_route_prefers_the_configured_base_url() -> None:
    """A configured provider endpoint must override the public default."""
    call = _call(RequestDialect.MESSAGES, "claude-x", "anthropic/claude-x")

    route = await default_route(
        call,
        {
            "ANTHROPIC_API_KEY": "k",
            "ANTHROPIC_BASE_URL": "https://proxy.example/anthropic",
        },
    )

    assert route.base_url == "https://proxy.example/anthropic"


async def test_default_route_falls_back_across_google_key_aliases() -> None:
    """Harbor's Google key aliases must each make a default Google route usable."""
    call = _call(RequestDialect.GEMINI, "gemini-2.0", "google/gemini-2.0")

    route = await default_route(call, {"GOOGLE_GENERATIVE_AI_API_KEY": "google-key"})

    assert route == Route(
        "https://generativelanguage.googleapis.com",
        {"x-goog-api-key": "google-key"},
        "gemini-2.0",
    )


async def test_default_route_names_all_required_key_variables_when_none_is_available() -> (
    None
):
    """A missing provider credential should tell the operator every accepted alias."""
    call = _call(RequestDialect.GEMINI, "gemini-2.0", "google/gemini-2.0")

    with pytest.raises(
        RuntimeError,
        match="GOOGLE_API_KEY.*GEMINI_API_KEY.*GOOGLE_GENERATIVE_AI_API_KEY",
    ):
        _ = await default_route(call, {})


def test_default_protected_values_returns_only_present_provider_keys() -> None:
    """Only routed credentials belong in the sandbox and recording redaction set."""
    environment = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GOOGLE_API_KEY": "google-api-key",
        "GEMINI_API_KEY": "gemini-key",
        "UNRELATED": "not-a-provider-key",
    }

    assert default_protected_values(environment) == frozenset(
        {"anthropic-key", "google-api-key", "gemini-key"}
    )
