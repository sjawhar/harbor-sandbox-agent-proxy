from __future__ import annotations

from harbor_sandbox_agent_proxy.collaborators import DefaultSecretSource, DefaultUpstream, RequestDialect


async def test_default_upstream_reads_the_host_key_for_each_request() -> None:
    host_environment = {"ANTHROPIC_API_KEY": "first-host-key"}
    upstream = DefaultUpstream(host_environment)

    first = await upstream.resolve(
        dialect=RequestDialect.MESSAGES,
        model="claude-sonnet-4-5",
    )
    host_environment["ANTHROPIC_API_KEY"] = "renewed-host-key"
    second = await upstream.resolve(
        dialect=RequestDialect.MESSAGES,
        model="claude-sonnet-4-5",
    )

    assert first.url == "https://api.anthropic.com/v1/messages"
    assert first.headers == {"x-api-key": "first-host-key"}
    assert second.headers == {"x-api-key": "renewed-host-key"}


def test_default_secret_source_reports_public_provider_keys() -> None:
    source = DefaultSecretSource(
        {
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "OPENAI_API_KEY": "openai-secret",
            "GEMINI_API_KEY": "gemini-secret",
            "UNRELATED": "not-a-secret",
        }
    )

    assert source.protected_values() == frozenset(
        {"anthropic-secret", "openai-secret", "gemini-secret"}
    )
