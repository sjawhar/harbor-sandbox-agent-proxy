"""Host-independent routing of sandbox model calls."""

from __future__ import annotations

import dataclasses
import enum
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Final


class RequestDialect(enum.StrEnum):
    """The provider API addressed by a sandbox proxy request."""

    MESSAGES = "messages"
    CHAT_COMPLETIONS = "chat-completions"
    RESPONSES = "responses"
    GEMINI = "gemini"

    @property
    def provider(self) -> str:
        """The default-route provider for this request dialect."""
        match self:
            case RequestDialect.MESSAGES:
                return "anthropic"
            case RequestDialect.CHAT_COMPLETIONS | RequestDialect.RESPONSES:
                return "openai"
            case RequestDialect.GEMINI:
                return "google"


@dataclasses.dataclass(frozen=True)
class ModelCall:
    """The model identity an agent presented to one provider endpoint."""

    api: RequestDialect
    model: str | None
    declared_models: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Route:
    """A host-selected destination for one permitted model call."""

    base_url: str
    headers: Mapping[str, str]
    model: str


class RouteRefused(Exception):
    """A host refused to send the sandbox model call upstream."""


type RouteCall = Callable[[ModelCall], Awaitable[Route]]


KEY_VARIABLES: Final[dict[str, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
}
"""Harbor's public-provider API-key environment variables by provider."""

BASE_URL_VARIABLES: Final[dict[str, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_BASE_URL",),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "google": ("GOOGLE_BASE_URL",),
}
"""Harbor's public-provider base-URL environment variables by provider."""

PUBLIC_BASE_URLS: Final[dict[str, str]] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com",
}

_MODEL_NAME_RE: Final = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def match_declared(call: ModelCall) -> str:
    """Return the one declared wire model that permits ``call``.

    The default host understands only an optional provider prefix. Hosts whose model
    declarations have richer semantics implement those in ``RelayEnvironment.route``.
    """
    if call.model is None:
        raise RouteRefused(_undeclared_message(call))
    wire_model = _model_name_for_endpoint(call.model, call.api.provider)
    matches = [
        declared
        for declared in call.declared_models
        if _declared_model_name(declared, call.api.provider) == wire_model
    ]
    if len(matches) == 1:
        return wire_model
    if len(matches) > 1:
        raise RouteRefused(
            f"model {call.model!r} matches multiple launch declarations for the "
            f"{call.api.provider} endpoint"
        )
    raise RouteRefused(_undeclared_message(call))


def auth_headers(api: RequestDialect, key: str) -> dict[str, str]:
    """Return the authentication header the request dialect accepts."""
    match api:
        case RequestDialect.MESSAGES:
            return {"x-api-key": key}
        case RequestDialect.CHAT_COMPLETIONS | RequestDialect.RESPONSES:
            return {"authorization": f"Bearer {key}"}
        case RequestDialect.GEMINI:
            return {"x-goog-api-key": key}


def endpoint_url(base_url: str, api: RequestDialect, model: str) -> str:
    """Return the upstream endpoint for ``api`` at ``base_url``."""
    match api:
        case RequestDialect.MESSAGES:
            path = "v1/messages"
        case RequestDialect.CHAT_COMPLETIONS:
            path = "chat/completions"
        case RequestDialect.RESPONSES:
            path = "responses"
        case RequestDialect.GEMINI:
            if not _MODEL_NAME_RE.fullmatch(model):
                raise RouteRefused(f"model name {model!r} is not a valid model id")
            path = f"v1beta/models/{model}:streamGenerateContent?alt=sse"
    return f"{base_url.rstrip('/')}/{path}"


async def default_route(
    call: ModelCall, environment: Mapping[str, str] | None = None
) -> Route:
    """Route an allowed call through the public endpoint configured in the host environment."""
    values = os.environ if environment is None else environment
    provider = call.api.provider
    key = _first_set(values, KEY_VARIABLES[provider])
    if key is None:
        names = ", ".join(KEY_VARIABLES[provider])
        raise RuntimeError(f"none of {names} is set in the host environment")
    base_url = (
        _first_set(values, BASE_URL_VARIABLES[provider]) or PUBLIC_BASE_URLS[provider]
    )
    return Route(base_url, auth_headers(call.api, key), match_declared(call))


def default_protected_values(
    environment: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Return the default host credentials that must not enter a sandbox or record."""
    values = os.environ if environment is None else environment
    return frozenset(
        value
        for names in KEY_VARIABLES.values()
        for name in names
        if (value := values.get(name))
    )


def _model_name_for_endpoint(model: str, provider: str) -> str:
    """Return an endpoint's bare model name or reject another provider prefix."""
    prefix, separator, bare = model.partition("/")
    if not separator:
        return model
    if prefix != provider:
        raise RouteRefused(
            f"model {model!r} is not one the launch declared for the {provider} endpoint"
        )
    return bare


def _declared_model_name(model: str, provider: str) -> str | None:
    """Return a declaration's model name when it belongs to ``provider``."""
    prefix, separator, bare = model.partition("/")
    if not separator:
        return model
    return bare if prefix == provider else None


def _undeclared_message(call: ModelCall) -> str:
    return (
        f"model {call.model!r} is not one the launch declared for this phase "
        f"({sorted(call.declared_models)})"
    )


def _first_set(environment: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if value := environment.get(name):
            return value
    return None
