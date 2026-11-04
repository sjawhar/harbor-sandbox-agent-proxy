"""Package-neutral configuration for a sandbox-agent relay."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .budget import ShadowBudgetProposal, TokenBudgetConfig


class RelayConfig(BaseModel):
    """Configuration shared by the agent and environment wrappers."""

    model_config = ConfigDict(extra="forbid")

    declared_models: tuple[str, ...] = Field(min_length=1)
    port: int = Field(default=13131, ge=1, le=65535)
    polling_interval: float | None = Field(default=None, gt=0)
    tools: Literal["any", "bridged-only"] = "any"
    verifier_tools: Literal["any", "bridged-only"] | None = None
    permitted_tool_names: tuple[str, ...] = ()
    permitted_tool_prefixes: tuple[str, ...] = ()
    allowed_betas: tuple[str, ...] | None = None
    verifier_allowed_betas: tuple[str, ...] | None = None
    max_request_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    max_calls: int = Field(default=10_000, gt=0)
    max_recorded_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    # "full" keeps every call and Inspect event for outcome.json and the verifier's
    # conversation. "usage" keeps neither: each call appends one line of token usage to
    # sandbox-agent-proxy/usage.jsonl, and the verifier's conversation is refused.
    record: Literal["full", "usage"] = "full"
    token_budget: TokenBudgetConfig | ShadowBudgetProposal | None = Field(default=None, exclude_if=lambda value: value is None)
    conversation_path: PurePosixPath = PurePosixPath(
        "/tests/sandbox-agent-proxy/conversation.json"
    )

    @field_validator("token_budget", mode="before")
    @classmethod
    def budget_policy(cls, value):
        if value is None or isinstance(value, (TokenBudgetConfig, ShadowBudgetProposal)):
            return value
        try:
            return TokenBudgetConfig.model_validate(value)
        except ValidationError:
            if isinstance(value, dict) and value.get("enforce") is False:
                return ShadowBudgetProposal.model_validate(value)
            raise

    @model_validator(mode="after")
    def budget_models(self):
        if self.token_budget is not None and self.token_budget.enforce:
            routed = {model.split("/", 1)[-1] for model in self.declared_models}
            if routed != set(self.token_budget.output_ceiling):
                raise ValueError("budget profiles must cover every routed declared model")
        return self
