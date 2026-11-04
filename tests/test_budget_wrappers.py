"""Trusted task requirements and per-trial lifecycle integration."""
from __future__ import annotations
import json
import pytest
from harbor_sandbox_agent_proxy.config import RelayConfig
from harbor_sandbox_agent_proxy.task_metadata import read_bridged_servers
from .test_budget import settings


def test_45_budget_profile_covers_routed_models():
    with pytest.raises(ValueError):
        RelayConfig(declared_models=("m", "other"), token_budget=settings())


def test_62_task_budget_requirement_is_read_without_breaking_tools(tmp_path):
    from harbor_sandbox_agent_proxy.task_metadata import check_token_budget
    (tmp_path / "task.toml").write_text('[metadata.sandbox_agent_proxy]\ntoken_budget_required=true\ntoken_budget_limit=1000000\n')
    assert read_bridged_servers(tmp_path) == ()
    with pytest.raises(ValueError):
        check_token_budget(tmp_path, None)
    check_token_budget(tmp_path, settings())
    with pytest.raises(ValueError):
        check_token_budget(tmp_path, settings(enforce=False))
    with pytest.raises(ValueError):
        check_token_budget(tmp_path, settings(limit=999))


@pytest.mark.asyncio
async def test_59_60_trial_relay_closes_and_publishes_budget(tmp_path):
    from .test_trial_relay import _relay, _settings
    from harbor_sandbox_agent_proxy.outcome import TrialIdentity
    from harbor_sandbox_agent_proxy.trial_relay import TrialRelay
    from .test_budget import api, body, measured
    cfg = _settings()
    cfg.declared_models = ("m",)
    cfg.token_budget = settings()
    r = _relay()
    r = TrialRelay(trial=TrialIdentity("t", tmp_path), config=cfg, route=r.route, sink=r.sink, connection=r.connection)
    a = await r.budget.admit(r.budget.prepare("anthropic", "m", body()), "a")
    r.budget.settle(a, measured(10, 1))
    outcome = await r.close()
    assert outcome.budget["spent"] == 11
    assert api().read_budget(tmp_path)["complete"] is True


def test_61_disabled_budget_keeps_serialized_records_identical():
    from .test_trial_relay import _call
    call = _call()
    assert "budget" not in call.model_dump(mode="json")
    assert "token_budget" not in RelayConfig(declared_models=("m",)).model_dump(mode="json")
