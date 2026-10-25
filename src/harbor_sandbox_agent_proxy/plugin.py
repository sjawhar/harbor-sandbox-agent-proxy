"""Harbor observer plugin for the package's default per-trial JSON outcome."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, override

from harbor.models.job.plugin import BaseJobPlugin

logger = logging.getLogger(__name__)


class SandboxAgentProxyPlugin(BaseJobPlugin):
    """Report completed trial call counts without owning relay state."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @override
    async def on_job_start(self, job: Any) -> None:
        _ = job.on_trial_ended(self._on_trial_ended)

    @override
    async def on_job_end(self, job_result: Any) -> None:
        del job_result

    async def _on_trial_ended(self, event: Any) -> None:
        outcome_path = Path(event.config.trials_dir) / event.trial_name / "sandbox-agent-proxy" / "outcome.json"
        try:
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            counts = outcome.get("counts", {})
        except (OSError, ValueError, TypeError):
            return
        logger.info("sandbox-agent-proxy trial %s calls: %s", event.trial_name, counts)
