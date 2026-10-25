"""Host-secret collaborators; this module deliberately stores no process-global state."""

from __future__ import annotations

from .collaborators import DefaultSecretSource, SecretSource

MIN_SECRET_LENGTH = 16

__all__ = ["DefaultSecretSource", "MIN_SECRET_LENGTH", "SecretSource"]
