"""Resolve import-path collaborators without retaining process-wide trial state."""

from __future__ import annotations

from importlib import import_module
from typing import TypeVar

T = TypeVar("T")


def resolve_collaborator(value: object, *, label: str) -> object:
    """Resolve a ``module:attribute`` value supplied in JSON wrapper kwargs."""
    if not isinstance(value, str):
        return value
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"{label} must be an object or a module:attribute import path")
    resolved = getattr(import_module(module_name), attribute)
    return resolved() if isinstance(resolved, type) else resolved


def collaborator(value: object, *, label: str, expected: type[T]) -> T:
    """Resolve and type-check one collaborator supplied to a wrapper."""
    resolved = resolve_collaborator(value, label=label)
    if not isinstance(resolved, expected):
        raise TypeError(f"{label} must resolve to {expected.__name__}")
    return resolved
