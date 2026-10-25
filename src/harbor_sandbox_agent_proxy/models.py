"""Public model-routing types.

Model parsing and deployment routing are intentionally supplied by an injected
``Upstream`` rather than encoded in this package.
"""

from __future__ import annotations

from .collaborators import RequestDialect, Upstream, UpstreamTarget

__all__ = ["RequestDialect", "Upstream", "UpstreamTarget"]
