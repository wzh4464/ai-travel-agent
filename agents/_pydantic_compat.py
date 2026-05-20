"""Compatibility shim for Pydantic across langchain versions.

LangChain ``<=0.2`` shipped the ``langchain.pydantic_v1`` re-export so that
tool schemas could be written against Pydantic v1 while the wider
ecosystem migrated to v2. LangChain ``>=1.0`` removed that module and
expects ``pydantic`` (v2) directly.

We prefer the direct import so modern environments work out of the box,
and fall back to the legacy shim when it is present. Any other fallback
order would silently break one of the two ecosystems.
"""

from __future__ import annotations

try:  # Pydantic v2 (langchain >= 1.x or standalone pydantic)
    from pydantic import BaseModel, Field  # type: ignore[assignment]
except ImportError:  # pragma: no cover - exercised only on very old stacks
    from langchain.pydantic_v1 import BaseModel, Field  # type: ignore[no-redef]

__all__ = ['BaseModel', 'Field']
