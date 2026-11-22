"""Local lexicon bundles: contract, builder, and domain-tool registry (v0.4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from biosemalign.exceptions import LexiconError
from biosemalign.plugins import ensure_plugins_loaded

__all__ = ["get_lexicon_tool", "register_lexicon_tool", "unregister_lexicon_tool"]

_TOOLS: dict[str, Any] = {}


def register_lexicon_tool(name: str, tool: Any) -> None:
    """Register a lexicon CLI helper (such as ``link`` or ``import_omop``)."""
    _TOOLS[name.casefold()] = tool


def unregister_lexicon_tool(name: str) -> None:
    """Remove a registered lexicon helper."""
    _TOOLS.pop(name.casefold(), None)


def get_lexicon_tool(name: str) -> Callable[..., Any]:
    """Return a registered lexicon helper by name, loading plugins first."""
    ensure_plugins_loaded()
    tool = _TOOLS.get(name.casefold())
    if tool is None:
        raise LexiconError(
            f"lexicon tool {name!r} is unavailable; enable the domain pack that provides it"
        )
    return tool
