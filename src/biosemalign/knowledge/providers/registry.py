"""Knowledge-provider factory registry (§5.7, §6).

External and domain-specific terminology providers (such as ``rxnav``) register
themselves here rather than being imported by the core knowledge router.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from biosemalign.exceptions import ResourceNotPermitted
from biosemalign.knowledge.providers.base import KnowledgeProvider
from biosemalign.plugins import ensure_plugins_loaded
from biosemalign.resources.registry import ResourceSpec
from biosemalign.settings import Settings
from biosemalign.storage.cache import ResponseCache

__all__ = [
    "ProviderContext",
    "ProviderFactory",
    "available_providers",
    "build_provider",
    "register_provider",
    "unregister_provider",
]


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Inputs supplied to a registered provider factory."""

    spec: ResourceSpec
    settings: Settings
    cache: ResponseCache


ProviderFactory = Callable[[ProviderContext], KnowledgeProvider]

_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under ``name``."""
    _PROVIDERS[name.casefold()] = factory


def unregister_provider(name: str) -> None:
    """Remove a registered provider factory, if present."""
    _PROVIDERS.pop(name.casefold(), None)


def available_providers() -> list[str]:
    """Sorted names of registered providers (plus the built-in local lexicon)."""
    ensure_plugins_loaded()
    return sorted({*_PROVIDERS, "lexicon"})


def build_provider(name: str, context: ProviderContext) -> KnowledgeProvider:
    """Instantiate a registered provider by name."""
    ensure_plugins_loaded()
    factory = _PROVIDERS.get(name.casefold())
    if factory is None:
        raise ResourceNotPermitted(
            f"resource {context.spec.name!r} maps to provider {name!r}, which is "
            f"not implemented in this release. Implemented: {available_providers()}"
        )
    return factory(context)
