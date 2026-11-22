"""Dynamic domain-pack and plugin loader (§5.7, §6).

Nothing in the core imports a domain pack statically. Built-in packs and
third-party distributions advertising the ``biosemalign.plugins`` entry-point
group are discovered and loaded dynamically through this module.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "BUILTIN_DOMAINS",
    "PLUGIN_ENTRY_POINT_GROUP",
    "ensure_plugins_loaded",
    "load_domain",
    "load_entry_point_plugins",
    "loaded_domains",
    "register_domain_cleanup",
    "unload_domain",
    "without_domain",
]

PLUGIN_ENTRY_POINT_GROUP = "biosemalign.plugins"

#: Built-in domain packs shipped inside the distribution, loaded dynamically
#: rather than imported statically by the core.
BUILTIN_DOMAINS: dict[str, str] = {
    "drug": "biosemalign.domains.drug",
    "hierarchy": "biosemalign.domains.hierarchy",
}

_LOADED: set[str] = set()
_BLOCKED: set[str] = set()
_CLEANUPS: dict[str, list[Callable[[], None]]] = {}
_ENTRY_POINTS_LOADED: set[str] = set()
_INITIALIZED = False
_ENTRY_POINTS_SCANNED = False


def _disabled_from_env() -> set[str]:
    raw = os.getenv("BIOSEMALIGN_DISABLE_DOMAINS", "")
    return {part.strip().casefold() for part in raw.split(",") if part.strip()}


def register_domain_cleanup(domain: str, cleanup: Callable[[], None]) -> None:
    """Record a callback that reverses a domain pack's registrations on unload."""
    _CLEANUPS.setdefault(domain.casefold(), []).append(cleanup)


def loaded_domains() -> frozenset[str]:
    """Names of currently loaded domain packs."""
    return frozenset(_LOADED)


def load_domain(name: str) -> bool:
    """Load and register one domain pack by name, if available and not blocked."""
    key = name.casefold()
    if key in _BLOCKED or key in _disabled_from_env():
        return False
    if key in _LOADED:
        return True
    module_name = BUILTIN_DOMAINS.get(key)
    if module_name is None:
        return _load_entry_point_by_name(key)
    module = importlib.import_module(module_name)
    _invoke_register(module, key)
    _LOADED.add(key)
    return True


def _invoke_register(target: Any, domain_key: str) -> None:
    register_fn = target if callable(target) else getattr(target, "register", None)
    if register_fn is None:
        return
    cleanups = register_fn()
    if cleanups is None:
        return
    if callable(cleanups):
        register_domain_cleanup(domain_key, cleanups)
    else:
        for fn in cleanups:
            if callable(fn):
                register_domain_cleanup(domain_key, fn)


def _entry_points() -> list[importlib.metadata.EntryPoint]:
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=PLUGIN_ENTRY_POINT_GROUP))
    return []


def _load_entry_point_by_name(name: str) -> bool:
    key = name.casefold()
    for ep in _entry_points():
        if ep.name.casefold() == key:
            if ep.name in _ENTRY_POINTS_LOADED:
                return True
            target = ep.load()
            _invoke_register(target, key)
            _ENTRY_POINTS_LOADED.add(ep.name)
            _LOADED.add(key)
            return True
    return False


def load_entry_point_plugins() -> list[str]:
    """Load every third-party plugin declared under ``biosemalign.plugins``."""
    disabled = _disabled_from_env() | _BLOCKED
    loaded_now: list[str] = []
    for ep in _entry_points():
        key = ep.name.casefold()
        if key in disabled or ep.name in _ENTRY_POINTS_LOADED:
            continue
        target = ep.load()
        _invoke_register(target, key)
        _ENTRY_POINTS_LOADED.add(ep.name)
        _LOADED.add(key)
        loaded_now.append(ep.name)
    return loaded_now


def ensure_plugins_loaded(*, reload_entry_points: bool = False) -> None:
    """Idempotently load built-in domain packs (unless disabled) and entry-point plugins."""
    global _INITIALIZED, _ENTRY_POINTS_SCANNED
    if not _INITIALIZED:
        _INITIALIZED = True
        for domain_name in BUILTIN_DOMAINS:
            load_domain(domain_name)
    if not _ENTRY_POINTS_SCANNED or reload_entry_points:
        _ENTRY_POINTS_SCANNED = True
        load_entry_point_plugins()


def unload_domain(name: str) -> None:
    """Reverse all registrations for ``name`` and drop its modules from ``sys.modules``."""
    key = name.casefold()
    for cleanup in reversed(_CLEANUPS.pop(key, [])):
        cleanup()
    _LOADED.discard(key)
    for ep_name in list(_ENTRY_POINTS_LOADED):
        if ep_name.casefold() == key:
            _ENTRY_POINTS_LOADED.discard(ep_name)
    prefix = BUILTIN_DOMAINS.get(key)
    if prefix is not None:
        for mod_name in list(sys.modules):
            if mod_name == prefix or mod_name.startswith(prefix + "."):
                sys.modules.pop(mod_name, None)


@contextmanager
def without_domain(name: str) -> Iterator[None]:
    """Temporarily unload and block a domain pack, restoring it on exit."""
    key = name.casefold()
    was_loaded = key in _LOADED
    unload_domain(key)
    _BLOCKED.add(key)
    try:
        yield
    finally:
        _BLOCKED.discard(key)
        if was_loaded:
            load_domain(key)
