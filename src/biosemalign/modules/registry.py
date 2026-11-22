"""Named operator factories, so profiles can assemble workflows in YAML (§5.3).

A profile names operators; this resolves them. Unknown names fail when the
pipeline is built rather than when the first record reaches the missing module,
and a factory that returns something not satisfying its protocol fails the
same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import ValidationError

from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import (
    Check,
    Cleaner,
    Expander,
    Linker,
    Matcher,
    OpContext,
    Resolver,
    Selector,
)
from biosemalign.plugins import ensure_plugins_loaded

__all__ = [
    "REGISTRY",
    "OpFactory",
    "OpKind",
    "OperatorRegistry",
    "available",
    "build",
    "default_matcher_for_domain",
    "register",
    "register_domain_default_matcher",
    "unregister",
    "unregister_domain_default_matcher",
]

OpKind = Literal["clean", "match", "expand", "link", "resolve", "select", "check"]
OpFactory = Callable[[OpContext, Mapping[str, Any]], object]

_PROTOCOLS: dict[str, type] = {
    "clean": Cleaner,
    "match": Matcher,
    "expand": Expander,
    "link": Linker,
    "resolve": Resolver,
    "select": Selector,
    "check": Check,
}


def _check_kind(kind: str) -> str:
    if kind not in _PROTOCOLS:
        raise ValueError(f"unknown operator kind {kind!r}; expected one of {sorted(_PROTOCOLS)}")
    return kind


class OperatorRegistry:
    """Registry of operator factories keyed by ``(kind, name)``."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], OpFactory] = {}

    def register(self, kind: OpKind, name: str, factory: OpFactory) -> None:
        """Register an operator factory, replacing any existing one of that name."""
        self._entries[(_check_kind(kind), name)] = factory

    def unregister(self, kind: OpKind, name: str) -> None:
        """Remove an operator factory from the registry, if present."""
        self._entries.pop((_check_kind(kind), name), None)

    def available(self, kind: OpKind) -> list[str]:
        """Sorted names of registered operators for ``kind``."""
        checked = _check_kind(kind)
        return sorted(name for registered_kind, name in self._entries if registered_kind == checked)

    def build(
        self,
        kind: OpKind,
        name: str,
        context: OpContext,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Construct a registered operator and verify it implements its protocol."""
        checked = _check_kind(kind)
        try:
            factory = self._entries[(checked, name)]
        except KeyError:
            raise ConfigurationError(
                f"no {checked} operator named {name!r} is registered; "
                f"available: {self.available(kind)}"
            ) from None
        try:
            op = factory(context, dict(options or {}))
        except ValidationError as exc:
            raise ConfigurationError(
                f"invalid options for {checked} operator {name!r}: {exc}"
            ) from exc
        protocol = _PROTOCOLS[checked]
        if not isinstance(op, protocol):
            raise ConfigurationError(
                f"{checked} operator {name!r} does not implement the {protocol.__name__} protocol"
            )
        return op


REGISTRY = OperatorRegistry()


def register(kind: OpKind, name: str, factory: OpFactory) -> None:
    """Register an operator factory in the global registry."""
    REGISTRY.register(kind, name, factory)


def unregister(kind: OpKind, name: str) -> None:
    """Remove an operator factory from the global registry."""
    REGISTRY.unregister(kind, name)


def available(kind: OpKind) -> list[str]:
    """Registered operator names for ``kind``."""
    ensure_plugins_loaded()
    return REGISTRY.available(kind)


def build(
    kind: OpKind,
    name: str,
    context: OpContext,
    options: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a registered operator from the global registry."""
    ensure_plugins_loaded()
    return REGISTRY.build(kind, name, context, options)


_DOMAIN_DEFAULT_MATCHERS: dict[str, str] = {}


def register_domain_default_matcher(domain: str, op_name: str) -> None:
    """Register the fallback default matcher operator for ``domain``."""
    _DOMAIN_DEFAULT_MATCHERS[domain.casefold()] = op_name


def unregister_domain_default_matcher(domain: str) -> None:
    """Remove a registered default matcher for ``domain``."""
    _DOMAIN_DEFAULT_MATCHERS.pop(domain.casefold(), None)


def default_matcher_for_domain(domain: str | None) -> str:
    """Default fallback matcher operator name for ``domain``, or core ``'exact'``."""
    if not domain:
        return "exact"
    ensure_plugins_loaded()
    return _DOMAIN_DEFAULT_MATCHERS.get(domain.casefold(), "exact")
