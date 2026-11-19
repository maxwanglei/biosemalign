"""Named operator factories, so profiles can assemble workflows in YAML.

A profile names operators; this resolves them. Unknown names fail when the
pipeline is built rather than when the first record reaches the missing stage,
and a factory that returns something not satisfying its protocol fails the
same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import ValidationError

from biosemalign.exceptions import ConfigurationError
from biosemalign.ops.base import (
    FuseOp,
    OpContext,
    ProposeOp,
    RetrieveOp,
    TransformOp,
    ValidateOp,
)

__all__ = ["OpFactory", "OpKind", "available_ops", "build_op", "register_op"]

OpKind = Literal["transform", "retrieve", "fuse", "propose", "validate"]
OpFactory = Callable[[OpContext, Mapping[str, Any]], object]

_PROTOCOLS: dict[str, type] = {
    "transform": TransformOp,
    "retrieve": RetrieveOp,
    "fuse": FuseOp,
    "propose": ProposeOp,
    "validate": ValidateOp,
}

_REGISTRY: dict[tuple[str, str], OpFactory] = {}


def register_op(kind: OpKind, name: str, factory: OpFactory) -> None:
    """Register an operator factory, replacing any existing one of that name."""
    if kind not in _PROTOCOLS:
        raise ValueError(f"unknown operator kind {kind!r}; expected one of {sorted(_PROTOCOLS)}")
    _REGISTRY[(kind, name)] = factory


def available_ops(kind: OpKind) -> list[str]:
    """Registered operator names of one kind."""
    return sorted(name for registered, name in _REGISTRY if registered == kind)


def build_op(
    kind: OpKind, name: str, context: OpContext, options: Mapping[str, Any] | None = None
) -> Any:
    """Construct a registered operator and check it satisfies its protocol."""
    try:
        factory = _REGISTRY[(kind, name)]
    except KeyError:
        raise ConfigurationError(
            f"no {kind} operator named {name!r} is registered; available: {available_ops(kind)}"
        ) from None
    try:
        op = factory(context, dict(options or {}))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid options for {kind} operator {name!r}: {exc}") from exc
    if not isinstance(op, _PROTOCOLS[kind]):
        raise ConfigurationError(
            f"{kind} operator {name!r} does not implement the {_PROTOCOLS[kind].__name__} protocol"
        )
    return op
