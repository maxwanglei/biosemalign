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
    CandidateSet,
    Coder,
    FuseOp,
    Linker,
    OpContext,
    ProposeOp,
    Resolver,
    RetrieveOp,
    TransformOp,
    ValidateOp,
)

__all__ = ["OpFactory", "OpKind", "available_ops", "build_op", "canonical_kind", "register_op"]

OpKind = Literal["transform", "retrieve", "fuse", "propose", "validate", "code", "link", "resolve"]
OpFactory = Callable[[OpContext, Mapping[str, Any]], object]

#: Module names of the three middle stages (v0.3) and the stage each names.
_MODULE_KINDS = {"code": "retrieve", "link": "fuse", "resolve": "propose"}

_PROTOCOLS: dict[str, type] = {
    "transform": TransformOp,
    "retrieve": RetrieveOp,
    "fuse": FuseOp,
    "propose": ProposeOp,
    "validate": ValidateOp,
}

_REGISTRY: dict[tuple[str, str], OpFactory] = {}


def canonical_kind(kind: str) -> str:
    """The stage a kind names: ``code`` is ``retrieve``, ``link`` is ``fuse``, ..."""
    canonical = _MODULE_KINDS.get(kind, kind)
    if canonical not in _PROTOCOLS:
        known = sorted({*_PROTOCOLS, *_MODULE_KINDS})
        raise ValueError(f"unknown operator kind {kind!r}; expected one of {known}")
    return canonical


def register_op(kind: OpKind, name: str, factory: OpFactory) -> None:
    """Register an operator factory, replacing any existing one of that name.

    ``code``, ``link``, and ``resolve`` register into the ``retrieve``,
    ``fuse``, and ``propose`` stages they name.
    """
    _REGISTRY[(canonical_kind(kind), name)] = factory


def available_ops(kind: OpKind) -> list[str]:
    """Registered operator names of one kind."""
    canonical = canonical_kind(kind)
    return sorted(name for registered, name in _REGISTRY if registered == canonical)


def build_op(
    kind: OpKind, name: str, context: OpContext, options: Mapping[str, Any] | None = None
) -> Any:
    """Construct a registered operator and check it satisfies its protocol.

    An operator implementing only the module spelling (``code``, ``link``,
    ``resolve``) is adapted to its stage's protocol.
    """
    kind = canonical_kind(kind)  # type: ignore[assignment]
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
        op = _adapt(kind, op)
    if not isinstance(op, _PROTOCOLS[kind]):
        raise ConfigurationError(
            f"{kind} operator {name!r} does not implement the {_PROTOCOLS[kind].__name__} protocol"
        )
    return op


# -- adapters for module-spelled operators --------------------------------------


def _adapt(kind: str, op: Any) -> Any:
    if kind == "retrieve" and isinstance(op, Coder):
        return _CoderStage(op)
    if kind == "fuse" and isinstance(op, Linker):
        return _LinkerStage(op)
    if kind == "propose" and isinstance(op, Resolver):
        return _ResolverStage(op)
    return op


class _Adapted:
    """Presents a module-spelled operator under its stage's method name."""

    def __init__(self, op: Any) -> None:
        self.inner = op
        self.op_id = op.op_id
        self.version = op.version

    @property
    def options(self) -> dict[str, Any]:
        return dict(getattr(self.inner, "options", {}))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _CoderStage(_Adapted):
    @property
    def requires_network(self) -> bool:
        return bool(self.inner.requires_network)

    def retrieve(self, sau: Any) -> CandidateSet:
        result: CandidateSet = self.inner.code(sau)
        return result


class _LinkerStage(_Adapted):
    def fuse(self, sau: Any, candidate_sets: Any) -> Any:
        return self.inner.link(sau, candidate_sets)


class _ResolverStage(_Adapted):
    @property
    def requires_network(self) -> bool:
        return bool(self.inner.requires_network)

    def propose(self, sau: Any, package: Any) -> Any:
        return self.inner.resolve(sau, package)

    def provenance(self) -> Any:
        return self.inner.provenance()
