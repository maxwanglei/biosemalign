"""Base model classes and the schema-version policy.

Three layers, so that strictness and payload noise are both controlled:

``BioSemAlignModel``
    Strict base. Unknown fields are rejected, not silently dropped.
``ExtensibleModel``
    Adds an ``extensions`` mapping. Project-specific data goes here, which is
    what lets AI-BenchFmKB or MarketScan attach their own fields *without*
    editing the core contract (Milestone 0, exit criterion 3).
``VersionedModel``
    Adds ``schema_version``. Applied to the top-level contracts only, since
    stamping every nested object would triple the size of serialized output for
    no benefit.
"""

from __future__ import annotations

from typing import Any, Never

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SCHEMA_VERSION",
    "BioSemAlignModel",
    "ExtensibleModel",
    "FrozenDict",
    "VersionedModel",
    "deep_freeze",
]

SCHEMA_VERSION = "0.2.0"
"""Version of the core data contracts.

Bump the minor component when a field is added, the major component when a
field is removed or changes meaning. ``docs/schemas/`` is regenerated from these
models and committed, so any change to this file shows up as a reviewable diff.
"""


class FrozenDict(dict[str, Any]):
    """A normal JSON-serializable mapping that rejects mutation."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise TypeError("immutable mapping cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def deep_freeze(value: Any) -> Any:
    """Recursively freeze JSON-like containers while retaining serialization."""
    if isinstance(value, dict):
        return FrozenDict({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(deep_freeze(item) for item in value)
    return value


class BioSemAlignModel(BaseModel):
    """Strict base for every BioSemAlign data contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=False,
        ser_json_timedelta="iso8601",
    )


class ExtensibleModel(BioSemAlignModel):
    """A contract that project-specific profiles may attach extra data to."""

    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Project-specific fields. The core schema never interprets these; "
            "they travel with the record and are preserved in output."
        ),
    )


class VersionedModel(ExtensibleModel):
    """A top-level contract that records the schema version it was written under."""

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Version of the BioSemAlign core schema this record conforms to.",
    )
