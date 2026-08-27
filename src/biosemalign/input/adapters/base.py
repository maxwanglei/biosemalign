"""Source adapters: heterogeneous input in, :class:`SourceRecord` out.

Adapters translate structure. They do not contain semantic-alignment logic
(§5.3) — a FAERS adapter knows that a reported product name lives in ``prod_ai``
and that ``role_cod`` is worth carrying along, and nothing about RxNorm.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from biosemalign.enums import SourceType
from biosemalign.exceptions import UnsupportedInput
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.source import SourceRecord

__all__ = ["ADAPTER_VERSION", "SourceAdapter", "get_adapter", "register_adapter"]

ADAPTER_VERSION = "0.1.0"


@runtime_checkable
class SourceAdapter(Protocol):
    """Turns one request into the source records it describes."""

    adapter_name: str
    adapter_version: str

    def to_source_records(
        self, request: AlignmentRequest, profile: TaskProfile
    ) -> list[SourceRecord]: ...


_REGISTRY: dict[SourceType, SourceAdapter] = {}


def register_adapter(source_type: SourceType, adapter: SourceAdapter) -> None:
    """Register an adapter for a source type, replacing any existing one."""
    _REGISTRY[source_type] = adapter


def get_adapter(source_type: SourceType) -> SourceAdapter:
    """Look up the adapter for a source type."""
    try:
        return _REGISTRY[source_type]
    except KeyError:
        raise UnsupportedInput(
            f"no adapter registered for source type {source_type.value!r}; "
            f"available: {sorted(t.value for t in _REGISTRY)}"
        ) from None
