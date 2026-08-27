"""Where the value being mapped came from."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from biosemalign.enums import SourceType
from biosemalign.schemas.base import ExtensibleModel

__all__ = ["SourceRecord"]


class SourceRecord(ExtensibleModel):
    """The originating record, preserved verbatim.

    ``raw_value`` is never normalized, trimmed, or case-folded. Downstream
    stages work on derived copies; this field is the audit anchor that lets a
    reviewer see exactly what the source system reported (§5.4, Milestone 2
    exit criterion 2).
    """

    source_type: SourceType
    raw_value: str = Field(description="The original value, byte-for-byte as received.")
    source_system: str | None = Field(
        default=None, description="Originating system, e.g. 'FAERS', 'MarketScan'."
    )
    table_name: str | None = None
    column_name: str | None = None
    record_id: str | None = Field(
        default=None, description="Identifier of the source row, e.g. a FAERS primaryid."
    )
    row: dict[str, Any] = Field(
        default_factory=dict,
        description="Neighbouring fields from the same record, unmodified.",
    )
