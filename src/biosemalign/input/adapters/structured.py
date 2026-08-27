"""Adapters for structured input: single fields, whole records, and batches."""

from __future__ import annotations

from typing import Any

from biosemalign.enums import SourceType
from biosemalign.exceptions import UnsupportedInput
from biosemalign.input.adapters.base import ADAPTER_VERSION, register_adapter
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.source import SourceRecord

__all__ = [
    "StructuredBatchAdapter",
    "StructuredFieldAdapter",
    "StructuredRecordAdapter",
    "register_default_adapters",
]


class StructuredFieldAdapter:
    """One database field: a bare value plus whatever context came with it."""

    adapter_name = "structured_field"
    adapter_version = ADAPTER_VERSION

    def to_source_records(
        self, request: AlignmentRequest, profile: TaskProfile
    ) -> list[SourceRecord]:
        if not isinstance(request.value, str):
            raise UnsupportedInput(
                f"{self.adapter_name} expects a string value, got {type(request.value).__name__}"
            )
        return [
            SourceRecord(
                source_type=SourceType.STRUCTURED_FIELD,
                raw_value=request.value,
                source_system=request.source_system,
                column_name=profile.input.value_field,
                record_id=_as_optional_str(request.context.get("record_id")),
                row=dict(request.context),
            )
        ]


class StructuredRecordAdapter:
    """A whole row. The profile says which key holds the value to be mapped."""

    adapter_name = "structured_record"
    adapter_version = ADAPTER_VERSION

    def to_source_records(
        self, request: AlignmentRequest, profile: TaskProfile
    ) -> list[SourceRecord]:
        if not isinstance(request.value, dict):
            raise UnsupportedInput(
                f"{self.adapter_name} expects a mapping value, got {type(request.value).__name__}"
            )

        field = profile.input.value_field
        if field is None:
            raise UnsupportedInput(
                f"profile {profile.profile_name!r} uses record input but does not set "
                "input.value_field, so the adapter cannot tell which key to map"
            )
        if field not in request.value:
            raise UnsupportedInput(
                f"record does not contain the configured value field {field!r}; "
                f"available keys: {sorted(request.value)}"
            )

        raw = request.value[field]
        if raw is None or str(raw).strip() == "":
            raise UnsupportedInput(f"record field {field!r} is empty")

        # The rest of the record travels along as neighbouring-field context,
        # merged under any explicitly supplied context (§5.4).
        row: dict[str, Any] = {k: v for k, v in request.value.items() if k != field}
        row.update(request.context)

        return [
            SourceRecord(
                source_type=SourceType.STRUCTURED_RECORD,
                raw_value=str(raw),
                source_system=request.source_system,
                column_name=field,
                record_id=_as_optional_str(request.value.get("record_id") or row.get("record_id")),
                row=row,
            )
        ]


class StructuredBatchAdapter:
    """A batch table, expanded one row at a time through the record adapter."""

    adapter_name = "structured_batch"
    adapter_version = ADAPTER_VERSION

    def __init__(self) -> None:
        self._record_adapter = StructuredRecordAdapter()

    def to_source_records(
        self, request: AlignmentRequest, profile: TaskProfile
    ) -> list[SourceRecord]:
        if not isinstance(request.value, dict) or "rows" not in request.value:
            raise UnsupportedInput(
                f"{self.adapter_name} expects a mapping with a 'rows' key holding a list"
            )
        rows = request.value["rows"]
        if not isinstance(rows, list):
            raise UnsupportedInput("'rows' must be a list of record mappings")

        records: list[SourceRecord] = []
        for batch_index, row in enumerate(rows, start=1):
            row_request = request.model_copy(update={"value": row})
            row_records = self._record_adapter.to_source_records(row_request, profile)
            for record in row_records:
                record.extensions["batch_index"] = batch_index
            records.extend(row_records)
        for record in records:
            record.source_type = SourceType.STRUCTURED_BATCH
        return records


def _as_optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def register_default_adapters() -> None:
    """Register the built-in structured adapters."""
    register_adapter(SourceType.STRUCTURED_FIELD, StructuredFieldAdapter())
    register_adapter(SourceType.STRUCTURED_RECORD, StructuredRecordAdapter())
    register_adapter(SourceType.STRUCTURED_BATCH, StructuredBatchAdapter())
