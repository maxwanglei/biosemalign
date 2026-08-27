"""Flat, analysis-ready output (§17.1).

The crosswalk table is what actually gets joined back to report-level FAERS
data (§18.3), so it is deliberately flat and deliberately keeps the original
reported string as the join key. Nested detail stays in the JSONL record.

``pyarrow`` is an optional dependency: the base install must stay light enough
for restricted environments, so Parquet lives behind ``biosemalign[data]`` and
CSV is always available as a fallback.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from biosemalign.schemas.decision import AlignmentDecision
from biosemalign.storage.paths import ensure_directory

__all__ = ["CROSSWALK_COLUMNS", "decision_to_row", "write_csv", "write_parquet"]

CROSSWALK_COLUMNS: tuple[str, ...] = (
    "sau_id",
    "reported_value",
    "record_id",
    "decision",
    "relationship_to_source",
    "source_level_id",
    "source_level_label",
    "source_level_class",
    "analysis_level_id",
    "analysis_level_label",
    "selected_ids",
    "selected_labels",
    "semantic_loss_kinds",
    "validation_status",
    "routing_status",
    "review_reasons",
    "vocabulary_release",
    "profile_version",
    "package_version",
    "run_id",
)


def decision_to_row(decision: AlignmentDecision) -> dict[str, Any]:
    """Flatten one decision into a crosswalk row."""
    source = decision.source_level_concept
    analysis = decision.analysis_level_concept
    release = next(
        (r.terminology_release for r in decision.provenance.resources if r.terminology_release),
        None,
    )
    return {
        "sau_id": decision.sau_id,
        "reported_value": decision.source_value,
        "record_id": decision.record_id,
        "decision": decision.decision.value,
        "relationship_to_source": decision.relationship_to_source.value,
        "source_level_id": source.concept_id if source else None,
        "source_level_label": source.label if source else None,
        "source_level_class": source.concept_class if source else None,
        "analysis_level_id": analysis.concept_id if analysis else None,
        "analysis_level_label": analysis.label if analysis else None,
        # Pipe-separated so a combination product stays on one row and the
        # crosswalk keeps a stable one-row-per-reported-string shape.
        "selected_ids": "|".join(c.concept_id for c in decision.selected_concepts),
        "selected_labels": "|".join(c.label for c in decision.selected_concepts),
        "semantic_loss_kinds": "|".join(
            sorted({loss.kind.value for loss in decision.semantic_loss})
        ),
        "validation_status": decision.validation.status.value,
        "routing_status": decision.routing_status.value,
        "review_reasons": "; ".join(decision.routing_reasons),
        "vocabulary_release": release,
        "profile_version": decision.provenance.profile_version,
        "package_version": decision.provenance.package_version,
        "run_id": decision.provenance.run_id,
    }


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
) -> int:
    """Write crosswalk rows as CSV. Always available, no optional dependency."""
    target = Path(path)
    ensure_directory(target.parent, dir_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    os.chmod(temporary, file_mode)
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CROSSWALK_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_safe(row.get(k)) for k in CROSSWALK_COLUMNS})
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, file_mode)
        _fsync_directory(target.parent)
        return count
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_parquet(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = 10_000,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
) -> int:
    """Write crosswalk rows as Parquet. Requires ``biosemalign[data]``."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Parquet output requires pyarrow. Install it with: pip install 'biosemalign[data]'"
        ) from exc

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    target = Path(path)
    ensure_directory(target.parent, dir_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    os.chmod(temporary, file_mode)
    schema = pa.schema([(name, pa.string()) for name in CROSSWALK_COLUMNS])
    count = 0
    writer = None
    chunk: list[dict[str, Any]] = []
    try:
        writer = pq.ParquetWriter(temporary, schema)
        for row in rows:
            chunk.append({k: row.get(k) for k in CROSSWALK_COLUMNS})
            if len(chunk) >= batch_size:
                writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
                count += len(chunk)
                chunk.clear()
        if chunk:
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
            count += len(chunk)
        writer.close()
        writer = None
        # PyArrow has closed its descriptor, so explicitly sync the completed
        # temporary artifact before making the atomic rename visible.
        sync_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(sync_descriptor)
        finally:
            os.close(sync_descriptor)
        os.replace(temporary, target)
        os.chmod(target, file_mode)
        _fsync_directory(target.parent)
        return count
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def _csv_safe(value: Any) -> Any:
    """Prevent spreadsheet formula execution without altering typed outputs."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:  # pragma: no cover - unsupported by some non-POSIX filesystems
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
