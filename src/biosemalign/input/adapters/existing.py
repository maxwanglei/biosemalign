"""Compatibility boundary around the current cleaner.

The caller configures and owns the pipeline. This adapter deliberately adds no
normalization, policy decisions, filtering, or report deduplication: its M0
contract is equivalence to the existing row-preserving batch interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biosemalign.pipelines.base import BatchInputError, BioSemAlign
    from biosemalign.schemas.artifact import BatchAlignmentRecord
    from biosemalign.schemas.provenance import RunContext

__all__ = ["ExistingCleanerAdapter"]


class ExistingCleanerAdapter:
    """Expose the current cleaner while retaining every source-row outcome.

    Unlike ``align_batch()``, ``align_batch_records()`` includes skipped and
    failed rows. Closing the pipeline remains the caller's responsibility.
    """

    def __init__(self, cleaner: BioSemAlign) -> None:
        self.cleaner = cleaner

    def clean_records(
        self,
        rows: Iterable[dict[str, Any] | BatchInputError],
        *,
        value_column: str,
        id_columns: list[str] | None = None,
        context_columns: list[str] | None = None,
        start_row: int = 1,
        run: RunContext | None = None,
    ) -> Iterator[BatchAlignmentRecord]:
        """Stream existing results unchanged, including errors and blank rows."""
        yield from self.cleaner.align_batch_records(
            rows,
            value_column=value_column,
            id_columns=id_columns,
            context_columns=context_columns,
            start_row=start_row,
            run=run,
        )
