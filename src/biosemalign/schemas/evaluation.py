"""Versioned contracts for reproducible benchmark evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from biosemalign.schemas.base import VersionedModel

__all__ = ["GoldMapping"]


class GoldMapping(VersionedModel):
    """One expert-adjudicated benchmark case.

    ``case_id`` is the join key. Reported biomedical values are not unique, so
    matching gold and predictions by ``source_value`` can silently discard or
    double-count repeated observations. ``context`` preserves the evidence
    available to the annotator without making that evidence part of the join.
    """

    case_id: str = Field(description="Stable, unique identifier for this benchmark case.")
    source_value: str = Field(description="The source value shown to the alignment system.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Source context available when the gold answer was adjudicated.",
    )
    expected_concept_ids: list[str] = Field(
        description="Expected target identifiers; empty means no valid mapping."
    )
    category: str | None = Field(
        default=None,
        description="Benchmark slice used for category-level reporting.",
    )
    note: str | None = None

    @field_validator("case_id")
    @classmethod
    def _require_stable_case_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        if value != value.strip():
            raise ValueError("must not contain surrounding whitespace")
        return value

    @field_validator("source_value")
    @classmethod
    def _require_nonblank_source_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_concept_ids")
    @classmethod
    def _require_distinct_concept_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("expected concept identifiers must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("expected concept identifiers must be unique")
        return values
