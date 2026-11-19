"""Vocabulary-qualified codes and their equivalence groups (v0.2).

Kept apart from the knowledge package so that provenance — which records what
each workflow tier matched — can use them without importing the package.
"""

from __future__ import annotations

from pydantic import Field

from biosemalign.schemas.base import BioSemAlignModel

__all__ = ["EquivalenceGroup", "QualifiedCode"]


class QualifiedCode(BioSemAlignModel):
    """A concept identified by vocabulary and code, e.g. ``UNII:GEB06NHM23``."""

    vocabulary: str = Field(min_length=1)
    concept_code: str = Field(min_length=1)
    label: str | None = None


class EquivalenceGroup(BioSemAlignModel):
    """Exact matches that permitted equivalence keys resolve to one entity."""

    members: list[QualifiedCode] = Field(min_length=1)
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Independent sources asserting the matched names, canonicalized so that a "
            "redistributed atom counts once under its originating source."
        ),
    )
    conflict: bool = Field(
        default=False,
        description=(
            "True when the group holds two codes from one vocabulary — or, when the "
            "profile names conflict vocabularies, two codes from that pool."
        ),
    )
