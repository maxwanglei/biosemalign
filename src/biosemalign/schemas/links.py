"""Typed links from matched codes to target codes (v0.3).

Coding attaches codes; linking says where each code leads in the target
terminology, at the requested granularity, and how: as the same entity
(``EQUIVALENCE``) or by a change of level (``PROJECTION``). A code that
leads to more than one distinct set of targets is reported as ambiguous,
never collapsed to one.

The kernel derives a package's ``LinkSet`` from the package itself — its
concordance groups and the target-family concepts' edges — after the
leakage filter, and artifact verification derives it again. It is a typed,
auditable view of evidence, not an assertion a linker can make up.
"""

from __future__ import annotations

from pydantic import Field

from biosemalign.enums import LinkKind
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.codes import QualifiedCode

__all__ = ["CodeLink", "LinkSet"]


class CodeLink(BioSemAlignModel):
    """One matched code's route to one target code."""

    source: QualifiedCode = Field(description="The matched code.")
    via: QualifiedCode = Field(
        description="The target-family concept the source is, or is equivalent to."
    )
    target: QualifiedCode = Field(description="The target code at the requested granularity.")
    kind: LinkKind
    predicates: list[str] = Field(
        default_factory=list, description="Relationships walked from ``via`` to ``target``."
    )
    assertion_sources: list[str] = Field(
        default_factory=list, description="Who asserted or derived the walked relationships."
    )


class LinkSet(BioSemAlignModel):
    """Every matched code's links into the target, for one SAU."""

    target_spec: str = Field(description="The target spec the links follow, e.g. 'rxnorm@1.0'.")
    target_spec_sha256: str = Field(min_length=64, max_length=64)
    granularity: str | None = None
    links: list[CodeLink] = Field(default_factory=list)
    ambiguous: list[QualifiedCode] = Field(
        default_factory=list,
        description="Codes whose target-family concepts lead to different sets of targets.",
    )

    def targets_of(self, vocabulary: str, concept_code: str) -> list[QualifiedCode]:
        """Targets reached through one target-family concept, in link order."""
        wanted = (vocabulary.casefold(), concept_code)
        seen: list[QualifiedCode] = []
        for link in self.links:
            if (link.via.vocabulary.casefold(), link.via.concept_code) != wanted:
                continue
            if all(
                (t.vocabulary, t.concept_code) != (link.target.vocabulary, link.target.concept_code)
                for t in seen
            ):
                seen.append(link.target)
        return seen
