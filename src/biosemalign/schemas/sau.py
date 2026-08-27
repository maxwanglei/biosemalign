"""The Semantic Alignment Unit — the framework's central data structure.

An SAU says *what needs to be mapped* and carries everything required to map it
correctly. It deliberately does not say what the answer is; concept selection
happens in Module 3.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from biosemalign.enums import ReadinessStatus, SemanticType
from biosemalign.schemas.base import BioSemAlignModel, ExtensibleModel, VersionedModel
from biosemalign.schemas.provenance import Provenance
from biosemalign.schemas.request import MappingTask
from biosemalign.schemas.source import SourceRecord

__all__ = [
    "ContextEnvelope",
    "ContextualAttributes",
    "Mention",
    "ReadinessAssessment",
    "SemanticAlignmentUnit",
]


class Mention(BioSemAlignModel):
    """The span to be mapped, plus its provisional typing (§5.5)."""

    text: str = Field(description="The mention exactly as it appears in the source.")
    normalized_text: str = Field(description="Whitespace-collapsed, case-folded form.")
    start: int | None = Field(default=None, description="Character offset into the source text.")
    end: int | None = None
    semantic_types: list[SemanticType] = Field(
        default_factory=list,
        description=(
            "Provisional types. More than one is permitted and expected when the "
            "evidence is genuinely ambiguous (§5.5) — collapsing to a single "
            "guess here would hide the ambiguity from the adjudicator."
        ),
    )
    detection_method: str = Field(
        default="user_specified",
        description="How this span was identified, e.g. 'user_specified', 'profile_default'.",
    )


class ContextEnvelope(ExtensibleModel):
    """Everything surrounding the mention (§5.4).

    Structured and free-text context share one envelope because the same SAU
    schema serves both input families; irrelevant fields simply stay ``None``.
    """

    # Structured provenance context
    source_system: str | None = None
    table_name: str | None = None
    column_name: str | None = None
    field_description: str | None = None
    data_dictionary_definition: str | None = None
    record_id: str | None = None
    neighboring_fields: dict[str, Any] = Field(default_factory=dict)

    # Product / clinical qualifiers carried by the record
    dose: str | None = None
    unit: str | None = None
    route: str | None = None
    drug_role: str | None = None
    indication: str | None = None
    date: str | None = None
    species: str | None = None

    # Free-text context
    sentence: str | None = None
    paragraph: str | None = None
    section_heading: str | None = None
    document_type: str | None = None
    table_caption: str | None = None
    footnote: str | None = None
    study_design: str | None = None
    evidence_block_id: str | None = None


class ContextualAttributes(ExtensibleModel):
    """Modifiers asserted *about the mention* (§5.6).

    Kept separate from concept identity on purpose. "Suspected adrenocortical
    carcinoma" maps to an ACC concept with ``certainty='suspected'``; folding
    the qualifier into the concept would silently convert a suspicion into a
    confirmed diagnosis.

    Several names also appear on :class:`ContextEnvelope`. That is not
    duplication: the envelope records what the surrounding record said, while
    these record what was asserted about this particular mention.
    """

    negation: bool | None = None
    certainty: str | None = None
    assertion_status: str | None = None
    temporality: str | None = None
    severity: str | None = None
    species: str | None = None
    anatomical_location: str | None = None
    dose: str | None = None
    strength: str | None = None
    route: str | None = None
    formulation: str | None = None
    specimen: str | None = None
    assay_method: str | None = None
    unit: str | None = None
    experimental_system: str | None = None


class ReadinessAssessment(BioSemAlignModel):
    """Whether this SAU can be mapped, and what is missing if not (§5.8)."""

    status: ReadinessStatus
    missing_context: list[str] = Field(
        default_factory=list,
        description="Context fields the profile wanted but the source did not supply.",
    )
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_mappable(self) -> bool:
        return self.status in {
            ReadinessStatus.READY_FOR_MAPPING,
            ReadinessStatus.READY_WITH_LIMITED_CONTEXT,
        }


class SemanticAlignmentUnit(VersionedModel):
    """One thing to be mapped, with its context preserved (§11.2)."""

    sau_id: str
    mapping_key: str = Field(
        description=(
            "Profile-defined, context-aware key used to reuse equivalent mappings. "
            "Unlike sau_id, this intentionally excludes source-row identity."
        )
    )
    source: SourceRecord
    mention: Mention
    context: ContextEnvelope
    attributes: ContextualAttributes
    task: MappingTask
    provenance: Provenance
    readiness: ReadinessAssessment
