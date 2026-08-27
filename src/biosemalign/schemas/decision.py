"""Adjudication proposals, validated decisions, and reviewer adjudications.

The split between :class:`AlignmentProposal` and :class:`AlignmentDecision` is
deliberate and load-bearing. A proposal is what the adjudicator claims; a
decision is what survived deterministic validation and routing. Section 2 of
the design gives final authority to the validator, and keeping the two types
distinct makes it impossible to accidentally serialize an unvalidated
adjudicator claim as a finished mapping.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from biosemalign.enums import (
    ConceptStatus,
    DecisionType,
    ErrorCategory,
    RelationshipType,
    RoutingStatus,
    SemanticLossKind,
    Severity,
    ValidationStatus,
)
from biosemalign.schemas.base import BioSemAlignModel, ExtensibleModel, VersionedModel
from biosemalign.schemas.provenance import Provenance

__all__ = [
    "AlignmentDecision",
    "AlignmentProposal",
    "ConceptValidation",
    "ReviewDecision",
    "ReviewQueueItem",
    "RoutingFeatures",
    "SelectedConcept",
    "SemanticLoss",
    "ValidationFinding",
    "ValidationResult",
]


class SelectedConcept(BioSemAlignModel):
    """One concept chosen from the candidate set."""

    concept_id: str
    label: str
    vocabulary: str
    concept_class: str | None = None
    status: ConceptStatus = ConceptStatus.UNKNOWN
    relationship_to_source: RelationshipType = RelationshipType.EXACT
    note: str | None = None


class SemanticLoss(BioSemAlignModel):
    """Information discarded by mapping to a coarser concept (§8.2, goal 7)."""

    kind: SemanticLossKind
    description: str
    source_value: str | None = Field(
        default=None, description="The source detail that the target concept does not carry."
    )


class ValidationFinding(BioSemAlignModel):
    """One thing the deterministic validator noticed."""

    code: str = Field(description="Stable machine-readable code, e.g. 'UNRETRIEVED_IDENTIFIER'.")
    severity: Severity
    message: str
    concept_id: str | None = None
    error_category: ErrorCategory | None = None


class ValidationResult(BioSemAlignModel):
    """The validator's verdict on a whole proposal (§8.3)."""

    status: ValidationStatus
    findings: list[ValidationFinding] = Field(default_factory=list)
    rules_version: str

    @property
    def errors(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]


class ConceptValidation(BioSemAlignModel):
    """A provider's verdict on a single identifier (§13.1).

    The design document names this ``ValidationResult`` as well; it is renamed
    here because a per-concept existence check and a whole-proposal verdict are
    different things and sharing one name for both invites confusion at exactly
    the point where correctness matters.
    """

    concept_id: str
    exists: bool
    status: ConceptStatus = ConceptStatus.UNKNOWN
    vocabulary: str | None = None
    concept_class: str | None = None
    remapped_to: list[str] = Field(default_factory=list)
    message: str | None = None


class RoutingFeatures(BioSemAlignModel):
    """Observable features backing the routing decision (§9.1).

    Deliberately excludes any self-reported model confidence: the first routing
    policy uses only features that can be checked against the retrieved
    evidence.
    """

    exact_terminology_match: bool = False
    official_synonym_match: bool = False
    official_crosswalk_available: bool = False
    retrieval_rank: int | None = None
    lexical_similarity: float | None = None
    embedding_similarity: float | None = None
    channel_agreement: int = Field(
        default=0, description="Number of distinct retrieval channels that found the selection."
    )
    top_candidate_margin: float | None = Field(
        default=None, description="Score gap between the best and second-best candidate."
    )
    concept_is_active: bool = False
    semantic_type_agreement: bool | None = None
    relationship_type: RelationshipType = RelationshipType.INSUFFICIENT_CONTEXT
    missing_attributes: list[str] = Field(default_factory=list)
    validator_error_count: int = 0
    validator_warning_count: int = 0
    multiple_concepts_required: bool = False
    known_difficult_pattern: str | None = None


class AlignmentProposal(VersionedModel):
    """What the adjudicator proposes, before validation (§7.4).

    Referenced throughout the design (§7.4, Milestone 5) but never given a
    schema; defined here as the pre-validation counterpart of
    :class:`AlignmentDecision`.
    """

    sau_id: str
    decision: DecisionType
    selected_concepts: list[SelectedConcept] = Field(default_factory=list)
    source_level_concept: SelectedConcept | None = Field(
        default=None, description="The concept matching the source at its own granularity."
    )
    analysis_level_concept: SelectedConcept | None = Field(
        default=None, description="The coarser concept the analysis will actually group on."
    )
    relationship_to_source: RelationshipType
    preserved_attributes: dict[str, Any] = Field(default_factory=dict)
    semantic_loss: list[SemanticLoss] = Field(default_factory=list)
    unsupported_inferences: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(
        default_factory=list,
        description="Identifiers of the knowledge statements this proposal relied on.",
    )
    rationale: str | None = None
    requested_human_review: bool = False


class AlignmentDecision(VersionedModel):
    """A validated, routed, provenance-stamped mapping (§11.5)."""

    sau_id: str
    decision: DecisionType
    source_value: str | None = Field(
        default=None,
        description=(
            "The originally reported value, carried through so the crosswalk can be "
            "joined back to report-level source data without a second lookup (§18.3)."
        ),
    )
    record_id: str | None = None
    selected_concepts: list[SelectedConcept] = Field(default_factory=list)
    source_level_concept: SelectedConcept | None = None
    analysis_level_concept: SelectedConcept | None = None
    relationship_to_source: RelationshipType
    preserved_attributes: dict[str, Any] = Field(default_factory=dict)
    semantic_loss: list[SemanticLoss] = Field(default_factory=list)
    unsupported_inferences: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    rationale: str | None = None
    validation: ValidationResult
    routing_status: RoutingStatus
    routing_features: RoutingFeatures | None = None
    routing_reasons: list[str] = Field(default_factory=list)
    provenance: Provenance

    @property
    def validation_status(self) -> ValidationStatus:
        """Convenience accessor matching the §11.5 field name."""
        return self.validation.status


class ReviewDecision(VersionedModel):
    """One AL-MedLit adjudication of a BioSemAlign decision (§11.6)."""

    sau_id: str
    original_decision: AlignmentDecision
    adjudicated_decision: AlignmentDecision
    reviewer_id: str
    review_timestamp: datetime
    correction_reason: str | None = None
    error_categories: list[ErrorCategory] = Field(default_factory=list)
    guideline_version: str | None = None


class ReviewQueueItem(ExtensibleModel):
    """A decision packaged for human review (§10.1).

    Carries the evidence a reviewer needs to adjudicate without re-running the
    pipeline: the candidates considered, not merely the one that was chosen.
    """

    sau_id: str
    priority: int = Field(default=0, description="Higher values are reviewed first.")
    decision: AlignmentDecision
    source_value: str
    mention_text: str
    candidates_considered: list[SelectedConcept] = Field(default_factory=list)
    validator_messages: list[str] = Field(default_factory=list)
    reason_for_review: list[str] = Field(default_factory=list)
