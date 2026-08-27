"""Module 5: confidence calibration and decision routing.

Assembles the final :class:`AlignmentDecision` — the proposal, the validator's
verdict, the routing outcome, and the provenance needed to explain all three.
"""

from __future__ import annotations

from biosemalign.enums import ConceptStatus, DecisionType, RelationshipType, ValidationStatus
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.routing.features import extract_features
from biosemalign.routing.policy import ROUTING_POLICY_VERSION, route_decision
from biosemalign.schemas.artifact import AlignmentArtifact, ReviewArtifact
from biosemalign.schemas.decision import (
    AlignmentDecision,
    AlignmentProposal,
    ReviewQueueItem,
    RoutingFeatures,
    SelectedConcept,
    ValidationResult,
)
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.provenance import Provenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["DecisionRouter", "to_review_artifact", "to_review_item"]

log = get_logger(__name__)


class DecisionRouter:
    """Turns a validated proposal into a routed, provenance-stamped decision."""

    version = ROUTING_POLICY_VERSION

    def __init__(self, profile: TaskProfile) -> None:
        self.profile = profile

    def route(
        self,
        *,
        sau: SemanticAlignmentUnit,
        package: KnowledgePackage,
        proposal: AlignmentProposal,
        validation: ValidationResult,
        provenance: Provenance,
    ) -> AlignmentDecision:
        features = extract_features(
            sau=sau, package=package, proposal=proposal, validation=validation
        )
        status, reasons = route_decision(
            proposal=proposal,
            validation=validation,
            features=features,
            package=package,
            profile=self.profile,
        )
        selected, source_level, analysis_level = _canonical_decision_concepts(
            proposal,
            package,
            validation,
            target_vocabulary=self.profile.task.target_terminology,
        )
        rejected = validation.status is ValidationStatus.REJECTED
        if rejected:
            features = RoutingFeatures(
                relationship_type=RelationshipType.UNSUPPORTED_MAPPING,
                missing_attributes=list(sau.readiness.missing_context),
                validator_error_count=len(validation.errors),
                validator_warning_count=len(validation.warnings),
            )

        log.info(
            "decision.routed",
            sau_id=sau.sau_id,
            decision=(DecisionType.ABSTAINED if rejected else proposal.decision).value,
            relationship=(
                RelationshipType.UNSUPPORTED_MAPPING
                if rejected
                else proposal.relationship_to_source
            ).value,
            validation=validation.status.value,
            routing=status.value,
        )

        return AlignmentDecision(
            sau_id=sau.sau_id,
            decision=DecisionType.ABSTAINED if rejected else proposal.decision,
            source_value=sau.source.raw_value,
            record_id=sau.source.record_id,
            selected_concepts=selected,
            source_level_concept=source_level,
            analysis_level_concept=analysis_level,
            relationship_to_source=(
                RelationshipType.UNSUPPORTED_MAPPING
                if rejected
                else proposal.relationship_to_source
            ),
            preserved_attributes=sau.attributes.model_dump(
                exclude_none=True, exclude={"extensions"}
            ),
            semantic_loss=[] if rejected else proposal.semantic_loss,
            unsupported_inferences=[] if rejected else proposal.unsupported_inferences,
            missing_information=(
                list(sau.readiness.missing_context) if rejected else proposal.missing_information
            ),
            evidence_references=[] if rejected else proposal.evidence_references,
            rationale=None if rejected else proposal.rationale,
            validation=validation,
            routing_status=status,
            routing_features=features,
            routing_reasons=reasons,
            provenance=provenance,
        )


def _canonical_decision_concepts(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    validation: ValidationResult,
    *,
    target_vocabulary: str,
) -> tuple[list[SelectedConcept], SelectedConcept | None, SelectedConcept | None]:
    """Return evidence-owned metadata and suppress rejected mapping claims."""
    if validation.status is ValidationStatus.REJECTED:
        return [], None, None

    verified_active = {
        finding.concept_id
        for finding in validation.findings
        if finding.code == "STATUS_VERIFIED_ACTIVE" and finding.concept_id is not None
    }
    evidence_index = package.qualified_concept_evidence_index()
    canonical_vocabulary = target_vocabulary.casefold()

    def canonical(concept: SelectedConcept | None) -> SelectedConcept | None:
        if concept is None:
            return None
        evidence = evidence_index.get((canonical_vocabulary, concept.concept_id))
        if evidence is None:
            # This is reachable only for profiles that explicitly disable the
            # unretrieved-identifier guard. Keep the claim out of a final
            # decision; the complete proposal remains in AlignmentArtifact.
            return None
        status = evidence.status
        if concept.concept_id in verified_active:
            status = ConceptStatus.ACTIVE
        return SelectedConcept(
            concept_id=evidence.concept_id,
            label=evidence.label or "[label unavailable in bounded evidence]",
            vocabulary=evidence.vocabulary,
            concept_class=evidence.concept_class,
            status=status,
            relationship_to_source=concept.relationship_to_source,
            note=concept.note,
        )

    selected: list[SelectedConcept] = []
    for concept in proposal.selected_concepts:
        item = canonical(concept)
        if item is not None:
            selected.append(item)
    return (
        selected,
        canonical(proposal.source_level_concept),
        canonical(proposal.analysis_level_concept),
    )


def to_review_item(
    decision: AlignmentDecision,
    sau: SemanticAlignmentUnit,
    package: KnowledgePackage,
) -> ReviewQueueItem:
    """Package a decision for AL-MedLit review (§10.1).

    Carries every candidate that was considered, not merely the one chosen — a
    reviewer who can only see the selected concept cannot tell whether the
    right answer was available and passed over.
    """
    considered = [
        SelectedConcept(
            concept_id=c.candidate_id,
            label=c.preferred_label,
            vocabulary=c.vocabulary,
            concept_class=c.concept_class,
            status=c.status,
            note=(
                "channels: "
                + ", ".join(sorted(ch.value for ch in c.channels))
                + f"; best score {c.best_score:.1f}"
            ),
        )
        for c in package.candidates
    ]

    # Reviewer effort is finite: rank by how much is at stake and how uncertain
    # the pipeline was.
    priority = len(decision.validation.errors) * 10 + len(decision.validation.warnings)

    return ReviewQueueItem(
        sau_id=decision.sau_id,
        priority=priority,
        decision=decision,
        source_value=sau.source.raw_value,
        mention_text=sau.mention.text,
        candidates_considered=considered,
        validator_messages=[f"{f.code}: {f.message}" for f in decision.validation.findings],
        reason_for_review=decision.routing_reasons,
    )


def to_review_artifact(artifact: AlignmentArtifact) -> ReviewArtifact:
    """Prepare human review without discarding retrieved evidence."""
    return ReviewArtifact(
        artifact=artifact,
        review_item=to_review_item(
            artifact.decision,
            artifact.sau,
            artifact.knowledge_package,
        ),
    )
