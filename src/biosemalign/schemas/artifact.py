"""Complete, auditable pipeline and batch records.

An :class:`AlignmentDecision` is intentionally compact.  It is not, by itself,
enough to reproduce or review a mapping because it omits the source context,
retrieved alternatives, and pre-validation proposal.  These contracts keep
that full evidence chain together in JSONL output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from biosemalign.enums import (
    ConceptStatus,
    DecisionType,
    RelationshipType,
    RoutingStatus,
    Severity,
    ValidationStatus,
)
from biosemalign.schemas.base import VersionedModel
from biosemalign.schemas.decision import (
    AlignmentDecision,
    AlignmentProposal,
    ReviewQueueItem,
    RoutingFeatures,
    SelectedConcept,
    ValidationResult,
)
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.provenance import ResourceProvenance, require_unique_resources
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import normalize_text, stable_id

__all__ = ["AlignmentArtifact", "BatchAlignmentRecord", "ReviewArtifact"]


class AlignmentArtifact(VersionedModel):
    """Every material input and output from one alignment attempt."""

    artifact_id: str = Field(description="Unique identifier for this run-specific artifact.")
    run_id: str
    mapping_key: str
    sau: SemanticAlignmentUnit
    knowledge_package: KnowledgePackage
    proposal: AlignmentProposal
    validation: ValidationResult
    decision: AlignmentDecision

    @model_validator(mode="after")
    def _identities_are_consistent(self) -> AlignmentArtifact:
        sau_id = self.sau.sau_id
        identities = {
            "knowledge_package": self.knowledge_package.sau_id,
            "proposal": self.proposal.sau_id,
            "decision": self.decision.sau_id,
        }
        mismatches = {name: value for name, value in identities.items() if value != sau_id}
        if mismatches:
            raise ValueError(
                f"artifact components do not share SAU identity {sau_id!r}: {mismatches}"
            )
        if self.mapping_key != self.sau.mapping_key:
            raise ValueError("artifact mapping_key does not match its SAU")
        if self.run_id != self.decision.provenance.run_id:
            raise ValueError("artifact run_id does not match decision provenance")
        if self.run_id != self.sau.provenance.run_id:
            raise ValueError("artifact run_id does not match SAU provenance")
        if self.validation != self.decision.validation:
            raise ValueError("artifact validation does not match the routed decision")
        _validate_provenance_consistency(self)
        if self.decision.source_value != self.sau.source.raw_value:
            raise ValueError("decision source value does not match the SAU source")
        if self.decision.record_id != self.sau.source.record_id:
            raise ValueError("decision record ID does not match the SAU source")
        canonical_attributes = self.sau.attributes.model_dump(
            exclude_none=True, exclude={"extensions"}
        )
        if self.decision.preserved_attributes != canonical_attributes:
            raise ValueError("decision preserved attributes do not match the SAU")

        if self.validation.status is ValidationStatus.REJECTED:
            if not _is_sanitized_rejection(self):
                raise ValueError("rejected proposals must be sanitized and routed to review")
            _validate_verdict(self)
            _validate_artifact_id(self)
            return self

        _validate_verdict(self)
        if self.proposal.preserved_attributes != canonical_attributes:
            raise ValueError("validated proposal preserved attributes do not match the SAU")

        proposal_ids = [concept.concept_id for concept in self.proposal.selected_concepts]
        decision_ids = [concept.concept_id for concept in self.decision.selected_concepts]
        if proposal_ids != decision_ids:
            raise ValueError("decision selections do not match the validated proposal")
        if self.decision.decision is not self.proposal.decision:
            raise ValueError("decision outcome does not match the validated proposal")
        if self.decision.relationship_to_source is not self.proposal.relationship_to_source:
            raise ValueError("decision relationship does not match the validated proposal")
        for name in ("source_level_concept", "analysis_level_concept"):
            proposed = getattr(self.proposal, name)
            decided = getattr(self.decision, name)
            if (proposed.concept_id if proposed else None) != (
                decided.concept_id if decided else None
            ):
                raise ValueError(f"decision {name} does not match the validated proposal")

        for name in (
            "semantic_loss",
            "unsupported_inferences",
            "missing_information",
            "evidence_references",
            "rationale",
        ):
            if getattr(self.decision, name) != getattr(self.proposal, name):
                raise ValueError(f"decision {name} does not match the validated proposal")

        _validate_canonical_decision_concepts(self)
        _validate_evidence_references(self)
        _validate_routing_features(self)
        _validate_artifact_id(self)
        return self


def _validate_provenance_consistency(artifact: AlignmentArtifact) -> None:
    sau_provenance = artifact.sau.provenance
    decision_provenance = artifact.decision.provenance
    for field_name in (
        "package_version",
        "profile_name",
        "profile_version",
        "source_adapter",
        "source_adapter_version",
        "identifier_hmac_key_fingerprint",
    ):
        if getattr(sau_provenance, field_name) != getattr(decision_provenance, field_name):
            raise ValueError(f"decision provenance {field_name} does not match SAU provenance")
    if decision_provenance.validation_rules_version != artifact.validation.rules_version:
        raise ValueError("decision provenance validation rules do not match artifact validation")

    decided_resources = {
        (resource.resource_name.casefold(), resource.provider_name.casefold()): resource
        for resource in decision_provenance.resources
    }
    for packaged in artifact.knowledge_package.resource_provenance:
        decided = decided_resources.get(
            (packaged.resource_name.casefold(), packaged.provider_name.casefold())
        )
        if decided is None:
            raise ValueError("decision provenance omits a Knowledge Package resource")
        for field_name in (
            "provider_version",
            "terminology_release",
            "api_version",
            "mode",
        ):
            if getattr(packaged, field_name) != getattr(decided, field_name):
                raise ValueError(
                    f"decision resource provenance {field_name} does not match the package"
                )
        if (
            decided.cache_hits < packaged.cache_hits
            or decided.network_calls < packaged.network_calls
        ):
            raise ValueError("decision resource counters precede the Knowledge Package snapshot")


def _validate_verdict(artifact: AlignmentArtifact) -> None:
    findings = artifact.validation.findings
    if any(finding.severity is Severity.ERROR for finding in findings):
        expected = ValidationStatus.REJECTED
    elif artifact.proposal.requested_human_review:
        expected = ValidationStatus.HUMAN_REVIEW_REQUIRED
    elif any(finding.severity is Severity.WARNING for finding in findings):
        expected = ValidationStatus.VALID_WITH_WARNING
    elif artifact.proposal.semantic_loss:
        expected = ValidationStatus.VALID_WITH_SEMANTIC_LOSS
    else:
        expected = ValidationStatus.VALID
    if artifact.validation.status is not expected:
        raise ValueError("artifact validation status does not match findings and proposal")


def _is_sanitized_rejection(artifact: AlignmentArtifact) -> bool:
    decision = artifact.decision
    expected_features = RoutingFeatures(
        relationship_type=RelationshipType.UNSUPPORTED_MAPPING,
        missing_attributes=list(artifact.sau.readiness.missing_context),
        validator_error_count=len(artifact.validation.errors),
        validator_warning_count=len(artifact.validation.warnings),
    )
    expected_reasons = [
        f"validation rejected the proposal: {finding.code}"
        for finding in artifact.validation.errors
    ]
    return (
        decision.decision is DecisionType.ABSTAINED
        and decision.relationship_to_source is RelationshipType.UNSUPPORTED_MAPPING
        and not decision.selected_concepts
        and decision.source_level_concept is None
        and decision.analysis_level_concept is None
        and not decision.semantic_loss
        and not decision.unsupported_inferences
        and decision.missing_information == list(artifact.sau.readiness.missing_context)
        and not decision.evidence_references
        and decision.rationale is None
        and decision.routing_status is RoutingStatus.HUMAN_REVIEW
        and decision.routing_features == expected_features
        and decision.routing_reasons == expected_reasons
    )


def _validate_canonical_decision_concepts(artifact: AlignmentArtifact) -> None:
    target_vocabulary = artifact.sau.task.target_terminology
    verified_active = {
        finding.concept_id
        for finding in artifact.validation.findings
        if finding.code == "STATUS_VERIFIED_ACTIVE" and finding.concept_id is not None
    }

    def canonical(proposed: SelectedConcept | None) -> SelectedConcept | None:
        if proposed is None:
            return None
        evidence = artifact.knowledge_package.evidence_for(target_vocabulary, proposed.concept_id)
        if evidence is None:
            raise ValueError("validated proposal concept is absent from canonical package evidence")
        if proposed.vocabulary.casefold() != evidence.vocabulary.casefold():
            raise ValueError("validated proposal concept vocabulary is not canonical")
        if evidence.label is not None and normalize_text(proposed.label) != normalize_text(
            evidence.label
        ):
            raise ValueError("validated proposal concept label is not canonical")
        if (
            proposed.concept_class is not None
            and evidence.concept_class is not None
            and proposed.concept_class.casefold() != evidence.concept_class.casefold()
        ):
            raise ValueError("validated proposal concept class is not canonical")
        if proposed.status is not ConceptStatus.UNKNOWN and proposed.status is not evidence.status:
            raise ValueError("validated proposal concept status is not canonical")
        status = ConceptStatus.ACTIVE if proposed.concept_id in verified_active else evidence.status
        return SelectedConcept(
            concept_id=evidence.concept_id,
            label=evidence.label or "[label unavailable in bounded evidence]",
            vocabulary=evidence.vocabulary,
            concept_class=evidence.concept_class,
            status=status,
            relationship_to_source=proposed.relationship_to_source,
            note=proposed.note,
        )

    expected_selected = [canonical(concept) for concept in artifact.proposal.selected_concepts]
    if artifact.decision.selected_concepts != expected_selected:
        raise ValueError("decision selected concept metadata is not canonical package evidence")
    if artifact.decision.source_level_concept != canonical(artifact.proposal.source_level_concept):
        raise ValueError("decision source-level concept metadata is not canonical package evidence")
    if artifact.decision.analysis_level_concept != canonical(
        artifact.proposal.analysis_level_concept
    ):
        raise ValueError(
            "decision analysis-level concept metadata is not canonical package evidence"
        )


def _validate_evidence_references(artifact: AlignmentArtifact) -> None:
    allowed = {
        f"{statement.subject_id}--{statement.predicate}->{statement.object_id}"
        for candidate in artifact.knowledge_package.candidates
        for statement in candidate.relationships
    }
    allowed.update(policy.policy_id for policy in artifact.knowledge_package.coding_policies)
    unexpected = set(artifact.proposal.evidence_references) - allowed
    if unexpected:
        raise ValueError(
            f"proposal evidence references are absent from the package: {sorted(unexpected)}"
        )


def _validate_routing_features(artifact: AlignmentArtifact) -> None:
    features = artifact.decision.routing_features
    if features is None:
        raise ValueError("auditable decisions require routing features")
    # Keep the persisted audit vector coupled to the same evidence-derived
    # calculation used by the router. Checking only the relationship and
    # lifecycle fields would still permit forged match, rank, channel, or
    # lexical signals to survive serialization.
    from biosemalign.routing.features import extract_features

    expected = extract_features(
        sau=artifact.sau,
        package=artifact.knowledge_package,
        proposal=artifact.proposal,
        validation=artifact.validation,
    )
    if features != expected:
        raise ValueError("routing features do not match canonical artifact evidence")
    if (
        artifact.validation.status
        in {
            ValidationStatus.HUMAN_REVIEW_REQUIRED,
            ValidationStatus.VALID_WITH_WARNING,
        }
        and artifact.decision.routing_status is not RoutingStatus.HUMAN_REVIEW
    ):
        raise ValueError("review-required validation status must route to human review")


def _validate_artifact_id(artifact: AlignmentArtifact) -> None:
    expected = stable_id(
        "ART",
        {
            "run_id": artifact.run_id,
            "sau": artifact.sau.model_dump(mode="json"),
            "knowledge_package": artifact.knowledge_package.model_dump(mode="json"),
            "proposal": artifact.proposal.model_dump(mode="json"),
            "validation": artifact.validation.model_dump(mode="json"),
            "decision": artifact.decision.model_dump(mode="json"),
        },
        length=32,
    )
    if artifact.artifact_id != expected:
        raise ValueError("artifact_id does not match its canonical evidence payload")


class BatchAlignmentRecord(VersionedModel):
    """Auditable outcome for one source row.

    Blank and failed rows are records rather than disappearing from output.
    Successful records carry a complete :class:`AlignmentArtifact`.
    """

    row_number: int = Field(ge=1)
    status: Literal["success", "error", "skipped"]
    source_value: str | None = None
    record_id: str | None = None
    artifact: AlignmentArtifact | None = None
    resource_provenance: list[ResourceProvenance] = Field(
        default_factory=list,
        description="Cumulative resource counters after this row, including failed rows.",
    )
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _payload_matches_status(self) -> BatchAlignmentRecord:
        require_unique_resources(self.resource_provenance)
        if self.status == "success" and self.artifact is None:
            raise ValueError("successful batch records require an artifact")
        if self.status == "success" and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("successful batch records must not carry error fields")
        if self.status != "success" and self.artifact is not None:
            raise ValueError("only successful batch records may carry an artifact")
        if self.status in {"error", "skipped"} and not self.error_code:
            raise ValueError(f"{self.status} batch records require an error_code")
        if self.artifact is not None:
            if self.source_value != self.artifact.sau.source.raw_value:
                raise ValueError("batch source value does not match its artifact")
            if self.record_id != self.artifact.sau.source.record_id:
                raise ValueError("batch record ID does not match its artifact")
            source_row_number = self.artifact.sau.source.row.get("source_row_number")
            if source_row_number is None:
                raise ValueError("batch artifact SAU has no canonical source row number")
            if isinstance(source_row_number, bool) or str(source_row_number) != str(
                self.row_number
            ):
                raise ValueError("batch row number does not match its artifact occurrence")
            decision_resources = self.artifact.decision.provenance.resources
            if not self.resource_provenance:
                object.__setattr__(
                    self,
                    "resource_provenance",
                    [resource.model_copy(deep=True) for resource in decision_resources],
                )
            snapshots = {
                (resource.resource_name, resource.provider_name): resource
                for resource in self.resource_provenance
            }
            for decision_resource in decision_resources:
                snapshot = snapshots.get(
                    (decision_resource.resource_name, decision_resource.provider_name)
                )
                if snapshot is None:
                    raise ValueError("batch resource snapshot omits a decision resource")
                if (
                    snapshot.terminology_release != decision_resource.terminology_release
                    or snapshot.provider_version != decision_resource.provider_version
                    or snapshot.mode is not decision_resource.mode
                    or snapshot.cache_hits < decision_resource.cache_hits
                    or snapshot.network_calls < decision_resource.network_calls
                ):
                    raise ValueError("batch resource snapshot contradicts decision provenance")
        return self


class ReviewArtifact(VersionedModel):
    """Human-review queue entry paired with its complete evidence artifact."""

    artifact: AlignmentArtifact
    review_item: ReviewQueueItem

    @model_validator(mode="after")
    def _review_identity_matches(self) -> ReviewArtifact:
        if self.review_item.sau_id != self.artifact.sau.sau_id:
            raise ValueError("review item does not refer to the artifact SAU")
        if self.review_item.decision != self.artifact.decision:
            raise ValueError("review item decision does not match the artifact decision")
        if self.review_item.source_value != self.artifact.sau.source.raw_value:
            raise ValueError("review source value does not match the artifact SAU")
        if self.review_item.mention_text != self.artifact.sau.mention.text:
            raise ValueError("review mention text does not match the artifact SAU")

        expected_candidates = [
            _review_candidate(candidate) for candidate in self.artifact.knowledge_package.candidates
        ]
        if self.review_item.candidates_considered != expected_candidates:
            raise ValueError("review candidates do not match canonical package candidates")
        expected_messages = [
            f"{finding.code}: {finding.message}" for finding in self.artifact.validation.findings
        ]
        if self.review_item.validator_messages != expected_messages:
            raise ValueError("review validator messages do not match artifact validation")
        if self.review_item.reason_for_review != self.artifact.decision.routing_reasons:
            raise ValueError("review reasons do not match artifact routing reasons")
        expected_priority = len(self.artifact.validation.errors) * 10 + len(
            self.artifact.validation.warnings
        )
        if self.review_item.priority != expected_priority:
            raise ValueError("review priority does not match artifact validation severity")
        return self


def _review_candidate(candidate: CandidateConcept) -> SelectedConcept:
    return SelectedConcept(
        concept_id=candidate.candidate_id,
        label=candidate.preferred_label,
        vocabulary=candidate.vocabulary,
        concept_class=candidate.concept_class,
        status=candidate.status,
        note=(
            "channels: "
            + ", ".join(sorted(channel.value for channel in candidate.channels))
            + f"; best score {candidate.best_score:.1f}"
        ),
    )
