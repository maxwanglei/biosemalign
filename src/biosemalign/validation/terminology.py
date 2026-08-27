"""Terminology and proposal-integrity validation (§8.1).

The Knowledge Package is the authority for concept metadata. Proposal fields
are claims to validate, never an alternative source of labels, classes, or
lifecycle status.
"""

from __future__ import annotations

from collections.abc import Iterable

from biosemalign.enums import (
    Authority,
    ConceptStatus,
    DecisionType,
    ErrorCategory,
    RelationshipType,
    Severity,
)
from biosemalign.knowledge.providers.rxnorm import RxNavProvider
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept, ValidationFinding
from biosemalign.schemas.knowledge import ConceptEvidence, KnowledgePackage
from biosemalign.util import normalize_text

__all__ = ["validate_terminology"]


_GRANULARITY_POLICY: dict[str, tuple[str, frozenset[str]]] = {
    "ingredient": ("has_ingredient", frozenset({"IN"})),
    "precise_ingredient": ("has_precise_ingredient", frozenset({"PIN"})),
    "active_moiety": ("has_boss", frozenset({"IN", "PIN"})),
}
_EMPTY_DECISIONS = {
    DecisionType.NO_VALID_MAPPING,
    DecisionType.INSUFFICIENT_CONTEXT,
    DecisionType.ABSTAINED,
}


def validate_terminology(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
    *,
    provider: RxNavProvider | None = None,
) -> list[ValidationFinding]:
    """Validate proposal structure and every concept-bearing field."""
    findings = _validate_structure(proposal, package)
    evidence_index = package.qualified_concept_evidence_index()
    target_vocabulary = profile.task.target_terminology.casefold()

    findings.extend(_validate_relation_policy(proposal, profile))
    findings.extend(_validate_relationship_coherence(proposal))
    role_findings = _validate_source_role(proposal, package, profile)
    role_findings.extend(_validate_target_roles(proposal, package, profile))
    findings.extend(role_findings)
    invalid_role_ids = {
        finding.concept_id
        for finding in role_findings
        if finding.severity is Severity.ERROR and finding.concept_id is not None
    }
    for field_name, concept in _concept_occurrences(proposal):
        findings.extend(
            _validate_concept(
                field_name,
                concept,
                evidence_index.get((target_vocabulary, concept.concept_id)),
                profile,
                (
                    provider
                    if concept.concept_id not in invalid_role_ids
                    and proposal.decision not in _EMPTY_DECISIONS
                    else None
                ),
            )
        )

    findings.extend(
        _validate_historical_source(proposal, evidence_index, profile.task.target_terminology)
    )
    return _deduplicate(findings)


def _validate_structure(
    proposal: AlignmentProposal, package: KnowledgePackage
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    selected = proposal.selected_concepts

    if proposal.sau_id != package.sau_id:
        findings.append(
            _error(
                "SAU_ID_MISMATCH",
                f"proposal SAU {proposal.sau_id!r} does not match package SAU {package.sau_id!r}",
            )
        )

    if proposal.decision is DecisionType.MAPPED and len(selected) != 1:
        findings.append(
            _error(
                "INVALID_MAPPED_CARDINALITY",
                f"MAPPED requires exactly one selected concept; received {len(selected)}",
            )
        )
    elif proposal.decision is DecisionType.MULTI_CONCEPT_MAPPED and len(selected) < 2:
        findings.append(
            _error(
                "INVALID_MULTI_CONCEPT_CARDINALITY",
                "MULTI_CONCEPT_MAPPED requires at least two selected concepts",
            )
        )
    elif proposal.decision in _EMPTY_DECISIONS and selected:
        findings.append(
            _error(
                "UNEXPECTED_SELECTED_CONCEPTS",
                f"{proposal.decision.value} must not select concepts",
            )
        )

    if proposal.decision in {DecisionType.MAPPED, DecisionType.MULTI_CONCEPT_MAPPED} and (
        proposal.source_level_concept is None
    ):
        findings.append(
            _error(
                "MISSING_SOURCE_LEVEL_CONCEPT",
                f"{proposal.decision.value} requires a source-level concept",
            )
        )

    unresolved_review = (
        proposal.decision is DecisionType.MAPPED
        and proposal.requested_human_review
        and proposal.analysis_level_concept is None
        and proposal.relationship_to_source is RelationshipType.RELATED_NOT_EQUIVALENT
    )
    if (
        proposal.decision is DecisionType.MAPPED
        and proposal.analysis_level_concept is None
        and not unresolved_review
    ):
        findings.append(
            _error(
                "MISSING_ANALYSIS_LEVEL_CONCEPT",
                "MAPPED requires an analysis-level concept unless it is an unresolved review case",
            )
        )
    if proposal.decision in _EMPTY_DECISIONS and (
        proposal.source_level_concept is not None or proposal.analysis_level_concept is not None
    ):
        findings.append(
            _error(
                "UNEXPECTED_CONCEPT_FIELDS",
                f"{proposal.decision.value} must not carry source- or analysis-level concepts",
            )
        )

    selected_ids = [concept.concept_id for concept in selected]
    if len(selected_ids) != len(set(selected_ids)):
        findings.append(_error("DUPLICATE_SELECTED_CONCEPT", "selected concept IDs must be unique"))

    analysis = proposal.analysis_level_concept
    if analysis is not None and analysis.concept_id not in set(selected_ids):
        findings.append(
            _error(
                "ANALYSIS_CONCEPT_NOT_SELECTED",
                f"analysis-level concept {analysis.concept_id} is not in selected_concepts",
                analysis.concept_id,
            )
        )
    if (
        proposal.decision is DecisionType.MAPPED
        and len(selected) == 1
        and analysis is not None
        and selected[0].concept_id != analysis.concept_id
    ):
        findings.append(
            _error(
                "ANALYSIS_SELECTION_MISMATCH",
                "the sole selected concept must be the analysis-level concept for MAPPED",
                analysis.concept_id,
            )
        )
    source = proposal.source_level_concept
    if (
        proposal.relationship_to_source is RelationshipType.EXACT
        and source is not None
        and analysis is not None
        and source.concept_id != analysis.concept_id
    ):
        findings.append(
            _error(
                "EXACT_CONCEPT_ID_MISMATCH",
                "EXACT requires source- and analysis-level concepts to have the same ID",
                analysis.concept_id,
                ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        )
    if (
        proposal.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
        and source is not None
        and analysis is not None
        and source.concept_id == analysis.concept_id
    ):
        findings.append(
            _error(
                "BROADER_CONCEPT_ID_MISMATCH",
                "TARGET_BROADER_THAN_SOURCE requires distinct source- and analysis-level concept IDs",
                analysis.concept_id,
                ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        )
    return findings


def _validate_relation_policy(
    proposal: AlignmentProposal, profile: TaskProfile
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    relations: list[tuple[str, RelationshipType, str | None]] = [
        ("proposal", proposal.relationship_to_source, None)
    ]
    relations.extend(
        ("selected_concepts", concept.relationship_to_source, concept.concept_id)
        for concept in proposal.selected_concepts
    )
    if proposal.source_level_concept is not None:
        relations.append(
            (
                "source_level_concept",
                proposal.source_level_concept.relationship_to_source,
                proposal.source_level_concept.concept_id,
            )
        )
    if proposal.analysis_level_concept is not None:
        relations.append(
            (
                "analysis_level_concept",
                proposal.analysis_level_concept.relationship_to_source,
                proposal.analysis_level_concept.concept_id,
            )
        )

    for field_name, relation, concept_id in relations:
        if profile.permits_relation(relation):
            continue
        findings.append(
            _error(
                "RELATION_NOT_PERMITTED",
                f"{field_name} emits {relation.value}, which profile {profile.profile_name} forbids",
                concept_id,
                ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        )
    return findings


def _validate_relationship_coherence(
    proposal: AlignmentProposal,
) -> list[ValidationFinding]:
    """Ensure nested relationship claims agree with their structural roles."""
    findings: list[ValidationFinding] = []
    top = proposal.relationship_to_source

    if proposal.decision in {DecisionType.NO_VALID_MAPPING, DecisionType.ABSTAINED}:
        if top is not RelationshipType.UNSUPPORTED_MAPPING:
            findings.append(
                _error(
                    "EMPTY_OUTCOME_RELATION_MISMATCH",
                    f"{proposal.decision.value} requires UNSUPPORTED_MAPPING, not {top.value}",
                    category=ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                )
            )
        return findings

    if proposal.decision is DecisionType.INSUFFICIENT_CONTEXT:
        if top is not RelationshipType.INSUFFICIENT_CONTEXT:
            findings.append(
                _error(
                    "INSUFFICIENT_CONTEXT_RELATION_MISMATCH",
                    f"INSUFFICIENT_CONTEXT requires its matching relationship, not {top.value}",
                    category=ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                )
            )
        return findings

    source = proposal.source_level_concept
    analysis = proposal.analysis_level_concept
    if proposal.decision is DecisionType.MULTI_CONCEPT_MAPPED:
        if top is not RelationshipType.MULTIPLE_COMPONENTS_REQUIRED:
            findings.append(
                _error(
                    "MULTI_CONCEPT_RELATION_MISMATCH",
                    "MULTI_CONCEPT_MAPPED requires MULTIPLE_COMPONENTS_REQUIRED",
                    category=ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                )
            )
        if analysis is not None:
            findings.append(
                _error(
                    "UNEXPECTED_MULTI_ANALYSIS_CONCEPT",
                    "multi-concept mappings must represent components in selected_concepts, "
                    "not collapse them into one analysis-level concept",
                    analysis.concept_id,
                    ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                )
            )
        for concept in proposal.selected_concepts:
            if concept.relationship_to_source is not RelationshipType.COMPONENT_OF:
                findings.append(
                    _error(
                        "COMPONENT_RELATION_MISMATCH",
                        f"selected component {concept.concept_id} must use COMPONENT_OF",
                        concept.concept_id,
                        ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                    )
                )
        if source is not None and source.relationship_to_source is not RelationshipType.EXACT:
            findings.append(
                _error(
                    "SOURCE_RELATION_MISMATCH",
                    "the source-level product concept must use EXACT",
                    source.concept_id,
                    ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                )
            )
        return findings

    if proposal.decision is not DecisionType.MAPPED:
        return findings

    reserved = {
        RelationshipType.UNSUPPORTED_MAPPING,
        RelationshipType.INSUFFICIENT_CONTEXT,
        RelationshipType.MULTIPLE_COMPONENTS_REQUIRED,
        RelationshipType.COMPONENT_OF,
    }
    if top in reserved:
        findings.append(
            _error(
                "MAPPED_RELATION_MISMATCH",
                f"MAPPED cannot use structural relationship {top.value}",
                category=ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        )

    unresolved = (
        proposal.requested_human_review
        and analysis is None
        and top is RelationshipType.RELATED_NOT_EQUIVALENT
    )
    expected_source_relation = top if unresolved else RelationshipType.EXACT
    if source is not None and source.relationship_to_source is not expected_source_relation:
        findings.append(
            _error(
                "SOURCE_RELATION_MISMATCH",
                f"source-level concept must use {expected_source_relation.value}",
                source.concept_id,
                ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        )
    for concept in proposal.selected_concepts:
        if concept.relationship_to_source is not top:
            findings.append(
                _error(
                    "SELECTED_RELATION_MISMATCH",
                    f"selected concept {concept.concept_id} uses "
                    f"{concept.relationship_to_source.value}; proposal uses {top.value}",
                    concept.concept_id,
                    ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
                )
            )
    if analysis is not None and analysis.relationship_to_source is not top:
        findings.append(
            _error(
                "ANALYSIS_RELATION_MISMATCH",
                f"analysis-level concept uses {analysis.relationship_to_source.value}; "
                f"proposal uses {top.value}",
                analysis.concept_id,
                ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        )
    return findings


def _concept_occurrences(
    proposal: AlignmentProposal,
) -> Iterable[tuple[str, SelectedConcept]]:
    for concept in proposal.selected_concepts:
        yield "selected_concepts", concept
    if proposal.source_level_concept is not None:
        yield "source_level_concept", proposal.source_level_concept
    if proposal.analysis_level_concept is not None:
        yield "analysis_level_concept", proposal.analysis_level_concept


def _validate_concept(
    field_name: str,
    concept: SelectedConcept,
    evidence: ConceptEvidence | None,
    profile: TaskProfile,
    provider: RxNavProvider | None,
) -> list[ValidationFinding]:
    if evidence is None:
        if not profile.validation.prohibit_unretrieved_identifiers:
            return _validate_target_vocabulary(field_name, concept, profile)
        return [
            _error(
                "UNRETRIEVED_IDENTIFIER",
                f"{field_name} contains {concept.concept_id}, which is absent from the bounded package evidence",
                concept.concept_id,
                ErrorCategory.ADJUDICATION_ERROR,
            )
        ]

    findings = _validate_target_vocabulary(field_name, concept, profile)
    if (
        field_name != "source_level_concept"
        and evidence.status is ConceptStatus.ACTIVE
        and not evidence.retrieved_directly
    ):
        findings.append(
            ValidationFinding(
                code="STATUS_FROM_RELEASE_EVIDENCE",
                severity=Severity.INFO,
                message=(
                    f"official release-bound terminology relationships establish "
                    f"{concept.concept_id} as active"
                ),
                concept_id=concept.concept_id,
            )
        )
    if concept.vocabulary.casefold() != evidence.vocabulary.casefold():
        findings.append(
            _error(
                "CONCEPT_VOCABULARY_MISMATCH",
                f"{field_name} claims {concept.vocabulary} for {concept.concept_id}; package evidence says {evidence.vocabulary}",
                concept.concept_id,
            )
        )
    if evidence.label is not None and normalize_text(concept.label) != normalize_text(
        evidence.label
    ):
        findings.append(
            _error(
                "CONCEPT_LABEL_MISMATCH",
                f"{field_name} label {concept.label!r} does not match package label {evidence.label!r} for {concept.concept_id}",
                concept.concept_id,
            )
        )
    if (
        concept.concept_class is not None
        and evidence.concept_class is not None
        and concept.concept_class.casefold() != evidence.concept_class.casefold()
    ):
        findings.append(
            _error(
                "CONCEPT_CLASS_MISMATCH",
                f"{field_name} class {concept.concept_class!r} does not match package class {evidence.concept_class!r} for {concept.concept_id}",
                concept.concept_id,
            )
        )

    status, status_findings = _canonical_status(field_name, concept, evidence, provider)
    findings.extend(status_findings)
    if profile.validation.require_active_concept and field_name != "source_level_concept":
        if status is ConceptStatus.UNKNOWN:
            findings.append(
                ValidationFinding(
                    code="STATUS_NOT_VERIFIED",
                    severity=Severity.WARNING,
                    message=f"lifecycle status of final concept {concept.concept_id} is not verified",
                    concept_id=concept.concept_id,
                    error_category=ErrorCategory.RESOURCE_COVERAGE_GAP,
                )
            )
        elif not status.is_usable:
            findings.append(
                _error(
                    "INACTIVE_CONCEPT",
                    f"{concept.concept_id} has canonical status {status.value}; the profile requires an active final concept",
                    concept.concept_id,
                )
            )
    return findings


def _validate_target_vocabulary(
    field_name: str, concept: SelectedConcept, profile: TaskProfile
) -> list[ValidationFinding]:
    if concept.vocabulary.casefold() == profile.task.target_terminology.casefold():
        return []
    return [
        _error(
            "WRONG_VOCABULARY",
            f"{field_name} concept {concept.concept_id} belongs to {concept.vocabulary}, but the task targets {profile.task.target_terminology}",
            concept.concept_id,
        )
    ]


def _canonical_status(
    field_name: str,
    concept: SelectedConcept,
    evidence: ConceptEvidence,
    provider: RxNavProvider | None,
) -> tuple[ConceptStatus, list[ValidationFinding]]:
    findings: list[ValidationFinding] = []
    status = evidence.status

    if status is ConceptStatus.UNKNOWN and provider is not None:
        verdict = provider.validate_concept(concept.concept_id)
        if not verdict.exists:
            findings.append(
                _error(
                    "UNKNOWN_IDENTIFIER",
                    f"{concept.concept_id} is not known to {concept.vocabulary}",
                    concept.concept_id,
                )
            )
        else:
            status = verdict.status
            if verdict.remapped_to:
                findings.append(
                    ValidationFinding(
                        code="CONCEPT_REMAPPED",
                        severity=Severity.WARNING,
                        message=(
                            f"{concept.concept_id} was remapped to {', '.join(verdict.remapped_to)}"
                        ),
                        concept_id=concept.concept_id,
                        error_category=ErrorCategory.VALIDATION_ERROR,
                    )
                )
            if status is ConceptStatus.ACTIVE:
                findings.append(
                    ValidationFinding(
                        code="STATUS_VERIFIED_ACTIVE",
                        severity=Severity.INFO,
                        message=f"provider verified {concept.concept_id} as active",
                        concept_id=concept.concept_id,
                    )
                )

    if concept.status is not ConceptStatus.UNKNOWN:
        if status is ConceptStatus.UNKNOWN:
            findings.append(
                _error(
                    "UNSUPPORTED_STATUS_CLAIM",
                    f"{field_name} claims status {concept.status.value} for {concept.concept_id}, but canonical evidence did not establish lifecycle status",
                    concept.concept_id,
                )
            )
        elif concept.status is not status:
            findings.append(
                _error(
                    "CONCEPT_STATUS_MISMATCH",
                    f"{field_name} claims status {concept.status.value} for {concept.concept_id}; canonical evidence says {status.value}",
                    concept.concept_id,
                )
            )
    return status, findings


def _validate_source_role(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
) -> list[ValidationFinding]:
    source = proposal.source_level_concept
    if (
        source is None
        or package.candidate_by_qualified_id(profile.task.target_terminology, source.concept_id)
        is not None
    ):
        return []
    return [
        _error(
            "SOURCE_CONCEPT_NOT_DIRECTLY_RETRIEVED",
            f"source-level concept {source.concept_id} is not a directly retrieved candidate",
            source.concept_id,
        )
    ]


def _validate_target_roles(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
) -> list[ValidationFinding]:
    granularity = profile.task.desired_granularity
    if granularity is None:
        return []
    policy = _GRANULARITY_POLICY.get(granularity)
    if policy is None:
        return [
            _error(
                "UNSUPPORTED_GRANULARITY",
                f"no validation policy exists for requested granularity {granularity!r}",
            )
        ]

    predicate, wanted_classes = policy
    if not profile.knowledge.permits(predicate):
        return [
            _error(
                "GRANULARITY_RELATION_NOT_PERMITTED",
                f"profile does not permit {predicate}, required for {granularity}",
            )
        ]

    # An unresolved mapping selects the directly retrieved source only as a
    # review anchor; it does not claim that source as the analysis target.
    if (
        proposal.requested_human_review
        and proposal.analysis_level_concept is None
        and proposal.relationship_to_source is RelationshipType.RELATED_NOT_EQUIVALENT
    ):
        return []

    index = package.qualified_concept_evidence_index()
    target_vocabulary = profile.task.target_terminology.casefold()
    source_id = (
        proposal.source_level_concept.concept_id
        if proposal.source_level_concept is not None
        else None
    )
    findings: list[ValidationFinding] = []
    canonical_targets = _canonical_source_targets(
        package,
        profile,
        source_id=source_id,
        predicate=predicate,
    )

    if proposal.decision is DecisionType.MULTI_CONCEPT_MAPPED:
        selected_ids = {concept.concept_id for concept in proposal.selected_concepts}
        if selected_ids != canonical_targets:
            missing = sorted(canonical_targets - selected_ids)
            unexpected = sorted(selected_ids - canonical_targets)
            details: list[str] = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            findings.append(
                _error(
                    "MULTI_COMPONENT_SET_MISMATCH",
                    "multi-concept selections must exactly equal the canonical "
                    f"{predicate} targets of source {source_id or '[missing]'}"
                    + (f": {'; '.join(details)}" if details else ""),
                    source_id,
                    ErrorCategory.ADJUDICATION_ERROR,
                )
            )

    for concept in proposal.selected_concepts:
        requires_source_path = proposal.relationship_to_source in {
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            RelationshipType.MULTIPLE_COMPONENTS_REQUIRED,
        }
        if requires_source_path and concept.concept_id not in canonical_targets:
            findings.append(
                _error(
                    "TARGET_NOT_REACHABLE_FROM_SOURCE",
                    f"{concept.concept_id} is not a canonical {predicate} target of "
                    f"source {source_id or '[missing]'}",
                    concept.concept_id,
                    ErrorCategory.ADJUDICATION_ERROR,
                )
            )

        evidence = index.get((target_vocabulary, concept.concept_id))
        if evidence is None:
            continue

        if evidence.retrieved_directly and evidence.concept_class in wanted_classes:
            continue

        predicates = set(evidence.relationship_predicates)
        reachable = source_id is None or source_id in evidence.source_candidate_ids
        class_compatible = (
            evidence.concept_class is None or evidence.concept_class in wanted_classes
        )
        if predicate in predicates and reachable and class_compatible:
            continue

        findings.append(
            _error(
                "INVALID_GRANULARITY_TARGET",
                f"{concept.concept_id} is evidenced as class {evidence.concept_class or 'unknown'} via {sorted(predicates) or ['direct retrieval']}, not as a {granularity} target reachable through {predicate}",
                concept.concept_id,
                ErrorCategory.UNSUPPORTED_SPECIFICITY,
            )
        )
    return findings


def _canonical_source_targets(
    package: KnowledgePackage,
    profile: TaskProfile,
    *,
    source_id: str | None,
    predicate: str,
) -> set[str]:
    """Return official, profile-permitted targets attached to the chosen source.

    A target being independently retrieved is not evidence that it is broader
    than, or a component of, the selected source. The relationship itself must
    be present in the bounded terminology response for that source.
    """
    if source_id is None or not profile.knowledge.permits(predicate):
        return set()
    source = package.candidate_by_qualified_id(profile.task.target_terminology, source_id)
    if source is None:
        return set()
    vocabulary = profile.task.target_terminology.casefold()
    return {
        statement.object_id
        for statement in source.relationships
        if statement.subject_id == source_id
        and statement.predicate == predicate
        and statement.source_resource.casefold() == vocabulary
        and statement.authority is Authority.OFFICIAL_CODING_RELATIONSHIP
    }


def _validate_historical_source(
    proposal: AlignmentProposal,
    evidence_index: dict[tuple[str, str], ConceptEvidence],
    target_vocabulary: str,
) -> list[ValidationFinding]:
    source = proposal.source_level_concept
    if source is None:
        return []
    evidence = evidence_index.get((target_vocabulary.casefold(), source.concept_id))
    if evidence is None or evidence.status in {ConceptStatus.ACTIVE, ConceptStatus.UNKNOWN}:
        return []
    return [
        ValidationFinding(
            code="HISTORICAL_SOURCE_CONCEPT",
            severity=Severity.WARNING,
            message=f"mapping rests on {source.concept_id}, whose canonical status is {evidence.status.value}",
            concept_id=source.concept_id,
            error_category=ErrorCategory.RESOURCE_COVERAGE_GAP,
        )
    ]


def _error(
    code: str,
    message: str,
    concept_id: str | None = None,
    category: ErrorCategory = ErrorCategory.VALIDATION_ERROR,
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        severity=Severity.ERROR,
        message=message,
        concept_id=concept_id,
        error_category=category,
    )


def _deduplicate(findings: list[ValidationFinding]) -> list[ValidationFinding]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[ValidationFinding] = []
    for finding in findings:
        key = (finding.code, finding.concept_id, finding.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique
