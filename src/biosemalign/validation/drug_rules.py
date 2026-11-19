"""Drug-specific semantic validation (§8.2).

Two failures here change exposure classification rather than merely annoying a
reviewer, so both are checked structurally rather than left to the adjudicator's
judgement:

* dropping a component of a combination product, which silently under-counts
  one drug and mis-attributes the other;
* conflating a salt form with its base ingredient in the *other* direction —
  claiming exactness where the source was more specific than the target.
"""

from __future__ import annotations

from biosemalign.enums import (
    ErrorCategory,
    RelationshipType,
    SemanticLossKind,
    Severity,
)
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, SemanticLoss, ValidationFinding
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["validate_drug_semantics"]


def validate_drug_semantics(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
    *,
    sau: SemanticAlignmentUnit | None = None,
) -> list[ValidationFinding]:
    """Apply drug-domain semantic rules to a proposal."""
    findings: list[ValidationFinding] = []

    findings.extend(_check_combination_integrity(proposal, package, profile))
    findings.extend(_check_salt_conflation(proposal, profile))
    findings.extend(_check_preserved_attributes(proposal, sau))
    findings.extend(_check_loss_reported(proposal, package, profile, sau))

    return findings


def _check_combination_integrity(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
) -> list[ValidationFinding]:
    """Every ingredient of a combination product must survive into the output."""
    if not profile.validation.preserve_combination_ingredients:
        return []

    source = proposal.source_level_concept
    if source is None:
        return []

    candidate = package.candidate_by_qualified_id(
        profile.task.source_vocabulary(source.vocabulary), source.concept_id
    )
    if candidate is None:
        return []

    spec = profile.target_spec()
    predicate = spec.predicate(profile.task.desired_granularity)
    level = spec.granularity(profile.task.desired_granularity)
    if predicate is None or level is None or level.multi != "components":
        return []
    ingredients = {s.object_id for s in candidate.relationships if s.predicate == predicate}
    if len(ingredients) <= 1:
        return []

    selected = {c.concept_id for c in proposal.selected_concepts}
    dropped = ingredients - selected
    if not dropped:
        return []

    labels = {
        s.object_id: (s.object_label or s.object_id)
        for s in candidate.relationships
        if s.predicate == predicate
    }
    return [
        ValidationFinding(
            code="COMBINATION_INGREDIENT_DROPPED",
            severity=Severity.ERROR,
            message=(
                f"{source.concept_id} is a combination of {len(ingredients)} ingredients but "
                f"the proposal selects {len(selected)}; missing: "
                + ", ".join(sorted(labels[i] for i in dropped))
            ),
            concept_id=source.concept_id,
            error_category=ErrorCategory.ADJUDICATION_ERROR,
        )
    ]


def _check_salt_conflation(
    proposal: AlignmentProposal, profile: TaskProfile
) -> list[ValidationFinding]:
    """An exact claim across a salt boundary is not exact."""
    if not profile.validation.flag_salt_active_moiety_conflation:
        return []
    if proposal.relationship_to_source is not RelationshipType.EXACT:
        return []

    source = proposal.source_level_concept
    analysis = proposal.analysis_level_concept
    if source is None or analysis is None:
        return []
    if source.concept_id == analysis.concept_id:
        return []

    # Precise ingredient on one side, base ingredient on the other, asserted as
    # equivalent: metoprolol succinate and metoprolol are not interchangeable
    # for formulation-sensitive analyses.
    spec = profile.target_spec()
    if spec.salt is None:
        return []
    pair = {
        spec.canonical_class(source.concept_class),
        spec.canonical_class(analysis.concept_class),
    }
    if pair & set(spec.salt.precise_classes) and pair & set(spec.salt.base_classes):
        return [
            ValidationFinding(
                code="SALT_ACTIVE_MOIETY_CONFLATION",
                severity=Severity.ERROR,
                message=(
                    f"{source.concept_id} ({source.concept_class}) and {analysis.concept_id} "
                    f"({analysis.concept_class}) differ by salt form and cannot be related as "
                    "EXACT; the correct relationship is TARGET_BROADER_THAN_SOURCE"
                ),
                concept_id=analysis.concept_id,
                error_category=ErrorCategory.RELATIONSHIP_CLASSIFICATION_ERROR,
            )
        ]
    return []


def _check_preserved_attributes(
    proposal: AlignmentProposal, sau: SemanticAlignmentUnit | None
) -> list[ValidationFinding]:
    if sau is None:
        return []
    expected = sau.attributes.model_dump(exclude_none=True, exclude={"extensions"})
    if proposal.preserved_attributes == expected:
        return []
    return [
        ValidationFinding(
            code="PRESERVED_ATTRIBUTES_MISMATCH",
            severity=Severity.ERROR,
            message=(
                "proposal preserved_attributes do not match the canonical contextual "
                "attributes carried by the SAU"
            ),
            error_category=ErrorCategory.ADJUDICATION_ERROR,
        )
    ]


def _check_loss_reported(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
    sau: SemanticAlignmentUnit | None,
) -> list[ValidationFinding]:
    """Require exactly the losses derivable from bounded terminology evidence."""
    reported_kinds = [loss.kind for loss in proposal.semantic_loss]
    reported = set(reported_kinds)
    loss_bearing_relations = {
        RelationshipType.TARGET_BROADER_THAN_SOURCE,
        RelationshipType.MULTIPLE_COMPONENTS_REQUIRED,
    }
    if proposal.relationship_to_source not in loss_bearing_relations:
        if not reported:
            return []
        return [
            ValidationFinding(
                code="UNEXPECTED_SEMANTIC_LOSS",
                severity=Severity.ERROR,
                message=(
                    f"relationship {proposal.relationship_to_source.value} cannot carry "
                    "semantic-loss claims"
                ),
                error_category=ErrorCategory.ADJUDICATION_ERROR,
            )
        ]

    canonical_losses = _expected_losses(proposal, package, profile, sau)
    expected = set(canonical_losses)
    missing = expected - reported
    unexpected = reported - expected
    duplicates = {kind for kind in reported if reported_kinds.count(kind) > 1}
    findings: list[ValidationFinding] = []

    if missing:
        names = ", ".join(sorted(kind.value for kind in missing))
        code = "UNREPORTED_SEMANTIC_LOSS" if not reported else "INCOMPLETE_SEMANTIC_LOSS"
        findings.append(
            ValidationFinding(
                code=code,
                severity=Severity.WARNING,
                message=(
                    "the mapping omits semantic-loss categories established by the package "
                    f"evidence: {names}"
                ),
                error_category=ErrorCategory.VALIDATION_ERROR,
            )
        )

    if unexpected:
        names = ", ".join(sorted(kind.value for kind in unexpected))
        findings.append(
            ValidationFinding(
                code="UNEXPECTED_SEMANTIC_LOSS",
                severity=Severity.ERROR,
                message=f"semantic-loss categories are not supported by package evidence: {names}",
                error_category=ErrorCategory.ADJUDICATION_ERROR,
            )
        )

    if duplicates:
        names = ", ".join(sorted(kind.value for kind in duplicates))
        findings.append(
            ValidationFinding(
                code="DUPLICATE_SEMANTIC_LOSS",
                severity=Severity.ERROR,
                message=f"semantic-loss categories must be unique: {names}",
                error_category=ErrorCategory.ADJUDICATION_ERROR,
            )
        )

    for loss in proposal.semantic_loss:
        canonical = canonical_losses.get(loss.kind)
        if canonical is None or loss == canonical:
            continue
        mismatched_fields: list[str] = []
        if loss.description != canonical.description:
            mismatched_fields.append("description")
        if loss.source_value != canonical.source_value:
            mismatched_fields.append("source_value")
        findings.append(
            ValidationFinding(
                code="SEMANTIC_LOSS_PAYLOAD_MISMATCH",
                severity=Severity.ERROR,
                message=(
                    f"{loss.kind.value} semantic loss has noncanonical "
                    f"{', '.join(mismatched_fields)}; both fields must come from bounded evidence"
                ),
                error_category=ErrorCategory.ADJUDICATION_ERROR,
            )
        )
    return findings


def _expected_losses(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
    sau: SemanticAlignmentUnit | None,
) -> dict[SemanticLossKind, SemanticLoss]:
    """Derive canonical loss payloads without trusting proposal prose or values.

    The losses come from the target spec's rules — the same function the
    deterministic adjudicator reports them with — applied to the bounded
    evidence rather than to the proposal's claims.
    """
    source = proposal.source_level_concept
    candidate = (
        package.candidate_by_qualified_id(
            profile.task.source_vocabulary(source.vocabulary), source.concept_id
        )
        if source is not None
        else None
    )
    if candidate is None:
        return {SemanticLossKind.SPECIFICITY: _specificity(None)}

    spec = profile.target_spec()
    selected_ids = [concept.concept_id for concept in proposal.selected_concepts]
    first_target = next(iter(proposal.selected_concepts), None)
    target_evidence = (
        package.evidence_for(profile.task.target_terminology, first_target.concept_id)
        if first_target is not None
        else None
    )
    route = (
        (sau.context.route or sau.attributes.route)
        if sau is not None
        else proposal.preserved_attributes.get("route")
    )
    expected = {
        loss.kind: loss
        for loss in spec.expected_losses(
            candidate,
            selected_ids=selected_ids,
            target_label=target_evidence.label if target_evidence is not None else None,
            mention_text=sau.mention.text if sau is not None else None,
            context={"route": route},
        )
    }

    granularity = profile.task.desired_granularity
    predicate = spec.predicate(granularity)
    components = {s.object_id for s in candidate.relationships if s.predicate == predicate}
    combination = (
        spec.combination_loss(
            granularity, count=len(components), source_label=candidate.preferred_label
        )
        if len(components) > 1
        else None
    )
    if combination is not None:
        expected[combination.kind] = combination

    # A broader mapping necessarily loses specificity even when the package
    # does not expose a more specific typed attribute.
    if not expected:
        expected[SemanticLossKind.SPECIFICITY] = _specificity(candidate.preferred_label)
    return expected


def _specificity(source_value: str | None) -> SemanticLoss:
    return SemanticLoss(
        kind=SemanticLossKind.SPECIFICITY,
        description=(
            "target concept is broader than the source; bounded evidence does not "
            "identify a typed source attribute"
        ),
        source_value=source_value,
    )
