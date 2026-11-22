"""Drug-domain semantic validation and check operator (`drug.semantics`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from biosemalign.enums import (
    Authority,
    ErrorCategory,
    RelationshipType,
    SemanticLossKind,
    Severity,
)
from biosemalign.input.context import context_view
from biosemalign.modules.base import OpContext
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentProposal, SemanticLoss, ValidationFinding
from biosemalign.schemas.knowledge import CodingPolicy, KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit, canonical_preserved_attributes

__all__ = [
    "DrugSemanticsCheck",
    "drug_coding_policies",
    "drug_semantics_factory",
    "validate_drug_semantics",
]


def validate_drug_semantics(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
    *,
    sau: SemanticAlignmentUnit | None = None,
    preserve_combination_ingredients: bool = True,
    flag_salt_active_moiety_conflation: bool = True,
) -> list[ValidationFinding]:
    """Apply drug-domain semantic rules to a proposal."""
    findings: list[ValidationFinding] = []

    findings.extend(
        _check_combination_integrity(
            proposal, package, profile, enabled=preserve_combination_ingredients
        )
    )
    findings.extend(
        _check_salt_conflation(proposal, profile, enabled=flag_salt_active_moiety_conflation)
    )
    findings.extend(_check_preserved_attributes(proposal, sau))
    findings.extend(_check_loss_reported(proposal, package, profile, sau))

    return findings


def _check_combination_integrity(
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    profile: TaskProfile,
    *,
    enabled: bool = True,
) -> list[ValidationFinding]:
    """Every ingredient of a combination product must survive into the output."""
    if not enabled:
        return []

    source = proposal.source_level_concept
    if source is None:
        return []

    candidate = package.candidate_by_qualified_id(
        profile.target.source_vocabulary(source.vocabulary), source.concept_id
    )
    if candidate is None:
        return []

    spec = profile.target_spec()
    predicate = spec.predicate(profile.target.level)
    level = spec.granularity(profile.target.level)
    if predicate is None or level is None or level.multi != "components":
        return []
    ingredients = {
        s.object_id
        for s in candidate.relationships
        if s.predicate == predicate and s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
    }
    if len(ingredients) <= 1:
        return []

    selected = {c.concept_id for c in proposal.selected_concepts}
    dropped = ingredients - selected
    if not dropped:
        return []

    labels = {
        s.object_id: (s.object_label or s.object_id)
        for s in candidate.relationships
        if s.predicate == predicate and s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
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
    proposal: AlignmentProposal,
    profile: TaskProfile,
    *,
    enabled: bool = True,
) -> list[ValidationFinding]:
    """An exact claim across a salt boundary is not exact."""
    if not enabled:
        return []
    if proposal.relationship_to_source is not RelationshipType.EXACT:
        return []

    source = proposal.source_level_concept
    analysis = proposal.analysis_level_concept
    if source is None or analysis is None:
        return []
    if source.concept_id == analysis.concept_id:
        return []

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
    expected = canonical_preserved_attributes(sau.attributes)
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
    """Derive canonical loss payloads without trusting proposal prose or values."""
    source = proposal.source_level_concept
    candidate = (
        package.candidate_by_qualified_id(
            profile.target.source_vocabulary(source.vocabulary), source.concept_id
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
        package.evidence_for(profile.target.terminology, first_target.concept_id)
        if first_target is not None
        else None
    )
    expected = {
        loss.kind: loss
        for loss in spec.expected_losses(
            candidate,
            selected_ids=selected_ids,
            target_label=target_evidence.label if target_evidence is not None else None,
            mention_text=sau.mention.text if sau is not None else None,
            context=(
                context_view(sau, "resolve", profile)
                if sau is not None
                else dict(proposal.preserved_attributes)
            ),
        )
    }

    granularity = profile.target.level
    predicate = spec.predicate(granularity)
    components = {
        s.object_id
        for s in candidate.relationships
        if s.predicate == predicate and s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
    }
    combination = (
        spec.combination_loss(
            granularity, count=len(components), source_label=candidate.preferred_label
        )
        if len(components) > 1
        else None
    )
    if combination is not None:
        expected[combination.kind] = combination

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


class _DrugSemanticsOptions(BioSemAlignModel):
    preserve_combination_ingredients: bool = True
    flag_salt_active_moiety_conflation: bool = True


class DrugSemanticsCheck:
    """Combination integrity, salt conflation, and semantic-loss accounting."""

    op_id = "drug.semantics"
    version = "0.1.0"
    uses_model = False

    def __init__(
        self,
        profile: TaskProfile,
        *,
        preserve_combination_ingredients: bool = True,
        flag_salt_active_moiety_conflation: bool = True,
    ) -> None:
        self.profile = profile
        self.preserve_combination_ingredients = preserve_combination_ingredients
        self.flag_salt_active_moiety_conflation = flag_salt_active_moiety_conflation

    @property
    def options(self) -> dict[str, Any]:
        return {
            "preserve_combination_ingredients": self.preserve_combination_ingredients,
            "flag_salt_active_moiety_conflation": self.flag_salt_active_moiety_conflation,
        }

    def check(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]:
        return validate_drug_semantics(
            proposal,
            package,
            self.profile,
            sau=sau,
            preserve_combination_ingredients=self.preserve_combination_ingredients,
            flag_salt_active_moiety_conflation=self.flag_salt_active_moiety_conflation,
        )


def drug_semantics_factory(context: OpContext, options: Mapping[str, Any]) -> DrugSemanticsCheck:
    parsed = _DrugSemanticsOptions.model_validate(options)
    return DrugSemanticsCheck(
        context.profile,
        preserve_combination_ingredients=parsed.preserve_combination_ingredients,
        flag_salt_active_moiety_conflation=parsed.flag_salt_active_moiety_conflation,
    )


def drug_coding_policies(options: Mapping[str, Any], profile: TaskProfile) -> list[CodingPolicy]:
    """Coding policies derived from ``drug.semantics`` options."""
    policies: list[CodingPolicy] = []
    op = profile.task.operation
    if bool(options.get("preserve_combination_ingredients", True)):
        policies.append(
            CodingPolicy(
                policy_id="combination_integrity",
                description="Combination products keep every component",
                rule=(
                    "A combination product must map to every one of its ingredients. "
                    "Silently dropping a component misclassifies exposure."
                ),
                applies_to=[op],
            )
        )
    if bool(options.get("flag_salt_active_moiety_conflation", True)):
        policies.append(
            CodingPolicy(
                policy_id="salt_active_moiety",
                description="Salt forms are not interchangeable",
                rule=(
                    "Do not treat one salt form as equivalent to another. Metoprolol "
                    "succinate and metoprolol tartrate share an ingredient but are "
                    "distinct precise ingredients with different formulations."
                ),
                applies_to=[op],
            )
        )
    return policies
