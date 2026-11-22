"""Hierarchy-domain semantic validation check (`hierarchy.semantics`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from biosemalign.enums import ErrorCategory, RelationshipType, SemanticLossKind, Severity
from biosemalign.exceptions import ConfigurationError
from biosemalign.input.context import context_view
from biosemalign.modules.base import OpContext
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, SemanticLoss, ValidationFinding
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit, canonical_preserved_attributes

__all__ = ["HierarchySemanticsCheck", "hierarchy_semantics_factory"]


class HierarchySemanticsCheck:
    """Attribute preservation and spec-driven semantic-loss accounting for hierarchy domains."""

    op_id = "hierarchy.semantics"
    version = "0.1.0"
    uses_model = False
    options: ClassVar[dict[str, Any]] = {}

    def __init__(self, profile: TaskProfile) -> None:
        self.profile = profile

    def check(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if sau is not None:
            expected_attrs = canonical_preserved_attributes(sau.attributes)
            if proposal.preserved_attributes != expected_attrs:
                findings.append(
                    ValidationFinding(
                        code="PRESERVED_ATTRIBUTES_MISMATCH",
                        severity=Severity.ERROR,
                        message=(
                            "proposal preserved_attributes do not match the canonical contextual "
                            "attributes carried by the SAU"
                        ),
                        error_category=ErrorCategory.ADJUDICATION_ERROR,
                    )
                )

        loss_bearing = {
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            RelationshipType.MULTIPLE_COMPONENTS_REQUIRED,
        }
        reported_kinds = [loss.kind for loss in proposal.semantic_loss]
        reported = set(reported_kinds)
        if proposal.relationship_to_source not in loss_bearing:
            if reported:
                findings.append(
                    ValidationFinding(
                        code="UNEXPECTED_SEMANTIC_LOSS",
                        severity=Severity.ERROR,
                        message=(
                            f"relationship {proposal.relationship_to_source.value} cannot carry "
                            "semantic-loss claims"
                        ),
                        error_category=ErrorCategory.ADJUDICATION_ERROR,
                    )
                )
            return findings

        canonical = _expected_losses(proposal, package, self.profile, sau)
        expected = set(canonical)
        missing = expected - reported
        unexpected = reported - expected
        if missing:
            code = "UNREPORTED_SEMANTIC_LOSS" if not reported else "INCOMPLETE_SEMANTIC_LOSS"
            findings.append(
                ValidationFinding(
                    code=code,
                    severity=Severity.WARNING,
                    message=(
                        "the mapping omits semantic-loss categories established by the package "
                        f"evidence: {', '.join(sorted(k.value for k in missing))}"
                    ),
                    error_category=ErrorCategory.VALIDATION_ERROR,
                )
            )
        if unexpected:
            findings.append(
                ValidationFinding(
                    code="UNEXPECTED_SEMANTIC_LOSS",
                    severity=Severity.ERROR,
                    message=(
                        "semantic-loss categories are not supported by package evidence: "
                        + ", ".join(sorted(k.value for k in unexpected))
                    ),
                    error_category=ErrorCategory.ADJUDICATION_ERROR,
                )
            )
        for loss in proposal.semantic_loss:
            ref = canonical.get(loss.kind)
            if ref is not None and loss != ref:
                findings.append(
                    ValidationFinding(
                        code="SEMANTIC_LOSS_PAYLOAD_MISMATCH",
                        severity=Severity.ERROR,
                        message=f"{loss.kind.value} semantic loss payload does not match bounded evidence",
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
    source = proposal.source_level_concept
    candidate = (
        package.candidate_by_qualified_id(
            profile.target.source_vocabulary(source.vocabulary), source.concept_id
        )
        if source is not None
        else None
    )
    if candidate is None:
        return {
            SemanticLossKind.SPECIFICITY: SemanticLoss(
                kind=SemanticLossKind.SPECIFICITY,
                description="target is broader than the source term",
                source_value=None,
            )
        }
    spec = profile.target_spec()
    selected_ids = [c.concept_id for c in proposal.selected_concepts]
    first_target = next(iter(proposal.selected_concepts), None)
    target_ev = (
        package.evidence_for(profile.target.terminology, first_target.concept_id)
        if first_target is not None
        else None
    )
    expected = {
        loss.kind: loss
        for loss in spec.expected_losses(
            candidate,
            selected_ids=selected_ids,
            target_label=target_ev.label if target_ev is not None else None,
            mention_text=sau.mention.text if sau is not None else None,
            context=(
                context_view(sau, "resolve", profile)
                if sau is not None
                else dict(proposal.preserved_attributes)
            ),
        )
    }
    if not expected:
        expected[SemanticLossKind.SPECIFICITY] = SemanticLoss(
            kind=SemanticLossKind.SPECIFICITY,
            description="target is broader than the source term",
            source_value=candidate.preferred_label,
        )
    return expected


def hierarchy_semantics_factory(
    context: OpContext, options: Mapping[str, Any]
) -> HierarchySemanticsCheck:
    if options:
        raise ConfigurationError(
            f"operator 'hierarchy.semantics' takes no options, got {sorted(options)}"
        )
    return HierarchySemanticsCheck(context.profile)
