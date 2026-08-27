"""Module 4: deterministic validation and integrity control.

The validator, not the adjudicator, holds final authority (§2). A second LLM
critic is explicitly not used in the first implementation — deterministic rules
are auditable, reproducible, and cannot themselves hallucinate.
"""

from __future__ import annotations

from biosemalign.enums import Severity, ValidationStatus
from biosemalign.knowledge.providers.rxnorm import RxNavProvider
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding, ValidationResult
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.validation.drug_rules import validate_drug_semantics
from biosemalign.validation.terminology import validate_terminology

__all__ = ["VALIDATION_RULES_VERSION", "Validator"]

VALIDATION_RULES_VERSION = "0.1.0"

log = get_logger(__name__)


class Validator:
    """Runs terminology and semantic rules and returns a single verdict."""

    version = VALIDATION_RULES_VERSION

    def __init__(self, profile: TaskProfile, *, provider: RxNavProvider | None = None) -> None:
        self.profile = profile
        self.provider = provider

    def validate(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None = None,
    ) -> ValidationResult:
        """Produce the verdict for one proposal (§8.3)."""
        findings: list[ValidationFinding] = []
        findings.extend(
            validate_terminology(proposal, package, self.profile, provider=self.provider)
        )
        findings.extend(validate_drug_semantics(proposal, package, self.profile, sau=sau))

        status = self._verdict(proposal, findings)

        log.debug(
            "validation.completed",
            sau_id=proposal.sau_id,
            status=status.value,
            errors=sum(1 for f in findings if f.severity is Severity.ERROR),
            warnings=sum(1 for f in findings if f.severity is Severity.WARNING),
        )
        return ValidationResult(
            status=status, findings=findings, rules_version=VALIDATION_RULES_VERSION
        )

    def _verdict(
        self, proposal: AlignmentProposal, findings: list[ValidationFinding]
    ) -> ValidationStatus:
        """Collapse findings into one status.

        Order matters: a single error rejects outright, and an explicit request
        for review outranks a clean rule sweep — the adjudicator saw something
        the rules do not encode.
        """
        if any(f.severity is Severity.ERROR for f in findings):
            return ValidationStatus.REJECTED

        if proposal.requested_human_review:
            return ValidationStatus.HUMAN_REVIEW_REQUIRED

        if any(f.severity is Severity.WARNING for f in findings):
            return ValidationStatus.VALID_WITH_WARNING

        if proposal.semantic_loss:
            return ValidationStatus.VALID_WITH_SEMANTIC_LOSS

        return ValidationStatus.VALID
