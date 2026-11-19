"""Module 4: deterministic validation and integrity control.

The validator, not the adjudicator, holds final authority (§2). A second LLM
critic is explicitly not used in the first implementation — deterministic rules
are auditable, reproducible, and cannot themselves hallucinate.

The terminology checks (unretrieved identifiers, concept status, relationship
policy) are mandatory and always run first. Domain rules are validation
operators configured by the profile and run after them; they can add findings
but cannot remove the mandatory ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from biosemalign.enums import Severity, ValidationStatus
from biosemalign.knowledge.providers.base import KnowledgeProvider
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding, ValidationResult
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.validation.terminology import validate_terminology

if TYPE_CHECKING:
    from biosemalign.ops.base import ValidateOp

__all__ = ["VALIDATION_RULES_VERSION", "Validator"]

#: Version of the mandatory terminology checks. The full ``rules_version``
#: recorded on each result also names every configured validation operator.
VALIDATION_RULES_VERSION = "0.1.0"

log = get_logger(__name__)


class Validator:
    """Runs terminology and semantic rules and returns a single verdict."""

    def __init__(
        self,
        profile: TaskProfile,
        *,
        provider: KnowledgeProvider | None = None,
        rules: Sequence[ValidateOp] | None = None,
    ) -> None:
        self.profile = profile
        self.provider = provider
        if rules is None:
            # Imported here so the operator registry, which imports this
            # package's rule modules, is never needed to import the validator.
            from biosemalign.ops import OpContext, build_op

            context = OpContext(profile=profile)
            rules = [
                build_op("validate", rule.op, context, rule.options)
                for rule in profile.validation.rules
            ]
        self.rules: tuple[ValidateOp, ...] = tuple(rules)
        self.version = "+".join(
            [VALIDATION_RULES_VERSION, *(f"{rule.op_id}@{rule.version}" for rule in self.rules)]
        )

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
        for rule in self.rules:
            findings.extend(rule.validate(proposal, package, sau))

        status = self._verdict(proposal, findings)

        log.debug(
            "validation.completed",
            sau_id=proposal.sau_id,
            status=status.value,
            errors=sum(1 for f in findings if f.severity is Severity.ERROR),
            warnings=sum(1 for f in findings if f.severity is Severity.WARNING),
        )
        return ValidationResult(status=status, findings=findings, rules_version=self.version)

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
