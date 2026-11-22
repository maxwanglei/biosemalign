"""Deterministic validation and integrity control (§5.5, §8).

The validator, not the resolver, holds final authority. The terminology checks
(unretrieved identifiers, concept status, relationship policy) are mandatory
and always run first. Domain checks are configured by the profile and run after
them; they can add findings but cannot remove the mandatory ones.
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
    from biosemalign.modules.base import Check
    from biosemalign.modules.model import ModelService

__all__ = ["VALIDATION_RULES_VERSION", "Validator"]

#: Version of the mandatory terminology checks. The full ``rules_version``
#: recorded on each result also names every configured check operator.
VALIDATION_RULES_VERSION = "0.1.0"

log = get_logger(__name__)


class Validator:
    """Runs terminology and check operators and returns a single verdict."""

    def __init__(
        self,
        profile: TaskProfile,
        *,
        provider: KnowledgeProvider | None = None,
        checks: Sequence[Check] | None = None,
        model_service: ModelService | None = None,
    ) -> None:
        self.profile = profile
        self.provider = provider
        self.model_service = model_service
        if checks is None:
            # Imported here so the operator registry, which imports this
            # package's check modules, is never needed to import the validator.
            from biosemalign.modules import OpContext, build

            context = OpContext(profile=profile, model_service=model_service)
            checks = [build("check", check.op, context, check.options) for check in profile.checks]
        self.checks: tuple[Check, ...] = tuple(checks)
        self.version = "+".join(
            [VALIDATION_RULES_VERSION, *(f"{check.op_id}@{check.version}" for check in self.checks)]
        )

    def validate(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None = None,
    ) -> ValidationResult:
        """Produce the verdict for one proposal (§8.3)."""
        from biosemalign.exceptions import AdjudicationFailure
        from biosemalign.kernel.guards import isolated_view

        findings: list[ValidationFinding] = []
        findings.extend(
            validate_terminology(proposal, package, self.profile, provider=self.provider, sau=sau)
        )
        for check in self.checks:
            check_sau = (
                isolated_view(sau, "check", self.profile)
                if sau is not None and check.uses_model
                else (sau.model_copy(deep=True) if sau is not None else None)
            )
            try:
                extra = check.check(
                    proposal.model_copy(deep=True),
                    package.model_copy(deep=True),
                    check_sau,
                )
            except AdjudicationFailure:
                if check.uses_model:
                    findings.append(
                        ValidationFinding(
                            code="MODEL_CHECK_UNAVAILABLE",
                            severity=Severity.WARNING,
                            message=(
                                f"model-backed check {check.op_id!r} returned an unusable reply; "
                                "sending decision to human review"
                            ),
                        )
                    )
                    continue
                raise
            findings.extend(extra)

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
        for review outranks a clean check sweep — the resolver saw something
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
