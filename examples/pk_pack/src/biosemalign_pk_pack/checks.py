"""Domain check operator for PK parameters (`pk.specimen_check`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from biosemalign.enums import DecisionType, Severity
from biosemalign.exceptions import ConfigurationError
from biosemalign.input.context import context_view
from biosemalign.modules.base import OpContext
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["PKSpecimenCheck", "specimen_check_factory"]


class PKSpecimenCheck:
    """Warn when a mapped PK parameter has no specimen attribute recorded."""

    op_id = "pk.specimen_check"
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
        del package
        if sau is None or proposal.decision not in {
            DecisionType.MAPPED,
            DecisionType.MULTI_CONCEPT_MAPPED,
        }:
            return []
        declared = context_view(sau, "check", self.profile)
        specimen = declared.get("specimen") or sau.attributes.specimen
        if specimen:
            return []
        return [
            ValidationFinding(
                code="MISSING_PK_SPECIMEN",
                severity=Severity.WARNING,
                message="Mapped PK parameter has no biological specimen; reviewer must confirm.",
            )
        ]


def specimen_check_factory(context: OpContext, options: Mapping[str, Any]) -> PKSpecimenCheck:
    if options:
        raise ConfigurationError(
            f"operator 'pk.specimen_check' takes no options, got {sorted(options)}"
        )
    return PKSpecimenCheck(context.profile)
