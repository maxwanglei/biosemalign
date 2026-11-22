"""SAU readiness assessment (§5.8).

Answers one question: does this unit carry enough information to be mapped
correctly? Detecting missing context *before* retrieval is what lets the
pipeline abstain honestly instead of guessing from a bare string.
"""

from __future__ import annotations

from biosemalign.enums import ReadinessStatus
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.sau import ContextEnvelope, ContextualAttributes, ReadinessAssessment

__all__ = ["assess_readiness"]


def assess_readiness(
    *,
    mention_text: str,
    envelope: ContextEnvelope,
    attributes: ContextualAttributes,
    profile: TaskProfile,
) -> ReadinessAssessment:
    """Classify how mappable this unit is, and record what is missing."""
    if not mention_text.strip():
        return ReadinessAssessment(
            status=ReadinessStatus.UNSUPPORTED_INPUT,
            warnings=["source value is empty"],
        )

    present = _present_fields(envelope, attributes)

    effective_fields = profile.effective_context_fields()
    missing_required = [f.name for f in effective_fields if f.required and f.name not in present]
    optional_source = profile.context.fields or effective_fields
    missing_optional = [
        f.name
        for f in optional_source
        if not f.required and f.use != ["key"] and f.name not in present
    ]

    warnings: list[str] = []
    if missing_optional:
        warnings.append("context not supplied: " + ", ".join(sorted(missing_optional)))

    if missing_required:
        return ReadinessAssessment(
            status=ReadinessStatus.INSUFFICIENT_CONTEXT,
            missing_context=sorted(missing_required),
            warnings=warnings,
        )

    if missing_optional:
        return ReadinessAssessment(
            status=ReadinessStatus.READY_WITH_LIMITED_CONTEXT,
            missing_context=sorted(missing_optional),
            warnings=warnings,
        )

    return ReadinessAssessment(status=ReadinessStatus.READY_FOR_MAPPING, warnings=warnings)


def _present_fields(envelope: ContextEnvelope, attributes: ContextualAttributes) -> set[str]:
    """Field names carrying a usable value across declared slots and extensions."""
    present: set[str] = set()
    for model in (envelope, attributes):
        for name, value in model.model_dump(exclude={"extensions", "neighboring_fields"}).items():
            if value is not None and value != "" and value != {}:
                present.add(name)
        for key, value in model.extensions.items():
            if value is not None and value != "" and value != {}:
                present.add(key)
    return present
