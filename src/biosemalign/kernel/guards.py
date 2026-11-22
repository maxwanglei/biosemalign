"""Kernel isolation and non-destruction guards (§5.5: G1, G3)."""

from __future__ import annotations

from typing import Any, Literal

from biosemalign.exceptions import WorkflowViolation
from biosemalign.input.context import normalize_context_key
from biosemalign.modules.base import CandidateSet
from biosemalign.profiles.schema import ContextUse, TaskProfile
from biosemalign.schemas.expansion import ExpansionMember
from biosemalign.schemas.sau import ContextEnvelope, SemanticAlignmentUnit

__all__ = [
    "isolated_view",
    "mark_candidate_set_from_model",
    "mark_expansion_member_from_model",
    "merge_model_cleaned_sau",
    "require_non_destructive",
    "resolver_view",
    "withheld_fields",
]

_CONTEXT_DATA_FIELDS: frozenset[str] = frozenset(ContextEnvelope.model_fields) - {
    "extensions",
    "neighboring_fields",
}


def _blank(field_name: str) -> object:
    return ContextEnvelope.model_fields[field_name].get_default(call_default_factory=True)


def withheld_fields(sau: SemanticAlignmentUnit, profile: TaskProfile) -> tuple[str, ...]:
    """Names of declared blinded context fields carrying values in ``sau``."""
    withheld: list[str] = []
    for name in profile.effective_blinded_fields():
        if name in _CONTEXT_DATA_FIELDS:
            if getattr(sau.context, name) != _blank(name):
                withheld.append(name)
        elif sau.context.extensions.get(name) not in (None, ""):
            withheld.append(name)
    return tuple(withheld)


def isolated_view(
    sau: SemanticAlignmentUnit,
    purpose: ContextUse | Literal["link"],
    profile: TaskProfile,
) -> SemanticAlignmentUnit:
    """An isolated copy of ``sau`` carrying only context permitted for ``purpose`` (G3)."""
    view = sau.model_copy(deep=True)
    permitted_fields = [
        field
        for field in profile.effective_context_fields()
        if purpose in field.use and not field.blinded
    ]
    allowed = {field.name for field in permitted_fields}
    allowed_source_keys = {
        normalize_context_key(key_name)
        for field in permitted_fields
        for key_name in (field.name, *field.aliases)
    }
    updates: dict[str, Any] = {
        name: _blank(name) for name in _CONTEXT_DATA_FIELDS if name not in allowed
    }
    updates["neighboring_fields"] = {}
    updates["extensions"] = {
        key: value for key, value in view.context.extensions.items() if key in allowed
    }
    scrubbed_source = view.source.model_copy(
        update={
            "row": {
                key: value
                for key, value in view.source.row.items()
                if normalize_context_key(key) in allowed_source_keys
            },
            "extensions": {
                key: value
                for key, value in view.source.extensions.items()
                if normalize_context_key(key) in allowed_source_keys
            },
        }
    )
    return view.model_copy(
        update={
            "source": scrubbed_source,
            "context": view.context.model_copy(update=updates),
        }
    )


def resolver_view(sau: SemanticAlignmentUnit, profile: TaskProfile) -> SemanticAlignmentUnit:
    """An isolated copy of ``sau`` holding only resolver-permitted context (G3)."""
    return isolated_view(sau, "resolve", profile)


def require_non_destructive(
    before: SemanticAlignmentUnit, after: SemanticAlignmentUnit, op_id: str
) -> None:
    """Allow a cleaner to set the query text and fill empty attributes only."""
    if not isinstance(after, SemanticAlignmentUnit):
        raise WorkflowViolation(f"cleaner {op_id!r} returned {type(after).__name__}, not an SAU")
    if before.mention.retrieval_text_from_model and not after.mention.retrieval_text_from_model:
        raise WorkflowViolation(
            f"cleaner {op_id!r} cleared the model-suggestion mark on mention.retrieval_text"
        )
    exclude: dict[str, Any] = {
        "mention": {"retrieval_text", "retrieval_text_from_model"},
        "attributes": True,
    }
    if before.model_dump(exclude=exclude) != after.model_dump(exclude=exclude):
        raise WorkflowViolation(
            f"cleaner {op_id!r} changed the source mention, identity, or context; cleaners "
            "may only set mention.retrieval_text and fill empty attributes"
        )
    previous = before.attributes.model_dump(exclude_none=True, exclude={"extensions"})
    current = after.attributes.model_dump(exclude_none=True, exclude={"extensions"})
    overwritten = sorted(name for name, value in previous.items() if current.get(name) != value)
    lost_extensions = sorted(
        key
        for key, value in before.attributes.extensions.items()
        if after.attributes.extensions.get(key) != value
    )
    if overwritten or lost_extensions:
        raise WorkflowViolation(
            f"cleaner {op_id!r} overwrote or removed asserted attributes: "
            f"{overwritten + lost_extensions}"
        )


def merge_model_cleaned_sau(
    original: SemanticAlignmentUnit, cleaned: SemanticAlignmentUnit
) -> SemanticAlignmentUnit:
    """Merge allowed clean outputs onto ``original``, keeping model marks and context intact."""
    mention_updates: dict[str, Any] = {}
    new_text = cleaned.mention.retrieval_text
    if new_text is not None and new_text != original.mention.retrieval_text:
        mention_updates["retrieval_text"] = new_text
        if new_text != original.mention.text:
            mention_updates["retrieval_text_from_model"] = True
    if cleaned.mention.retrieval_text_from_model:
        mention_updates["retrieval_text_from_model"] = True

    attr_updates: dict[str, Any] = {}
    for field_name in type(original.attributes).model_fields:
        if field_name == "extensions":
            continue
        if getattr(original.attributes, field_name) is None:
            candidate_val = getattr(cleaned.attributes, field_name)
            if candidate_val is not None:
                attr_updates[field_name] = candidate_val
    merged_ext = dict(original.attributes.extensions)
    ext_changed = False
    for key, value in cleaned.attributes.extensions.items():
        if key not in merged_ext and value is not None:
            merged_ext[key] = value
            ext_changed = True
    if ext_changed:
        attr_updates["extensions"] = merged_ext

    sau_updates: dict[str, Any] = {}
    if mention_updates:
        sau_updates["mention"] = original.mention.model_copy(update=mention_updates)
    if attr_updates:
        sau_updates["attributes"] = original.attributes.model_copy(update=attr_updates)
    return original.model_copy(update=sau_updates) if sau_updates else original


def mark_candidate_set_from_model(candidate_set: CandidateSet) -> CandidateSet:
    """Mark every retrieval evidence entry in ``candidate_set`` as model-suggested (M4, G10)."""
    marked = [
        candidate.model_copy(
            update={
                "retrieval_evidence": [
                    ev.model_copy(update={"from_model_suggestion": True})
                    for ev in candidate.retrieval_evidence
                ]
            }
        )
        for candidate in candidate_set.candidates
    ]
    return CandidateSet(
        matcher=candidate_set.matcher,
        candidates=marked,
        origin=candidate_set.origin,
    )


def mark_expansion_member_from_model(member: ExpansionMember) -> ExpansionMember:
    """Mark an expanded member and its retrieval evidence as model-suggested (M4, G10)."""
    marked_concept = member.concept.model_copy(
        update={
            "retrieval_evidence": [
                ev.model_copy(update={"from_model_suggestion": True})
                for ev in member.concept.retrieval_evidence
            ]
        }
    )
    return member.model_copy(update={"concept": marked_concept, "from_model_suggestion": True})
