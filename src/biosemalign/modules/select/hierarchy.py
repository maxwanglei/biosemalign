"""Hierarchy-driven concept-set selector (`hierarchy`, §5.4, WP6)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from biosemalign.enums import ConceptStatus, RelationshipType
from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import OpContext
from biosemalign.modules.registry import register
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet
from biosemalign.schemas.phenotyping import ConceptSetMember, ConceptSetProposal, ConceptSetScope
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["HierarchySelector", "hierarchy_selector_factory"]


class HierarchySelector:
    """Classify anchor and expanded concepts using recorded hierarchy paths."""

    op_id = "hierarchy"
    version = "0.1.0"
    uses_model = False
    requires_network = False
    options: ClassVar[dict[str, Any]] = {}

    def __init__(self, knowledge: KnowledgeRouter | None, profile: TaskProfile) -> None:
        self.knowledge = knowledge
        self.profile = profile

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(backend="deterministic", model_name=f"hierarchy@{self.version}")

    def select(self, sau: SemanticAlignmentUnit, expansion: ExpansionSet) -> ConceptSetProposal:
        members: list[ConceptSetMember] = []
        seen: set[tuple[str, str]] = set()

        for anchor in expansion.anchors:
            key = (anchor.vocabulary.casefold(), anchor.concept_code)
            if key in seen:
                continue
            seen.add(key)
            label = anchor.label or anchor.concept_code
            concept_class: str | None = None
            status = ConceptStatus.UNKNOWN
            if self.knowledge is not None:
                concept = self.knowledge.provider_for(anchor.vocabulary).lookup(anchor.concept_code)
                if concept is not None:
                    label = concept.preferred_label
                    concept_class = concept.concept_class
                    status = concept.status
            members.append(
                ConceptSetMember(
                    vocabulary=anchor.vocabulary,
                    concept_code=anchor.concept_code,
                    label=label,
                    concept_class=concept_class,
                    status=status,
                    included=True,
                    relation=RelationshipType.EXACT,
                    scope="narrow",
                    reason="Direct match for the target phenotype idea.",
                    anchor=anchor,
                    hops=0,
                    path=[],
                    via_operator=expansion.expander,
                )
            )

        for exp in expansion.members:
            key = (exp.concept.vocabulary.casefold(), exp.concept.candidate_id)
            if key in seen:
                continue
            seen.add(key)
            relation, scope, reason = _classify_member(exp)
            members.append(
                ConceptSetMember(
                    vocabulary=exp.concept.vocabulary,
                    concept_code=exp.concept.candidate_id,
                    label=exp.concept.preferred_label,
                    concept_class=exp.concept.concept_class,
                    status=exp.concept.status,
                    included=True,
                    relation=relation,
                    scope=scope,
                    reason=reason,
                    anchor=exp.anchor,
                    hops=exp.hops,
                    path=list(exp.path),
                    via_operator=exp.via_operator,
                    from_model_suggestion=exp.from_model_suggestion,
                )
            )

        return ConceptSetProposal(
            sau_id=sau.sau_id,
            idea=sau.mention.text,
            target_terminology=sau.task.target_terminology,
            level=sau.task.desired_granularity,
            members=members,
            rationale=(
                f"Proposed {len(members)} member(s) from terminology expansion ({expansion.expander}); "
                "requires human approval before export."
            ),
            requested_human_review=True,
        )


def _classify_member(
    member: ExpansionMember,
) -> tuple[RelationshipType, ConceptSetScope, str]:
    directions = {step.direction for step in member.path}
    preds = " -> ".join(step.predicate for step in member.path)
    if directions == {"incoming"}:
        return (
            RelationshipType.TARGET_NARROWER_THAN_SOURCE,
            "narrow",
            f"Subtype/descendant reached from anchor via {preds} ({member.hops} hop(s)).",
        )
    if directions == {"outgoing"}:
        return (
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            "broad",
            f"Broader ancestor reached from anchor via {preds} ({member.hops} hop(s)).",
        )
    return (
        RelationshipType.RELATED_NOT_EQUIVALENT,
        "broad",
        f"Related candidate reached via {member.via_operator} ({preds or 'proxy'}).",
    )


def hierarchy_selector_factory(context: OpContext, options: Mapping[str, Any]) -> HierarchySelector:
    if options:
        raise ConfigurationError(
            f"operator 'hierarchy' selector takes no options, got {sorted(options)}"
        )
    return HierarchySelector(context.knowledge, context.profile)


register("select", "hierarchy", hierarchy_selector_factory)
