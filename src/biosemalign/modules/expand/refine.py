"""Slot-based specificity refinement expander (`refine_by_slots`, WP4)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

from biosemalign.enums import ConceptStatus, RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.input.context import context_view
from biosemalign.knowledge.providers.lexicon import LexiconProvider, LexiconStore
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import OpContext
from biosemalign.modules.registry import register
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.codes import QualifiedCode, RefinementOutcome
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet, ExpansionStep
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import normalize_text

__all__ = ["RefineBySlotsExpander", "refine_by_slots_factory"]

_STRENGTH_TOKEN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|l|unit|units|iu|%)(?=\W|$)",
    re.IGNORECASE,
)

_UNIT_ALIASES: dict[str, str] = {
    "µg": "mcg",
    "ug": "mcg",
    "units": "unit",
}


def _normalize_strength_atoms(text: str) -> set[tuple[str, str]]:
    atoms: set[tuple[str, str]] = set()
    for raw_num, raw_unit in _STRENGTH_TOKEN.findall(text):
        try:
            num = format(Decimal(raw_num).normalize(), "f")
            if "." in num:
                num = num.rstrip("0").rstrip(".")
        except InvalidOperation:
            num = raw_num
        unit = _UNIT_ALIASES.get(raw_unit.casefold(), raw_unit.casefold())
        atoms.add((num, unit))
    return atoms


def _strength_agrees(slot_value: str, candidate: CandidateConcept) -> bool:
    wanted = _normalize_strength_atoms(slot_value)
    if not wanted:
        norm = normalize_text(slot_value)
        return any(
            norm in normalize_text(label)
            for label in (candidate.preferred_label, *candidate.synonyms)
        )
    cand_atoms: set[tuple[str, str]] = set()
    for label in (candidate.preferred_label, *candidate.synonyms):
        cand_atoms |= _normalize_strength_atoms(label)
    return wanted <= cand_atoms


def _formulation_agrees(slot_value: str, candidate: CandidateConcept, store: LexiconStore) -> bool:
    norm = normalize_text(slot_value)
    if any(
        norm in normalize_text(label) for label in (candidate.preferred_label, *candidate.synonyms)
    ):
        return True
    edges = store.relationships(candidate.vocabulary, [candidate.candidate_id], ["has_dose_form"])
    for edge in edges:
        concepts = store.concepts(edge.object_vocabulary, [edge.object_code])
        target = concepts.get(edge.object_code)
        if target is not None and norm in normalize_text(target.concept_name):
            return True
    return False


def _slot_agrees(
    slot_name: str, slot_value: Any, candidate: CandidateConcept, store: LexiconStore
) -> bool:
    text = str(slot_value)
    if slot_name == "strength":
        return _strength_agrees(text, candidate)
    if slot_name == "formulation":
        return _formulation_agrees(text, candidate, store)
    norm = normalize_text(text)
    return any(
        norm in normalize_text(label) for label in (candidate.preferred_label, *candidate.synonyms)
    )


def _format_slots(slots: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in slots.items())


class RefineBySlotsExpander:
    """Narrow a coarse matched concept to specific candidates that agree with SAU slots."""

    op_id = "refine_by_slots"
    version = "0.1.0"
    uses_model = False
    options: ClassVar[dict[str, Any]] = {}

    def __init__(self, knowledge: KnowledgeRouter, profile: TaskProfile) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.spec = profile.target_spec()

    def expand(
        self, sau: SemanticAlignmentUnit, anchors: Sequence[CandidateConcept]
    ) -> ExpansionSet:
        anchor_codes = [
            QualifiedCode(
                vocabulary=c.vocabulary,
                concept_code=c.candidate_id,
                label=c.preferred_label,
            )
            for c in anchors
        ]
        store = self.knowledge.lexicon
        if store is None or not anchors:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        expand_slots = context_view(sau, "expand", self.profile)
        if not expand_slots:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        target_vocab = self.profile.target.terminology.casefold()
        ordered_anchors = sorted(
            anchors,
            key=lambda c: 0 if c.vocabulary.casefold() == target_vocab else 1,
        )
        chosen = None
        for anchor in ordered_anchors:
            if not self.spec.in_family(anchor.vocabulary):
                continue
            rules = [
                rule
                for rule in self.spec.refinement_rules_for(anchor.concept_class)
                if any(slot in expand_slots for slot in rule.slots)
            ]
            if rules:
                chosen = (anchor, rules)
                break
        if chosen is None:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        anchor, rules = chosen
        anchor_code = QualifiedCode(
            vocabulary=anchor.vocabulary,
            concept_code=anchor.candidate_id,
            label=anchor.preferred_label,
        )
        slots_checked: dict[str, Any] = {}
        allowed_to: set[str] = set()
        allowed_inverse: list[str] = []
        for rule in rules:
            for slot in rule.slots:
                if slot in expand_slots and slot not in slots_checked:
                    slots_checked[slot] = expand_slots[slot]
            for to_class in rule.to_classes:
                allowed_to.add(self.spec.canonical_class(to_class) or to_class)
            for predicate in rule.inverse_of:
                if predicate not in allowed_inverse and self.profile.knowledge.permits(predicate):
                    allowed_inverse.append(predicate)

        if not allowed_inverse or not slots_checked:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        edges = store.incoming_relationships(
            anchor.vocabulary, [anchor.candidate_id], allowed_inverse
        )
        by_vocab: dict[str, list[tuple[str, str]]] = {}
        for edge in edges:
            subject_vocab = edge.subject_vocabulary or anchor.vocabulary
            by_vocab.setdefault(subject_vocab, []).append((edge.subject, edge.predicate))

        agreeing_members: list[ExpansionMember] = []
        seen_ids: set[tuple[str, str]] = set()
        limit = self.profile.limits.members

        for subject_vocab, pairs in by_vocab.items():
            provider = self.knowledge.provider_for(subject_vocab)
            if not isinstance(provider, LexiconProvider):
                continue
            pred_by_code: dict[str, str] = {}
            ordered_codes: list[str] = []
            for code, pred in pairs:
                if code not in pred_by_code:
                    pred_by_code[code] = pred
                    ordered_codes.append(code)
            candidates = provider.candidates_for_codes(
                ordered_codes,
                channel=RetrievalChannel.ONTOLOGY_NEIGHBORHOOD,
                query=sau.mention.text,
                provider_name=self.op_id,
            )
            for candidate in candidates:
                key = (candidate.vocabulary.casefold(), candidate.candidate_id)
                if key in seen_ids:
                    continue
                canon_class = self.spec.canonical_class(candidate.concept_class)
                if canon_class not in allowed_to:
                    continue
                if (
                    self.profile.target.require_active_concept
                    and candidate.status is not ConceptStatus.ACTIVE
                ):
                    continue
                if not all(
                    _slot_agrees(slot, val, candidate, store) for slot, val in slots_checked.items()
                ):
                    continue
                seen_ids.add(key)
                cand_code = QualifiedCode(
                    vocabulary=candidate.vocabulary,
                    concept_code=candidate.candidate_id,
                    label=candidate.preferred_label,
                )
                agreeing_members.append(
                    ExpansionMember(
                        concept=candidate,
                        anchor=anchor_code,
                        hops=1,
                        path=[
                            ExpansionStep(
                                from_code=anchor_code,
                                predicate=pred_by_code[candidate.candidate_id],
                                direction="incoming",
                                to_code=cand_code,
                            )
                        ],
                        via_operator=self.op_id,
                        matched_slots={k: str(v) for k, v in slots_checked.items()},
                    )
                )

        total_agreeing = len(agreeing_members)
        truncated = total_agreeing > limit
        bounded_members = agreeing_members[:limit]
        candidate_codes = [
            QualifiedCode(
                vocabulary=m.concept.vocabulary,
                concept_code=m.concept.candidate_id,
                label=m.concept.preferred_label,
            )
            for m in bounded_members
        ]
        slot_summary = _format_slots(slots_checked)
        if total_agreeing == 1:
            selected = candidate_codes[0]
            outcome = RefinementOutcome(
                status="unique",
                anchor=anchor_code,
                slots_checked=slots_checked,
                candidates=candidate_codes,
                selected=selected,
                detail=(
                    f"refined {anchor_code.vocabulary}:{anchor_code.concept_code} to "
                    f"{selected.vocabulary}:{selected.concept_code} matching {slot_summary}"
                ),
            )
            members = bounded_members
        elif total_agreeing == 0:
            outcome = RefinementOutcome(
                status="none",
                anchor=anchor_code,
                slots_checked=slots_checked,
                candidates=[],
                selected=None,
                detail=(
                    f"no specific concept for {anchor_code.vocabulary}:{anchor_code.concept_code} "
                    f"agreed with {slot_summary}"
                ),
            )
            members = []
        else:
            outcome = RefinementOutcome(
                status="multiple",
                anchor=anchor_code,
                slots_checked=slots_checked,
                candidates=candidate_codes,
                selected=None,
                detail=(
                    f"{total_agreeing} specific concepts for "
                    f"{anchor_code.vocabulary}:{anchor_code.concept_code} agreed with "
                    f"{slot_summary}; keeping coarse anchor"
                ),
            )
            members = bounded_members

        return ExpansionSet(
            expander=self.op_id,
            anchors=anchor_codes,
            members=members,
            refinement=outcome,
            truncated=truncated,
        )


def refine_by_slots_factory(
    context: OpContext, options: Mapping[str, Any]
) -> RefineBySlotsExpander:
    if options:
        raise ConfigurationError(
            f"operator 'refine_by_slots' takes no options, got {sorted(options)}"
        )
    if context.knowledge is None:
        raise ConfigurationError("operator 'refine_by_slots' needs the pipeline's knowledge router")
    return RefineBySlotsExpander(context.knowledge, context.profile)


register("expand", "refine_by_slots", refine_by_slots_factory)
