"""Deterministic drug-normalization adjudicator.

Baselines 1–3 of the evaluation comparison in §22.3, and the reference against
which the candidate-constrained LLM will be measured. It selects strictly from
retrieved evidence, walks the RxNorm hierarchy to the requested granularity,
and reports what that walk discards.

It is not a placeholder for the LLM. Most FAERS product strings are resolvable
by terminology alone, and sending them to a model would add cost, variance, and
an invented-identifier risk for no accuracy gain. The adjudicator that ships in
v0.2 exists for the residue this one cannot settle.
"""

from __future__ import annotations

import re

from biosemalign.adjudication.backend import GenerationConfig
from biosemalign.enums import (
    ConceptStatus,
    DecisionType,
    ReadinessStatus,
    RelationshipType,
    RetrievalChannel,
    SemanticLossKind,
)
from biosemalign.exceptions import AdjudicationFailure
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept, SemanticLoss
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage, KnowledgeStatement
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import lexical_similarity, normalize_text

__all__ = ["ADJUDICATOR_VERSION", "DeterministicAdjudicator"]

ADJUDICATOR_VERSION = "0.1.0"

#: Similarity at or above which a fuzzy hit is treated as naming the same
#: concept. Set high: this decides whether a mapping is reported as EXACT, and
#: a false EXACT is worse than an unnecessary review.
_EXACT_SIMILARITY = 0.85

log = get_logger(__name__)

#: Term types that denote a single bare ingredient.
#:
#: ``MIN`` is deliberately absent. A multi-ingredient concept such as
#: "hydrochlorothiazide / lisinopril" looks like an ingredient and is not one:
#: treating it as terminal collapses a combination product to a single concept
#: and misclassifies exposure for both drugs. It must be decomposed through
#: ``has_ingredient`` like any other product.
_INGREDIENT_TTYS = {"IN"}
_PRECISE_INGREDIENT_TTYS = {"PIN"}
#: Term types that are already at ingredient level and need no salt resolution.
_INGREDIENT_LEVEL_TTYS = _INGREDIENT_TTYS | _PRECISE_INGREDIENT_TTYS
# Which attributes a concept carries *at its own level*.
#
# This distinction is essential because RxNav's TTY traversal is bidirectional:
# asking an ingredient for its dose forms returns every form any product
# containing it comes in. Those are downstream products, not attributes of the
# ingredient — reporting them as semantic loss would claim that mapping
# "metoprolol tartrate" to "metoprolol" discards a dose form the source never
# named. Loss is therefore derived from the source concept's own term type.

#: Term types that carry brand identity.
_BRANDED_TTYS = {"SBD", "SBDC", "SBDF", "SBDG", "BN", "BPCK"}
#: Term types that carry a numeric strength.
_STRENGTH_BEARING_TTYS = {"SCD", "SBD", "SCDC", "SBDC", "GPCK", "BPCK"}
#: Term types that carry a dose form.
_DOSE_FORM_BEARING_TTYS = {"SCD", "SBD", "SCDF", "SBDF", "SCDG", "SBDG", "GPCK", "BPCK"}

_STRENGTH = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|l|unit|units|iu|%)\b", re.IGNORECASE
)

#: Predicate a granularity request resolves to.
_GRANULARITY_PREDICATE = {
    "ingredient": "has_ingredient",
    "precise_ingredient": "has_precise_ingredient",
    "active_moiety": "has_boss",
}


class DeterministicAdjudicator:
    """Rule-based, candidate-constrained adjudication for drug normalization."""

    adjudicator_name = "deterministic"

    def __init__(self, profile: TaskProfile) -> None:
        self.profile = profile

    # -- protocol ----------------------------------------------------------

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            backend=self.adjudicator_name,
            model_name=f"drug_normalization_rules@{ADJUDICATOR_VERSION}",
            generation_parameters=GenerationConfig(
                temperature=self.profile.adjudication.temperature
            ).model_dump(),
        )

    def adjudicate(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal:
        """Select concepts from the package and classify the relationship."""
        if sau.readiness.status is ReadinessStatus.INSUFFICIENT_CONTEXT:
            return self._abstain(
                sau,
                RelationshipType.INSUFFICIENT_CONTEXT,
                DecisionType.INSUFFICIENT_CONTEXT,
                missing=sau.readiness.missing_context,
                rationale="required context is missing; mapping would be a guess",
            )

        if not package.candidates:
            return self._abstain(
                sau,
                RelationshipType.UNSUPPORTED_MAPPING,
                DecisionType.NO_VALID_MAPPING,
                rationale=(
                    f"no {sau.task.target_terminology} candidate was retrieved for "
                    f"{sau.mention.text!r} on any enabled channel"
                ),
            )

        top = package.candidates[0]

        # A candidate that resembles nothing in the source is not a mapping.
        # RxNav's approximate matcher always returns its nearest neighbour, so
        # without this floor "ZZZQQQ NOT A DRUG 999" maps confidently to
        # "Myosotis arvensis extract" (§7.2: the adjudicator must be able to
        # reject every candidate).
        similarity = self._similarity(sau, top)
        floor = self.profile.routing.rejection_lexical_similarity
        if similarity < floor and not self._has_lexical_evidence(top):
            return self._abstain(
                sau,
                RelationshipType.UNSUPPORTED_MAPPING,
                DecisionType.NO_VALID_MAPPING,
                rationale=(
                    f"best candidate {top.candidate_id} ({top.preferred_label!r}) has lexical "
                    f"similarity {similarity:.2f} to {sau.mention.text!r}, below the "
                    f"rejection floor {floor:.2f}, and no exact or spelling-corrected match "
                    "corroborates it"
                ),
            )

        granularity = sau.task.desired_granularity
        predicate = _GRANULARITY_PREDICATE.get(granularity or "")
        if predicate is None:
            return self._abstain(
                sau,
                RelationshipType.UNSUPPORTED_MAPPING,
                DecisionType.ABSTAINED,
                rationale=(
                    "deterministic drug normalization has no policy for requested "
                    f"granularity {granularity!r}; it will not silently default to ingredient"
                ),
            )
        targets = self._targets_at_granularity(top, predicate)

        if not targets:
            return self._unresolved_granularity(sau, top, predicate)

        if len(targets) > 1:
            return self._combination(sau, top, targets)

        return self._single(sau, top, targets[0])

    # -- outcomes ----------------------------------------------------------

    def _single(
        self,
        sau: SemanticAlignmentUnit,
        source_concept: CandidateConcept,
        target: SelectedConcept,
    ) -> AlignmentProposal:
        """One concept at the requested granularity."""
        source_level = _to_selected(source_concept)
        is_same_concept = target.concept_id == source_concept.candidate_id

        # EXACT is a strong claim and needs lexical grounding, not merely the
        # observation that the candidate happens to sit at the requested
        # granularity. Without this, any approximate hit on an ingredient
        # concept would be reported as an exact mapping.
        if is_same_concept and self._is_exact_match(sau, source_concept, target):
            relationship = RelationshipType.EXACT
        elif is_same_concept:
            relationship = RelationshipType.RELATED_NOT_EQUIVALENT
        else:
            relationship = RelationshipType.TARGET_BROADER_THAN_SOURCE

        target.relationship_to_source = relationship
        loss = (
            []
            if relationship is RelationshipType.EXACT
            else self._semantic_loss(sau, source_concept, target)
        )

        return self._proposal(
            sau,
            decision=DecisionType.MAPPED,
            selected=[target],
            source_level=source_level,
            analysis_level=target,
            relationship=relationship,
            semantic_loss=loss,
            source_concept=source_concept,
            rationale=(
                f"selected {target.concept_id} ({target.label}) from "
                f"{source_concept.concept_class or 'concept'} {source_concept.candidate_id}"
            ),
        )

    def _combination(
        self,
        sau: SemanticAlignmentUnit,
        source_concept: CandidateConcept,
        targets: list[SelectedConcept],
    ) -> AlignmentProposal:
        """A combination product: every component is required (§8.2)."""
        for target in targets:
            target.relationship_to_source = RelationshipType.COMPONENT_OF

        loss = self._semantic_loss(sau, source_concept, targets[0])
        loss.append(
            SemanticLoss(
                kind=SemanticLossKind.COMBINATION_COMPONENT,
                description=(
                    "source is a combination product; it is represented by "
                    f"{len(targets)} separate ingredient concepts and cannot be "
                    "collapsed to one without misclassifying exposure"
                ),
                source_value=source_concept.preferred_label,
            )
        )

        return self._proposal(
            sau,
            decision=DecisionType.MULTI_CONCEPT_MAPPED,
            selected=targets,
            source_level=_to_selected(source_concept),
            analysis_level=None,
            relationship=RelationshipType.MULTIPLE_COMPONENTS_REQUIRED,
            semantic_loss=loss,
            source_concept=source_concept,
            rationale=(
                f"{source_concept.candidate_id} resolves to {len(targets)} ingredients: "
                + ", ".join(f"{t.label} ({t.concept_id})" for t in targets)
            ),
        )

    def _unresolved_granularity(
        self, sau: SemanticAlignmentUnit, top: CandidateConcept, predicate: str
    ) -> AlignmentProposal:
        """The top candidate exists but cannot be walked to the requested level."""
        selected = _to_selected(top)

        # An obsolete concept has no navigable relationships at all, which is a
        # different problem from a current concept that simply sits at the
        # wrong level — and it needs a different fix from the reviewer.
        if top.status is not ConceptStatus.ACTIVE:
            rationale = (
                f"top candidate {top.candidate_id} has status {top.status.value} and "
                "carries no navigable relationships; it cannot be resolved to "
                f"{sau.task.desired_granularity}"
            )
        else:
            rationale = (
                f"{top.candidate_id} ({top.concept_class or 'concept'}) has no "
                f"{predicate} relationship to reach {sau.task.desired_granularity}"
            )

        selected.relationship_to_source = RelationshipType.RELATED_NOT_EQUIVALENT
        return self._proposal(
            sau,
            decision=DecisionType.MAPPED,
            selected=[selected],
            source_level=selected,
            analysis_level=None,
            relationship=RelationshipType.RELATED_NOT_EQUIVALENT,
            semantic_loss=[],
            source_concept=top,
            rationale=rationale,
            request_review=True,
            missing=[f"{predicate} relationship for {top.candidate_id}"],
        )

    def _abstain(
        self,
        sau: SemanticAlignmentUnit,
        relationship: RelationshipType,
        decision: DecisionType,
        *,
        missing: list[str] | None = None,
        rationale: str,
    ) -> AlignmentProposal:
        if not self.profile.permits_relation(relationship):
            raise AdjudicationFailure(
                f"profile {self.profile.profile_name} cannot represent abstention relation "
                f"{relationship.value}"
            )
        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=decision,
            relationship_to_source=relationship,
            missing_information=list(missing or []),
            rationale=rationale,
            requested_human_review=decision is not DecisionType.NO_VALID_MAPPING,
        )

    # -- helpers -----------------------------------------------------------

    def _similarity(self, sau: SemanticAlignmentUnit, candidate: CandidateConcept) -> float:
        """Best lexical agreement between the mention and the candidate's names."""
        return max(
            [lexical_similarity(sau.mention.text, candidate.preferred_label)]
            + [lexical_similarity(sau.mention.text, s) for s in candidate.synonyms]
        )

    def _has_lexical_evidence(self, candidate: CandidateConcept) -> bool:
        """Whether a terminology-native lookup, not fuzzy scoring, found this."""
        return bool(
            candidate.channels
            & {
                RetrievalChannel.EXACT_LOOKUP,
                RetrievalChannel.NORMALIZED_LOOKUP,
                RetrievalChannel.SPELLING_SUGGESTION,
            }
        )

    def _is_exact_match(
        self,
        sau: SemanticAlignmentUnit,
        candidate: CandidateConcept,
        target: SelectedConcept,
    ) -> bool:
        """Is the target demonstrably the same thing the source named?"""
        mention = sau.mention.normalized_text
        if normalize_text(target.label) == mention:
            return True
        if any(normalize_text(s) == mention for s in candidate.synonyms):
            return True
        # An exact or normalized RxNorm lookup is a terminology-native identity
        # claim, which outranks any string comparison we could make ourselves.
        if self._has_lexical_evidence(candidate):
            return True
        return self._similarity(sau, candidate) >= _EXACT_SIMILARITY

    def _targets_at_granularity(
        self, candidate: CandidateConcept, predicate: str
    ) -> list[SelectedConcept]:
        """Concepts reachable at the requested granularity.

        A candidate that is already at the requested level is its own target —
        without this, "metoprolol" would fail to map, since an ingredient has
        no ``has_ingredient`` edge pointing at itself.
        """
        wanted_ttys = (
            _INGREDIENT_TTYS
            if predicate == "has_ingredient"
            else _PRECISE_INGREDIENT_TTYS
            if predicate == "has_precise_ingredient"
            else _INGREDIENT_TTYS
            if predicate == "has_boss"
            else set()
        )
        if candidate.concept_class in wanted_ttys:
            return [_to_selected(candidate)]

        statements = [s for s in candidate.relationships if s.predicate == predicate]
        return [_statement_to_selected(s, candidate) for s in _dedupe(statements)]

    def _semantic_loss(
        self,
        sau: SemanticAlignmentUnit,
        source_concept: CandidateConcept,
        target: SelectedConcept,
    ) -> list[SemanticLoss]:
        """What the coarser target concept does not carry (§8.2, goal 7)."""
        losses: list[SemanticLoss] = []
        source_label = source_concept.preferred_label
        tty = source_concept.concept_class or ""

        if tty in _STRENGTH_BEARING_TTYS:
            strength = _STRENGTH.search(source_label) or _STRENGTH.search(sau.mention.text)
            if strength and not _STRENGTH.search(target.label):
                losses.append(
                    SemanticLoss(
                        kind=SemanticLossKind.STRENGTH,
                        description="target concept does not carry product strength",
                        source_value=strength.group(0).strip(),
                    )
                )

        if tty in _DOSE_FORM_BEARING_TTYS:
            labels = [
                s.object_label
                for s in source_concept.relationships
                if s.predicate == "has_dose_form" and s.object_label
            ]
            if labels:
                losses.append(
                    SemanticLoss(
                        kind=SemanticLossKind.DOSE_FORM,
                        description="target concept does not carry dose form",
                        source_value=", ".join(labels),
                    )
                )

        if tty in _PRECISE_INGREDIENT_TTYS and source_concept.candidate_id != target.concept_id:
            losses.append(
                SemanticLoss(
                    kind=SemanticLossKind.SALT_FORM,
                    description="source names a precise ingredient; target is the base ingredient",
                    source_value=source_label,
                )
            )
        elif tty not in _INGREDIENT_LEVEL_TTYS:
            precise = [
                s
                for s in source_concept.relationships
                if s.predicate == "has_precise_ingredient" and s.object_id != target.concept_id
            ]
            if precise:
                losses.append(
                    SemanticLoss(
                        kind=SemanticLossKind.SALT_FORM,
                        description=(
                            "target is the base ingredient; the source names a specific salt "
                            "or precise ingredient, which the target does not distinguish"
                        ),
                        source_value=", ".join(s.object_label or s.object_id for s in precise),
                    )
                )

        if tty in _BRANDED_TTYS:
            brands = [
                s.object_label or s.object_id
                for s in source_concept.relationships
                if s.predicate == "has_tradename"
            ]
            losses.append(
                SemanticLoss(
                    kind=SemanticLossKind.BRAND_IDENTITY,
                    description="target concept is generic; brand identity is not preserved",
                    source_value=", ".join(brands) or source_label,
                )
            )

        route = sau.context.route or sau.attributes.route
        if route:
            losses.append(
                SemanticLoss(
                    kind=SemanticLossKind.ROUTE,
                    description="target concept does not carry route of administration",
                    source_value=route,
                )
            )

        return losses

    def _proposal(
        self,
        sau: SemanticAlignmentUnit,
        *,
        decision: DecisionType,
        selected: list[SelectedConcept],
        source_level: SelectedConcept | None,
        analysis_level: SelectedConcept | None,
        relationship: RelationshipType,
        semantic_loss: list[SemanticLoss],
        source_concept: CandidateConcept,
        rationale: str,
        request_review: bool = False,
        missing: list[str] | None = None,
    ) -> AlignmentProposal:
        if not self.profile.permits_relation(relationship):
            return self._abstain(
                sau,
                RelationshipType.UNSUPPORTED_MAPPING,
                DecisionType.ABSTAINED,
                rationale=(
                    f"deterministic result requires relationship {relationship.value}, "
                    f"which profile {self.profile.profile_name} does not permit"
                ),
            )
        review = request_review or source_concept.status is not ConceptStatus.ACTIVE

        unsupported: list[str] = []
        if source_concept.status is not ConceptStatus.ACTIVE:
            unsupported.append(
                f"mapping rests on {source_concept.candidate_id}, whose RxNorm status is "
                f"{source_concept.status.value}"
            )

        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=decision,
            selected_concepts=selected,
            source_level_concept=source_level,
            analysis_level_concept=analysis_level,
            relationship_to_source=relationship,
            preserved_attributes=sau.attributes.model_dump(
                exclude_none=True, exclude={"extensions"}
            ),
            semantic_loss=semantic_loss,
            unsupported_inferences=unsupported,
            missing_information=list(missing or []) + list(sau.readiness.missing_context),
            evidence_references=[_reference(s) for s in source_concept.relationships],
            rationale=rationale,
            requested_human_review=review,
        )


# -- module helpers --------------------------------------------------------


def _to_selected(candidate: CandidateConcept) -> SelectedConcept:
    return SelectedConcept(
        concept_id=candidate.candidate_id,
        label=candidate.preferred_label,
        vocabulary=candidate.vocabulary,
        concept_class=candidate.concept_class,
        status=candidate.status,
    )


def _statement_to_selected(
    statement: KnowledgeStatement, source: CandidateConcept
) -> SelectedConcept:
    return SelectedConcept(
        concept_id=statement.object_id,
        label=statement.object_label or statement.object_id,
        vocabulary=source.vocabulary,
        concept_class=statement.extensions.get("target_tty"),
        # Reached by traversal from a retrieved candidate; the traversal itself
        # is retrieved evidence, but the concept's own status was not fetched.
        status=ConceptStatus.UNKNOWN,
    )


def _dedupe(statements: list[KnowledgeStatement]) -> list[KnowledgeStatement]:
    seen: set[str] = set()
    unique: list[KnowledgeStatement] = []
    for statement in statements:
        if statement.object_id in seen:
            continue
        seen.add(statement.object_id)
        unique.append(statement)
    return unique


def _reference(statement: KnowledgeStatement) -> str:
    return f"{statement.subject_id}--{statement.predicate}->{statement.object_id}"
