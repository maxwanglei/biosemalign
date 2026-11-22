"""Deterministic rule-based resolver (`rules`)."""

from __future__ import annotations

from typing import Any

from biosemalign.enums import (
    Authority,
    ConceptStatus,
    DecisionType,
    LinkKind,
    ReadinessStatus,
    RelationshipType,
    RetrievalChannel,
)
from biosemalign.exceptions import AdjudicationFailure
from biosemalign.input.context import context_view
from biosemalign.logging import get_logger
from biosemalign.modules.link.targets import targets_at
from biosemalign.modules.resolve.backend import GenerationConfig
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.codes import RefinementOutcome
from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept, SemanticLoss
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    KnowledgePackage,
    KnowledgeStatement,
    statement_object_class,
)
from biosemalign.schemas.links import LinkSet
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit, canonical_preserved_attributes
from biosemalign.util import lexical_similarity, normalize_text

__all__ = ["RESOLVER_VERSION", "DeterministicResolver"]

RESOLVER_VERSION = "0.5.0"

_LEXICAL_CHANNELS = frozenset(
    {
        RetrievalChannel.EXACT_LOOKUP,
        RetrievalChannel.NORMALIZED_LOOKUP,
        RetrievalChannel.SPELLING_SUGGESTION,
    }
)

#: Similarity at or above which a fuzzy hit is treated as naming the same
#: concept. Set high: this decides whether a mapping is reported as EXACT, and
#: a false EXACT is worse than an unnecessary review.
_EXACT_SIMILARITY = 0.85

log = get_logger(__name__)


class DeterministicResolver:
    """Rule-based, candidate-constrained resolution for drug normalization."""

    op_id = "rules"
    version = RESOLVER_VERSION
    uses_model = False
    requires_network = False

    def __init__(self, profile: TaskProfile) -> None:
        self.profile = profile
        self.spec = profile.target_spec()

    @property
    def options(self) -> dict[str, Any]:
        return {}

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            backend="deterministic",
            model_name=f"rules@{self.version}",
            generation_parameters=GenerationConfig().model_dump(),
        )

    def resolve(self, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> AlignmentProposal:
        """Select concepts from the package and classify the relationship."""
        if sau.readiness.status is ReadinessStatus.INSUFFICIENT_CONTEXT:
            return self._abstain(
                sau,
                RelationshipType.INSUFFICIENT_CONTEXT,
                DecisionType.INSUFFICIENT_CONTEXT,
                missing=sau.readiness.missing_context,
                rationale="required context is missing; mapping would be a guess",
            )

        # Identity is the best-ranked candidate in the target terminology.
        # Candidates from other vocabularies are concordance evidence only.
        target = sau.task.target_terminology.casefold()
        top: CandidateConcept | None = None
        anchor: CandidateConcept | None = None
        if (
            package.refinement is not None
            and package.refinement.status == "unique"
            and package.refinement.selected is not None
            and package.refinement.selected.vocabulary.casefold() == target
        ):
            top = package.candidate_by_qualified_id(
                package.refinement.selected.vocabulary,
                package.refinement.selected.concept_code,
            )
            if top is not None and package.refinement.anchor is not None:
                anchor = package.candidate_by_qualified_id(
                    package.refinement.anchor.vocabulary,
                    package.refinement.anchor.concept_code,
                )
        if top is None:
            top = next((c for c in package.candidates if c.vocabulary.casefold() == target), None)
        if top is None:
            return self._abstain(
                sau,
                RelationshipType.UNSUPPORTED_MAPPING,
                DecisionType.NO_VALID_MAPPING,
                rationale=(
                    f"no {sau.task.target_terminology} candidate was retrieved for "
                    f"{sau.mention.text!r} on any enabled channel"
                ),
            )
        return self.project(
            sau, top, links=package.links, refinement=package.refinement, anchor=anchor
        )

    def project(
        self,
        sau: SemanticAlignmentUnit,
        top: CandidateConcept,
        *,
        links: LinkSet | None = None,
        refinement: RefinementOutcome | None = None,
        anchor: CandidateConcept | None = None,
    ) -> AlignmentProposal:
        """Resolve a chosen source identity to the requested target level.

        Separate from :meth:`resolve` so that another resolver can choose
        the identity — from cross-vocabulary concordance or a model — and still
        report relationship and semantic loss under the same rules.
        """
        if sau.readiness.status is ReadinessStatus.INSUFFICIENT_CONTEXT:
            return self._abstain(
                sau,
                RelationshipType.INSUFFICIENT_CONTEXT,
                DecisionType.INSUFFICIENT_CONTEXT,
                missing=sau.readiness.missing_context,
                rationale="required context is missing; mapping would be a guess",
            )

        similarity = self._similarity(sau, top)
        if anchor is not None:
            similarity = max(similarity, self._similarity(sau, anchor))
        corroborated = self._has_lexical_evidence(top) or (
            anchor is not None and self._has_lexical_evidence(anchor)
        )
        floor = self.profile.routing.rejection_lexical_similarity
        if similarity < floor and not corroborated:
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
        if not self.spec.has_granularity(granularity):
            return self._abstain(
                sau,
                RelationshipType.UNSUPPORTED_MAPPING,
                DecisionType.ABSTAINED,
                rationale=(
                    "deterministic normalization has no policy for requested "
                    f"granularity {granularity!r}"
                ),
            )
        targets = (
            self._targets_from_links(top, links)
            if links is not None and links.granularity == granularity
            else self._targets_at_granularity(top, granularity)
        )

        if not targets:
            return self._unresolved_granularity(sau, top, granularity, refinement=refinement)

        if len(targets) > 1:
            return self._combination(sau, top, targets, refinement=refinement)

        return self._single(sau, top, targets[0], refinement=refinement)

    # -- outcomes ----------------------------------------------------------

    def _single(
        self,
        sau: SemanticAlignmentUnit,
        source_concept: CandidateConcept,
        target: SelectedConcept,
        *,
        refinement: RefinementOutcome | None = None,
    ) -> AlignmentProposal:
        """One concept at the requested level."""
        source_level = _to_selected(source_concept)
        is_same_concept = target.concept_id == source_concept.candidate_id

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
            else self._semantic_loss(sau, source_concept, [target])
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
            refinement=refinement,
        )

    def _combination(
        self,
        sau: SemanticAlignmentUnit,
        source_concept: CandidateConcept,
        targets: list[SelectedConcept],
        *,
        refinement: RefinementOutcome | None = None,
    ) -> AlignmentProposal:
        """A combination product: every component is required (§8.2)."""
        for target in targets:
            target.relationship_to_source = RelationshipType.COMPONENT_OF

        loss = self._semantic_loss(sau, source_concept, targets)
        combination = self.spec.combination_loss(
            sau.task.desired_granularity,
            count=len(targets),
            source_label=source_concept.preferred_label,
        )
        if combination is not None:
            loss.append(combination)

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
            refinement=refinement,
        )

    def _unresolved_granularity(
        self,
        sau: SemanticAlignmentUnit,
        top: CandidateConcept,
        granularity: str | None,
        *,
        refinement: RefinementOutcome | None = None,
    ) -> AlignmentProposal:
        """The top candidate exists but cannot be walked to the requested level."""
        selected = _to_selected(top)
        level = self.spec.granularity(granularity)
        predicate = self.spec.predicate(granularity)

        if top.status is not ConceptStatus.ACTIVE:
            rationale = (
                f"top candidate {top.candidate_id} has status {top.status.value} and "
                "carries no navigable relationships; it cannot be resolved to "
                f"{granularity}"
            )
        elif level is not None and not level.via:
            rationale = (
                f"{top.candidate_id} ({top.concept_class or 'unknown'}) is not of terminal "
                f"class {sorted(level.terminal)} for identity level {granularity}"
            )
        else:
            rationale = (
                f"{top.candidate_id} ({top.concept_class or 'concept'}) has no "
                f"{predicate} relationship to reach {granularity}"
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
            missing=[f"{predicate or granularity} relationship for {top.candidate_id}"],
            refinement=refinement,
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
            preserved_attributes=_preserved_attributes(sau),
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
        return bool(candidate.channels & _LEXICAL_CHANNELS)

    def _has_direct_lexical_evidence(self, candidate: CandidateConcept) -> bool:
        """Whether non-model terminology lookup corroborated this candidate (M4, G10)."""
        return any(
            e.channel in _LEXICAL_CHANNELS and not e.from_model_suggestion
            for e in candidate.retrieval_evidence
        )

    def _is_exact_match(
        self,
        sau: SemanticAlignmentUnit,
        candidate: CandidateConcept,
        target: SelectedConcept,
    ) -> bool:
        """Is the target demonstrably the same thing the source named (against original text)?"""
        mention = sau.mention.normalized_text
        if normalize_text(target.label) == mention:
            return True
        if any(normalize_text(s) == mention for s in candidate.synonyms):
            return True
        if self._has_direct_lexical_evidence(candidate):
            return True
        return self._similarity(sau, candidate) >= _EXACT_SIMILARITY

    def targets_for(
        self, candidate: CandidateConcept, granularity: str | None
    ) -> list[SelectedConcept]:
        """Concepts ``candidate`` reaches at ``granularity``; empty when unknown."""
        if not self.spec.has_granularity(granularity):
            return []
        return self._targets_at_granularity(candidate, granularity)

    def _targets_from_links(
        self, candidate: CandidateConcept, links: LinkSet
    ) -> list[SelectedConcept]:
        """Target concepts from pre-derived ``LinkSet`` edges whose ``via`` is ``candidate``."""
        wanted = (candidate.vocabulary.casefold(), candidate.candidate_id)
        selected: list[SelectedConcept] = []
        seen: set[tuple[str, str]] = set()
        for link in links.links:
            if (link.via.vocabulary.casefold(), link.via.concept_code) != wanted:
                continue
            target_key = (link.target.vocabulary.casefold(), link.target.concept_code)
            if target_key in seen:
                continue
            seen.add(target_key)
            if link.kind is LinkKind.EQUIVALENCE and target_key == wanted:
                selected.append(_to_selected(candidate))
                continue
            preds = set(link.predicates)
            statement = next(
                (
                    s
                    for s in candidate.relationships
                    if s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
                    and s.object_id == link.target.concept_code
                    and (s.object_vocabulary or candidate.vocabulary).casefold()
                    == link.target.vocabulary.casefold()
                    and (not preds or s.predicate in preds)
                ),
                None,
            )
            if statement is not None:
                selected.append(_statement_to_selected(statement, candidate))
            else:
                selected.append(
                    SelectedConcept(
                        concept_id=link.target.concept_code,
                        label=link.target.label or link.target.concept_code,
                        vocabulary=link.target.vocabulary,
                        concept_class=(
                            self.spec.inferred_class(link.predicates[-1])
                            if link.predicates
                            else None
                        ),
                        status=ConceptStatus.UNKNOWN,
                    )
                )
        return selected

    def _targets_at_granularity(
        self, candidate: CandidateConcept, granularity: str | None
    ) -> list[SelectedConcept]:
        """Concepts reachable at the requested level (``modules.link.targets.targets_at``)."""
        return [
            _to_selected(candidate)
            if reached.statement is None
            else _statement_to_selected(reached.statement, candidate)
            for reached in targets_at(candidate, self.spec, granularity)
        ]

    def _semantic_loss(
        self,
        sau: SemanticAlignmentUnit,
        source_concept: CandidateConcept,
        targets: list[SelectedConcept],
    ) -> list[SemanticLoss]:
        """What the coarser target does not carry (§8.2, goal 7), by the target spec."""
        return self.spec.expected_losses(
            source_concept,
            selected_ids=[target.concept_id for target in targets],
            target_label=targets[0].label if targets else None,
            mention_text=sau.mention.text,
            context=context_view(sau, "resolve", self.profile),
        )

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
        refinement: RefinementOutcome | None = None,
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
                f"mapping rests on {source_concept.candidate_id}, whose {source_concept.vocabulary} status is "
                f"{source_concept.status.value}"
            )

        full_rationale = rationale
        if refinement is not None and refinement.status in {"unique", "none", "multiple"}:
            full_rationale = f"{rationale}; {refinement.detail}"

        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=decision,
            selected_concepts=selected,
            source_level_concept=source_level,
            analysis_level_concept=analysis_level,
            relationship_to_source=relationship,
            preserved_attributes=_preserved_attributes(sau),
            semantic_loss=semantic_loss,
            unsupported_inferences=unsupported,
            missing_information=list(missing or []) + list(sau.readiness.missing_context),
            evidence_references=[
                _reference(s)
                for s in source_concept.relationships
                if s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
            ],
            rationale=full_rationale,
            requested_human_review=review,
            refinement=refinement,
        )


def _preserved_attributes(sau: SemanticAlignmentUnit) -> dict[str, object]:
    return canonical_preserved_attributes(sau.attributes)


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
        vocabulary=statement.object_vocabulary or source.vocabulary,
        concept_class=statement_object_class(statement),
        status=ConceptStatus.UNKNOWN,
    )


def _reference(statement: KnowledgeStatement) -> str:
    return f"{statement.subject_id}--{statement.predicate}->{statement.object_id}"
