"""Drug-domain routing features and review triggers (unresolved salt)."""

from __future__ import annotations

from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, RoutingFeatures, ValidationResult
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.targets import spec_for_vocabulary

__all__ = [
    "drug_difficult_pattern",
    "drug_routing_reasons",
    "has_unresolved_salt",
]


def has_unresolved_salt(
    package: KnowledgePackage,
    concept_id: str | None,
    vocabulary: str | None = None,
) -> bool:
    """Whether the source names a product whose salt form could not be pinned.

    Only meaningful for *products*. An ingredient concept reaches every salt
    beneath it by definition — "metoprolol" is related to both the succinate
    and the tartrate — and counting those as ambiguity would send every
    correctly-mapped bare ingredient name to review.
    """
    if concept_id is None:
        return False
    candidate = (
        package.candidate_by_qualified_id(vocabulary, concept_id)
        if vocabulary is not None
        else package.candidate_by_id(concept_id)
    )
    if candidate is None:
        return False
    spec = spec_for_vocabulary(candidate.vocabulary)
    if spec is None or spec.salt is None:
        return False
    ingredient_level = set(spec.salt.precise_classes) | set(spec.salt.base_classes)
    if spec.canonical_class(candidate.concept_class) in ingredient_level:
        return False
    return len(set(candidate.related_ids(spec.salt.precise_predicate))) > 1


def drug_difficult_pattern(
    sau: SemanticAlignmentUnit,
    package: KnowledgePackage,
    source_level: object,
    source_candidate: CandidateConcept | None,
) -> str | None:
    del source_candidate
    concept_id = getattr(source_level, "concept_id", None)
    vocabulary = sau.task.source_vocabulary(getattr(source_level, "vocabulary", None))
    if has_unresolved_salt(package, concept_id, vocabulary):
        return "unresolved_salt_ambiguity"
    return None


def drug_routing_reasons(
    proposal: AlignmentProposal,
    validation: ValidationResult,
    features: RoutingFeatures,
    package: KnowledgePackage,
    profile: TaskProfile,
) -> list[str]:
    del validation, features
    check = profile.routing.options.get(
        "unresolved_salt_ambiguity_requires_review",
        profile.routing.unresolved_salt_ambiguity_requires_review,
    )
    if not check:
        return []
    source_id = (
        proposal.source_level_concept.concept_id
        if proposal.source_level_concept is not None
        else None
    )
    source_vocabulary = profile.target.source_vocabulary(
        proposal.source_level_concept.vocabulary
        if proposal.source_level_concept is not None
        else None
    )
    if has_unresolved_salt(package, source_id, source_vocabulary):
        return [
            "the source names a product with more than one reachable precise "
            "ingredient; the salt form is unresolved"
        ]
    return []
