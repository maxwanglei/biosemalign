"""Observable routing features (§9.1).

Every feature is checkable against retrieved evidence. There is deliberately no
model self-confidence score: an unvalidated number the adjudicator invented
about its own reliability is exactly the input a precision-first policy should
not depend on.
"""

from __future__ import annotations

from biosemalign.enums import ConceptStatus, RelationshipType, RetrievalChannel, Severity
from biosemalign.knowledge.candidate_fusion import channel_agreement
from biosemalign.schemas.decision import AlignmentProposal, RoutingFeatures, ValidationResult
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import lexical_similarity, normalize_text

__all__ = ["extract_features", "has_unresolved_salt"]


#: Term types that are already at ingredient granularity.
_INGREDIENT_LEVEL_TTYS = {"IN", "PIN"}


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
    if candidate.concept_class in _INGREDIENT_LEVEL_TTYS:
        return False
    return len(set(candidate.related_ids("has_precise_ingredient"))) > 1


def extract_features(
    *,
    sau: SemanticAlignmentUnit,
    package: KnowledgePackage,
    proposal: AlignmentProposal,
    validation: ValidationResult,
) -> RoutingFeatures:
    """Assemble the feature vector behind the routing decision."""
    source_level = proposal.source_level_concept
    target_vocabulary = sau.task.target_terminology
    source_candidate = _selected_source_candidate(package, proposal, target_vocabulary)

    exact_match = False
    synonym_match = False
    rank: int | None = None
    agreement = 0
    concept_active = False
    similarity: float | None = None

    if source_candidate is not None:
        normalized_mention = sau.mention.normalized_text
        exact_match = normalize_text(source_candidate.preferred_label) == normalized_mention
        synonym_match = any(
            normalize_text(s) == normalized_mention for s in source_candidate.synonyms
        )
        agreement = channel_agreement(source_candidate)
        similarity = max(
            [lexical_similarity(sau.mention.text, source_candidate.preferred_label)]
            + [lexical_similarity(sau.mention.text, s) for s in source_candidate.synonyms]
        )
        rank = min(
            (e.rank for e in source_candidate.retrieval_evidence if e.rank is not None),
            default=None,
        )

    concept_active = _all_final_concepts_active(package, proposal, validation, target_vocabulary)

    margin: float | None = None
    if source_candidate is not None:
        comparable = [
            candidate
            for candidate in package.candidates
            if candidate.candidate_id != source_candidate.candidate_id
            and candidate.channels & source_candidate.channels
        ]
        if comparable:
            margin = source_candidate.best_score - max(c.best_score for c in comparable)

    return RoutingFeatures(
        exact_terminology_match=exact_match,
        official_synonym_match=synonym_match,
        official_crosswalk_available=(
            source_candidate is not None
            and RetrievalChannel.OFFICIAL_CROSSWALK in source_candidate.channels
        ),
        retrieval_rank=rank,
        lexical_similarity=similarity,
        channel_agreement=agreement,
        top_candidate_margin=margin,
        concept_is_active=concept_active,
        relationship_type=proposal.relationship_to_source,
        missing_attributes=list(sau.readiness.missing_context),
        validator_error_count=sum(1 for f in validation.findings if f.severity is Severity.ERROR),
        validator_warning_count=sum(
            1 for f in validation.findings if f.severity is Severity.WARNING
        ),
        multiple_concepts_required=(
            proposal.relationship_to_source is RelationshipType.MULTIPLE_COMPONENTS_REQUIRED
        ),
        known_difficult_pattern=_difficult_pattern(sau, package, source_level, source_candidate),
    )


def _selected_source_candidate(
    package: KnowledgePackage,
    proposal: AlignmentProposal,
    target_vocabulary: str,
) -> CandidateConcept | None:
    """Return the directly retrieved candidate the proposal actually used."""
    if proposal.source_level_concept is not None:
        candidate = package.candidate_by_qualified_id(
            target_vocabulary, proposal.source_level_concept.concept_id
        )
        if candidate is not None:
            return candidate
    for selected in proposal.selected_concepts:
        candidate = package.candidate_by_qualified_id(target_vocabulary, selected.concept_id)
        if candidate is not None:
            return candidate
    return None


def _all_final_concepts_active(
    package: KnowledgePackage,
    proposal: AlignmentProposal,
    validation: ValidationResult,
    target_vocabulary: str,
) -> bool:
    """Use canonical evidence, including explicit provider verification."""
    if not proposal.selected_concepts:
        return False
    verified = {
        finding.concept_id
        for finding in validation.findings
        if finding.code == "STATUS_VERIFIED_ACTIVE" and finding.concept_id is not None
    }
    evidence = package.qualified_concept_evidence_index()
    vocabulary_key = target_vocabulary.casefold()
    for selected in proposal.selected_concepts:
        canonical = evidence.get((vocabulary_key, selected.concept_id))
        if canonical is None:
            return False
        if canonical.status is not ConceptStatus.ACTIVE and selected.concept_id not in verified:
            return False
    return True


def _difficult_pattern(
    sau: SemanticAlignmentUnit,
    package: KnowledgePackage,
    source_level: object,
    source_candidate: CandidateConcept | None,
) -> str | None:
    """Name the difficulty when the case matches a known-hard category (§22.4).

    Reported separately from the routing decision so that evaluation can break
    accuracy down by difficulty rather than reporting one average dominated by
    easy generic names.
    """
    if not package.candidates:
        return "no_candidates"

    concept_id = getattr(source_level, "concept_id", None)
    if has_unresolved_salt(package, concept_id, sau.task.target_terminology):
        return "unresolved_salt_ambiguity"

    candidate = source_candidate
    if candidate is None:
        return "selection_not_directly_retrieved"
    if candidate.status is not ConceptStatus.ACTIVE:
        return "historical_or_non_primary_source_concept"
    if RetrievalChannel.TERMINOLOGY_APPROXIMATE_HISTORICAL in candidate.channels:
        return "historical_channel_only"
    if RetrievalChannel.SPELLING_SUGGESTION in candidate.channels:
        return "spelling_corrected"
    if len(sau.mention.text.strip()) <= 3:
        return "very_short_reported_name"

    return None
