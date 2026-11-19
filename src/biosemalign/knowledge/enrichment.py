"""Candidate enrichment: attach the relationships the task permits (§6.4).

Only predicates on the profile's allowlist are requested, and only those the
provider actually supports are sent. That second filter is not paranoia: the
design document's own example profile lists ``has_active_moiety``, which RxNav
rejects with a plain-text error page rather than JSON.
"""

from __future__ import annotations

from biosemalign.knowledge.providers.base import KnowledgeProvider
from biosemalign.knowledge.providers.lexicon import LexiconProvider
from biosemalign.knowledge.providers.rxnorm import PREDICATE_TO_TTY, RELA_ONLY_PREDICATES
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept

__all__ = ["enrich_candidates", "supported_predicates"]

log = get_logger(__name__)

#: Unsupported-predicate combinations already reported, so a long batch logs
#: each once instead of once per candidate.
_REPORTED: set[tuple[str, str, tuple[str, ...]]] = set()


def supported_predicates(
    profile: TaskProfile, provider: KnowledgeProvider | None = None
) -> list[str]:
    """Allowlisted predicates that the provider can actually retrieve.

    Predicates the profile allows but the provider cannot serve are dropped
    with a warning rather than sent and failed — the profile is not wrong to
    describe the task, it just names something this provider does not model.
    A lexicon serves exactly the predicates its bundle holds.
    """
    retrievable = (
        set(provider.available_predicates())
        if isinstance(provider, LexiconProvider)
        else set(PREDICATE_TO_TTY) | RELA_ONLY_PREDICATES
    )
    allowed = profile.knowledge.allowed_relationships or sorted(retrievable)
    supported = [p for p in allowed if p in retrievable]
    unsupported = [p for p in allowed if p not in retrievable]
    report = (
        str(getattr(provider, "provider_name", "rxnav")),
        str(getattr(provider, "vocabulary", "")),
        tuple(unsupported),
    )
    if unsupported and report not in _REPORTED:
        _REPORTED.add(report)
        log.warning(
            "enrichment.unsupported_predicates",
            profile=profile.profile_name,
            predicates=unsupported,
            provider=getattr(provider, "provider_name", "rxnav"),
            note="not retrievable from this provider",
        )
    return supported


def enrich_candidates(
    candidates: list[CandidateConcept],
    *,
    provider: KnowledgeProvider,
    profile: TaskProfile,
    max_hops: int = 1,
) -> list[CandidateConcept]:
    """Attach permitted relationships progressively, in ranked batches.

    Once a batch establishes a path to the requested target granularity, lower
    ranked alternatives remain in the package but are not expanded. This keeps
    large studies from paying relationship-call costs for candidates the
    deterministic adjudicator cannot reach.

    Only candidates from the provider's own vocabulary are enriched; matches
    in other vocabularies are concordance evidence, not projection sources.
    """
    predicates = supported_predicates(profile, provider)
    if not predicates:
        return candidates

    vocabulary = str(getattr(provider, "vocabulary", "")).casefold()
    own = [c for c in candidates if not vocabulary or c.vocabulary.casefold() == vocabulary]
    batch_size = profile.retrieval.progressive_enrichment_batch_size
    for start in range(0, len(own), batch_size):
        stop = min(start + batch_size, len(own))
        for candidate in own[start:stop]:
            _enrich_one(
                candidate,
                provider=provider,
                profile=profile,
                predicates=predicates,
                max_hops=max_hops,
            )
        if _target_granularity_is_evidenced(own[:stop], profile):
            break

    return candidates


def _enrich_one(
    candidate: CandidateConcept,
    *,
    provider: KnowledgeProvider,
    profile: TaskProfile,
    predicates: list[str],
    max_hops: int,
) -> None:
    # Provider/cache failures are an integrity boundary. Empty relationships
    # are legitimate evidence; a failed request is not, and must not degrade
    # into a partially enriched candidate that could still be auto-accepted.
    statements = provider.get_relationships(
        candidate.candidate_id, predicates=predicates, max_hops=max_hops
    )

    for statement in statements:
        if not profile.knowledge.permits(statement.predicate):
            continue
        statement.subject_label = candidate.preferred_label
        candidate.relationships.append(statement)


def _target_granularity_is_evidenced(
    candidates: list[CandidateConcept], profile: TaskProfile
) -> bool:
    policy = {
        "ingredient": ("has_ingredient", {"IN"}),
        "precise_ingredient": ("has_precise_ingredient", {"PIN"}),
        "active_moiety": ("has_boss", {"IN", "PIN"}),
    }.get(profile.task.desired_granularity or "")
    if policy is None:
        return False
    predicate, terminal_classes = policy
    return any(
        candidate.concept_class in terminal_classes
        or any(statement.predicate == predicate for statement in candidate.relationships)
        for candidate in candidates
    )
