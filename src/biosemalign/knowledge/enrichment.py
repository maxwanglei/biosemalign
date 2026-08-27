"""Candidate enrichment: attach the relationships the task permits (§6.4).

Only predicates on the profile's allowlist are requested, and only those the
provider actually supports are sent. That second filter is not paranoia: the
design document's own example profile lists ``has_active_moiety``, which RxNav
rejects with a plain-text error page rather than JSON.
"""

from __future__ import annotations

from biosemalign.knowledge.providers.rxnorm import (
    PREDICATE_TO_TTY,
    RELA_ONLY_PREDICATES,
    RxNavProvider,
)
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept

__all__ = ["enrich_candidates", "supported_predicates"]

log = get_logger(__name__)


def supported_predicates(profile: TaskProfile) -> list[str]:
    """Allowlisted predicates that the RxNorm provider can actually retrieve.

    Predicates the profile allows but the provider cannot serve are dropped
    with a warning rather than sent and failed — the profile is not wrong to
    describe the task, it just names something this provider does not model.
    """
    retrievable = set(PREDICATE_TO_TTY) | RELA_ONLY_PREDICATES
    allowed = profile.knowledge.allowed_relationships or sorted(retrievable)
    supported = [p for p in allowed if p in retrievable]
    unsupported = [p for p in allowed if p not in retrievable]
    if unsupported:
        log.warning(
            "enrichment.unsupported_predicates",
            profile=profile.profile_name,
            predicates=unsupported,
            note="not retrievable from RxNav; see /relatypes.json for the valid set",
        )
    return supported


def enrich_candidates(
    candidates: list[CandidateConcept],
    *,
    provider: RxNavProvider,
    profile: TaskProfile,
    max_hops: int = 1,
) -> list[CandidateConcept]:
    """Attach permitted relationships progressively, in ranked batches.

    Once a batch establishes a path to the requested target granularity, lower
    ranked alternatives remain in the package but are not expanded. This keeps
    large studies from paying relationship-call costs for candidates the
    deterministic adjudicator cannot reach.
    """
    predicates = supported_predicates(profile)
    if not predicates:
        return candidates

    batch_size = profile.retrieval.progressive_enrichment_batch_size
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        for candidate in candidates[start:stop]:
            _enrich_one(
                candidate,
                provider=provider,
                profile=profile,
                predicates=predicates,
                max_hops=max_hops,
            )
        if _target_granularity_is_evidenced(candidates[:stop], profile):
            break

    return candidates


def _enrich_one(
    candidate: CandidateConcept,
    *,
    provider: RxNavProvider,
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
