"""Merge and rank candidates from multiple retrieval channels (§6.3).

Fusion never discards retrieval evidence. Agreement between independent
channels is a routing feature (§9.1), and collapsing three channels into one
"best score" would throw away the signal that makes automatic acceptance
defensible.
"""

from __future__ import annotations

from biosemalign.enums import ConceptStatus, RetrievalChannel
from biosemalign.knowledge.candidate_retrieval import CHANNEL_PRIORITY
from biosemalign.schemas.knowledge import CandidateConcept

__all__ = ["channel_agreement", "fuse_candidates", "rank_key"]


def rank_key(candidate: CandidateConcept) -> tuple[int, int, float, str]:
    """Sort key: best channel first, then active concepts, then score.

    Status ranks above score deliberately. A high-scoring obsolete concept is
    worse than a lower-scoring current one for every analytical purpose this
    framework serves.
    """
    best_channel = min(
        (CHANNEL_PRIORITY.get(c, 99) for c in candidate.channels),
        default=99,
    )
    status_rank = 0 if candidate.status is ConceptStatus.ACTIVE else 1
    return (best_channel, status_rank, -candidate.best_score, candidate.candidate_id)


def fuse_candidates(candidates: list[CandidateConcept], *, maximum: int) -> list[CandidateConcept]:
    """Deduplicate by identifier, merge evidence, and rank."""
    merged: dict[str, CandidateConcept] = {}

    for candidate in candidates:
        existing = merged.get(candidate.candidate_id)
        if existing is None:
            merged[candidate.candidate_id] = candidate.model_copy(deep=True)
            continue

        existing.retrieval_evidence.extend(candidate.retrieval_evidence)
        for vocabulary in candidate.source_vocabularies:
            if vocabulary not in existing.source_vocabularies:
                existing.source_vocabularies.append(vocabulary)
        for synonym in candidate.synonyms:
            if synonym not in existing.synonyms:
                existing.synonyms.append(synonym)
        # A definite status beats UNKNOWN; otherwise keep what we had.
        if existing.status is ConceptStatus.UNKNOWN:
            existing.status = candidate.status

    ranked = sorted(merged.values(), key=rank_key)
    return ranked[:maximum]


def channel_agreement(candidate: CandidateConcept) -> int:
    """How many distinct channels independently surfaced this candidate.

    The historical channel does not count toward agreement: it only ever runs
    when every other channel came back empty, so its presence is evidence of
    difficulty rather than of corroboration.
    """
    return len(candidate.channels - {RetrievalChannel.TERMINOLOGY_APPROXIMATE_HISTORICAL})
