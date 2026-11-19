"""Merge and rank candidates from multiple retrieval channels (§6.3).

Fusion never discards retrieval evidence. Agreement between independent
channels is a routing feature (§9.1), and collapsing three channels into one
"best score" would throw away the signal that makes automatic acceptance
defensible.
"""

from __future__ import annotations

from collections.abc import Sequence

from biosemalign.enums import ConceptStatus, RetrievalChannel
from biosemalign.knowledge.candidate_retrieval import CHANNEL_PRIORITY
from biosemalign.schemas.knowledge import CandidateConcept

__all__ = ["channel_agreement", "fuse_candidates", "fuse_ranked", "rank_key"]

#: Reciprocal rank fusion constant (Cormack, Clarke & Büttcher, 2009).
RRF_K = 60


def rank_key(candidate: CandidateConcept) -> tuple[int, int, float, str, str]:
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
    return (
        best_channel,
        status_rank,
        -candidate.best_score,
        candidate.vocabulary.casefold(),
        candidate.candidate_id,
    )


def fuse_candidates(
    candidates: list[CandidateConcept],
    *,
    maximum: int,
    target_vocabulary: str | None = None,
) -> list[CandidateConcept]:
    """Deduplicate by vocabulary-qualified identifier, merge evidence, and rank.

    Codes are only unique within a vocabulary, so identity is the pair. With
    a target vocabulary, its candidates rank ahead of every other
    vocabulary's: those are concordance evidence, and must never crowd the
    selectable concepts out of a bounded package.
    """
    merged: dict[tuple[str, str], CandidateConcept] = {}
    for candidate in candidates:
        _merge(merged, candidate)

    target = target_vocabulary.casefold() if target_vocabulary else None
    ranked = sorted(
        merged.values(),
        key=lambda c: (target is not None and c.vocabulary.casefold() != target, rank_key(c)),
    )
    return ranked[:maximum]


def fuse_ranked(
    ranked_lists: Sequence[Sequence[CandidateConcept]],
    *,
    maximum: int,
    target_vocabulary: str | None = None,
    k: int = RRF_K,
) -> list[CandidateConcept]:
    """Merge ranked lists by reciprocal rank fusion.

    Lexical and embedding scores are on incomparable scales; RRF uses only each
    list's ranks, so a candidate near the top of several lists outranks one at
    the top of a single list. Target-vocabulary candidates still rank first.
    """
    merged: dict[tuple[str, str], CandidateConcept] = {}
    scores: dict[tuple[str, str], float] = {}
    for candidates in ranked_lists:
        seen: set[tuple[str, str]] = set()
        for rank, candidate in enumerate(candidates, start=1):
            key = _merge(merged, candidate)
            if key not in seen:
                seen.add(key)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    target = target_vocabulary.casefold() if target_vocabulary else None
    ordered = sorted(
        merged,
        key=lambda key: (target is not None and key[0] != target, -scores[key], key),
    )
    return [merged[key] for key in ordered[:maximum]]


def _merge(
    merged: dict[tuple[str, str], CandidateConcept], candidate: CandidateConcept
) -> tuple[str, str]:
    """Add a candidate, merging evidence into an existing one of the same identity."""
    key = (candidate.vocabulary.casefold(), candidate.candidate_id)
    existing = merged.get(key)
    if existing is None:
        merged[key] = candidate.model_copy(deep=True)
        return key
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
    return key


def channel_agreement(candidate: CandidateConcept) -> int:
    """How many distinct channels independently surfaced this candidate.

    The historical channel does not count toward agreement: it only ever runs
    when every other channel came back empty, so its presence is evidence of
    difficulty rather than of corroboration.
    """
    return len(candidate.channels - {RetrievalChannel.TERMINOLOGY_APPROXIMATE_HISTORICAL})
