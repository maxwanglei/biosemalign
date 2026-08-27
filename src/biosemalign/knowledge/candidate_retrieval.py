"""Multi-channel candidate retrieval (§6.3).

Channels run in order of decreasing trust and stop early once a good answer is
in hand. The escalation matters: exact lookup fails outright on FAERS-shaped
strings like "TOPROL XL 50 MG" (RxNav returns an empty ``idGroup``), so
approximate matching is load-bearing rather than a fallback nicety.

The historical channel is deliberately last and conditional. RxNav's
approximate matcher without ``option=1`` happily returns obsolete atoms from
non-RxNorm source vocabularies — "TOPROL XL 50 MG" resolves to a ``NotCurrent``
MMSL concept literally named "PROPRIETARY" — which cannot be enriched and must
never be mistaken for a current concept.
"""

from __future__ import annotations

from biosemalign.enums import RetrievalChannel
from biosemalign.knowledge.providers.rxnorm import RxNavProvider
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept, RetrievalEvidence
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import lexical_similarity, stable_digest

__all__ = ["CHANNEL_PRIORITY", "RETRIEVER_VERSION", "CandidateRetriever"]

RETRIEVER_VERSION = "0.1.0"

log = get_logger(__name__)

#: Lower is better. Used as the primary ranking key because channel scores are
#: not on a common scale — an exact identifier match and an RxNav approximate
#: score cannot be meaningfully compared as numbers.
CHANNEL_PRIORITY: dict[RetrievalChannel, int] = {
    RetrievalChannel.EXACT_LOOKUP: 0,
    RetrievalChannel.NORMALIZED_LOOKUP: 1,
    RetrievalChannel.TERMINOLOGY_APPROXIMATE_CURRENT: 2,
    RetrievalChannel.SPELLING_SUGGESTION: 3,
    RetrievalChannel.TERMINOLOGY_APPROXIMATE_HISTORICAL: 4,
}


class CandidateRetriever:
    """Runs the retrieval channels a profile enables and returns raw candidates."""

    version = RETRIEVER_VERSION

    def __init__(self, provider: RxNavProvider, profile: TaskProfile) -> None:
        self.provider = provider
        self.profile = profile

    def retrieve(self, sau: SemanticAlignmentUnit) -> list[CandidateConcept]:
        """Retrieve candidates for one SAU, in channel order."""
        config = self.profile.retrieval
        term = sau.mention.text
        collected: list[CandidateConcept] = []

        if config.exact_lookup:
            collected.extend(self._exact(term))

        if config.terminology_approximate_match and (
            not collected or self._needs_corroboration(term, collected)
        ):
            approximate = self._approximate(term, current_only=True)
            collected.extend(approximate)

            # Exact/normalized lookups are a deterministic fast path. Fuzzy
            # results that do not clear the auto-accept lexical floor receive
            # an independent spelling channel. FROZEN executes this same plan
            # from its release-bound snapshot; a missing response remains a
            # hard cache miss and can never fall through to the network.
            if approximate and self._needs_corroboration(term, approximate):
                collected.extend(self._spelling(term))

        if not collected:
            collected.extend(self._spelling(term))

        # Last resort only. A historical hit is better than no candidate at all
        # for a withdrawn product named in an old report, but it is flagged so
        # that routing can refuse to auto-accept it.
        if config.include_historical_candidates and not collected:
            collected.extend(self._approximate(term, current_only=False))

        log.debug(
            "retrieval.completed",
            sau_id=sau.sau_id,
            term_digest=stable_digest(term, length=12),
            candidates=len(collected),
            channels=sorted({c.value for cand in collected for c in cand.channels}),
        )
        return collected

    def _needs_corroboration(self, term: str, candidates: list[CandidateConcept]) -> bool:
        # OFFLINE is a compatibility mode for older partial cassettes. FROZEN
        # is the reproducible production mode and must run the identical
        # adaptive plan used online, including surfacing a missing cache entry.
        if self.provider.mode.value == "OFFLINE" or not candidates:
            return False
        top = candidates[0]
        similarity = max(
            [lexical_similarity(term, top.preferred_label)]
            + [lexical_similarity(term, synonym) for synonym in top.synonyms]
        )
        return similarity < self.profile.routing.minimum_lexical_similarity

    # -- channels ----------------------------------------------------------

    def _exact(self, term: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        for rank, rxcui in enumerate(self.provider.find_by_name(term), start=1):
            concept = self.provider.lookup(rxcui)
            if concept is None:
                continue
            concept.retrieval_evidence.append(
                RetrievalEvidence(
                    channel=RetrievalChannel.NORMALIZED_LOOKUP,
                    query=term,
                    provider=self.provider.provider_name,
                    score=100.0,
                    rank=rank,
                    source_vocabulary="RXNORM",
                )
            )
            candidates.append(concept)
        return candidates

    def _approximate(self, term: str, *, current_only: bool) -> list[CandidateConcept]:
        channel = (
            RetrievalChannel.TERMINOLOGY_APPROXIMATE_CURRENT
            if current_only
            else RetrievalChannel.TERMINOLOGY_APPROXIMATE_HISTORICAL
        )
        max_entries = self.profile.retrieval.maximum_candidates

        hits = self.provider.approximate(term, max_entries=max_entries, current_only=current_only)

        candidates: list[CandidateConcept] = []
        seen: set[str] = set()
        for hit in hits:
            rxcui = str(hit["rxcui"])
            if rxcui in seen:
                # RxNav reports one RxCUI once per matching atom; collapse to
                # the concept and let fusion merge the evidence.
                continue
            seen.add(rxcui)
            concept = self.provider.lookup(rxcui)
            if concept is None:
                continue
            concept.retrieval_evidence.append(
                RetrievalEvidence(
                    channel=channel,
                    query=term,
                    provider=self.provider.provider_name,
                    score=_as_float(hit.get("score")),
                    rank=_as_int(hit.get("rank")),
                    source_atom_id=_as_str(hit.get("rxaui")),
                    source_vocabulary=_as_str(hit.get("source")),
                )
            )
            candidates.append(concept)
            if len(candidates) >= max_entries:
                break
        return candidates

    def _spelling(self, term: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        for suggestion in self.provider.spelling_suggestions(term)[:3]:
            for rank, rxcui in enumerate(self.provider.find_by_name(suggestion), start=1):
                concept = self.provider.lookup(rxcui)
                if concept is None:
                    continue
                concept.retrieval_evidence.append(
                    RetrievalEvidence(
                        channel=RetrievalChannel.SPELLING_SUGGESTION,
                        query=suggestion,
                        provider=self.provider.provider_name,
                        score=90.0,
                        rank=rank,
                        source_vocabulary="RXNORM",
                    )
                )
                candidates.append(concept)
        return candidates


# RxNav returns scores and ranks as JSON strings rather than numbers, so these
# coerce defensively: a malformed field should cost one candidate's score, not
# abort the retrieval.


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)
