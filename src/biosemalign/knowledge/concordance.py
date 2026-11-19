"""Cross-vocabulary concordance among exact matches (v0.2).

A drug name that RxNorm, UNII, and MeSH all list for the same substance is
stronger evidence than a name one dictionary lists alone: single-source names
are where dictionary noise lives. Two things make that intuition safe to
automate.

*Equivalence is explicit.* Matches are grouped only through equivalence keys
the profile permits (a concept's own vocabulary code, or cross-references such
as ``UNII`` or ``UMLS_CUI``). If a group would hold two different codes from
one vocabulary — a salt and its base, merged through an over-broad key — the
grouping is reported as a conflict instead of being trusted. A profile may
narrow that check to the vocabularies whose codes are the answer (say RxNorm
and RxNorm Extension): a SNOMED substance and the SNOMED product named after
it, both mapped to one RxNorm ingredient, are then recorded together without
making the target ambiguous. The listed vocabularies form one pool, so an
RxNorm code and an RxNorm Extension code in one group conflict too.

*Agreement counts sources, not vocabularies.* UMLS redistributes RxNorm and
MeSH; RxNorm carries DrugBank and MeSH atoms. Each matched name carries the
source that asserted it, so a name that reached three vocabularies from one
origin counts once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from biosemalign.enums import ConcordanceState, RetrievalChannel
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    ConcordanceAssessment,
    EquivalenceGroup,
    QualifiedCode,
)

__all__ = ["EXACT_CHANNELS", "KeyLookup", "assess_concordance"]

#: Channels that assert a name match rather than a similarity.
EXACT_CHANNELS = frozenset({RetrievalChannel.EXACT_LOOKUP, RetrievalChannel.NORMALIZED_LOOKUP})

Member = tuple[str, str]
#: Returns each member's equivalence keys as ``(key_type, key_value)`` pairs.
KeyLookup = Callable[[Iterable[Member]], Mapping[Member, set[tuple[str, str]]]]


def assess_concordance(
    candidates: Iterable[CandidateConcept],
    *,
    equivalence_keys: Iterable[str],
    key_lookup: KeyLookup | None = None,
    source_aliases: Mapping[str, str] | None = None,
    conflict_vocabularies: Iterable[str] | None = None,
) -> ConcordanceAssessment:
    """Group exact matches by permitted equivalence keys and classify agreement.

    ``conflict_vocabularies`` limits which duplicate codes make a group a
    conflict; None means any vocabulary's.
    """
    aliases = dict(source_aliases or {})
    allowed = set(equivalence_keys)
    pool = (
        {vocabulary.casefold() for vocabulary in conflict_vocabularies}
        if conflict_vocabularies is not None
        else None
    )

    exact: dict[Member, CandidateConcept] = {}
    sources: dict[Member, set[str]] = {}
    for candidate in candidates:
        evidence = [e for e in candidate.retrieval_evidence if e.channel in EXACT_CHANNELS]
        if not evidence:
            continue
        member = (candidate.vocabulary, candidate.candidate_id)
        exact.setdefault(member, candidate)
        for item in evidence:
            source = item.source_vocabulary or candidate.vocabulary
            sources.setdefault(member, set()).add(aliases.get(source, source))

    if not exact:
        return ConcordanceAssessment(
            state=ConcordanceState.NO_EXACT_MATCH, equivalence_keys=sorted(allowed)
        )

    members = sorted(exact)
    keys = (
        key_lookup(members) if key_lookup is not None else {member: {member} for member in members}
    )
    parent = {member: member for member in members}

    def root(member: Member) -> Member:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    owners: dict[tuple[str, str], Member] = {}
    for member in members:
        for key in sorted(keys.get(member, {member}) | {member}):
            if key[0] not in allowed:
                continue
            other = owners.setdefault(key, member)
            first, second = sorted((root(member), root(other)))
            parent[second] = first

    grouped: dict[Member, list[Member]] = {}
    for member in members:
        grouped.setdefault(root(member), []).append(member)

    groups: list[EquivalenceGroup] = []
    for group_members in sorted(grouped.values()):
        vocabularies = [vocabulary.casefold() for vocabulary, _ in group_members]
        if pool is None:
            conflict = len(vocabularies) != len(set(vocabularies))
        else:
            conflict = sum(vocabulary in pool for vocabulary in vocabularies) > 1
        groups.append(
            EquivalenceGroup(
                members=[
                    QualifiedCode(
                        vocabulary=vocabulary,
                        concept_code=code,
                        label=exact[(vocabulary, code)].preferred_label,
                    )
                    for vocabulary, code in group_members
                ],
                sources=sorted(set().union(*(sources[m] for m in group_members))),
                conflict=conflict,
            )
        )

    if any(group.conflict for group in groups):
        state = ConcordanceState.EQUIVALENCE_CONFLICT
    elif len(groups) > 1:
        state = ConcordanceState.CROSS_VOCAB_COLLISION
    elif len(groups[0].sources) >= 2:
        state = ConcordanceState.MULTI_SOURCE_CONCORDANT
    else:
        state = ConcordanceState.SINGLE_SOURCE_EXACT
    return ConcordanceAssessment(state=state, groups=groups, equivalence_keys=sorted(allowed))
