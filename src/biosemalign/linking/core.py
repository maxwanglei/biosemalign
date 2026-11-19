"""Where a code leads at a granularity: the one definition (v0.3).

Projection (the deterministic adjudicator), resolution (the concordant
resolver), and link derivation (the kernel) all ask the same question — what
does this target-family concept reach at the requested level? — and get the
answer from :func:`targets_at`, under the profile's target spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from biosemalign.enums import LinkKind
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage, KnowledgeStatement
from biosemalign.schemas.links import CodeLink, LinkSet
from biosemalign.targets.spec import TargetSpec

__all__ = ["Reached", "derive_link_set", "targets_at"]


@dataclass(frozen=True, slots=True)
class Reached:
    """A concept at the requested level, and the edge that led there (None: itself)."""

    vocabulary: str
    concept_code: str
    label: str | None
    concept_class: str | None
    statement: KnowledgeStatement | None = None


def targets_at(
    candidate: CandidateConcept, spec: TargetSpec, granularity: str | None
) -> list[Reached]:
    """Concepts ``candidate`` reaches at ``granularity``.

    A candidate already at the level is its own target — "metoprolol" maps to
    itself. Otherwise the spec's relationships for the level are followed, one
    hop, deduplicated by target, in the candidate's edge order.
    """
    if spec.is_terminal(candidate.concept_class, granularity):
        return [
            Reached(
                candidate.vocabulary,
                candidate.candidate_id,
                candidate.preferred_label,
                candidate.concept_class,
            )
        ]
    level = spec.granularity(granularity)
    if level is None:
        return []
    via = set(level.via)
    reached: dict[tuple[str, str], Reached] = {}
    for statement in candidate.relationships:
        if statement.predicate not in via:
            continue
        vocabulary = statement.object_vocabulary or candidate.vocabulary
        key = (vocabulary, statement.object_id)
        if key not in reached:
            target_class = statement.extensions.get("target_tty")
            reached[key] = Reached(
                vocabulary,
                statement.object_id,
                statement.object_label,
                str(target_class) if target_class is not None else None,
                statement,
            )
    return list(reached.values())


def derive_link_set(
    package: KnowledgePackage, spec: TargetSpec, granularity: str | None
) -> LinkSet | None:
    """The typed links of a package's exact matches, or None without concordance.

    Every member of an equivalence group links through each target-family
    concept of its group that the package holds as a candidate: as the same
    entity when that concept is itself at the requested level, by projection
    when the spec's relationship leads there.
    """
    if package.concordance is None:
        return None
    links: list[CodeLink] = []
    ambiguous: list[QualifiedCode] = []
    for group in package.concordance.groups:
        signatures: dict[tuple[str, str], set[frozenset[tuple[str, str]]]] = {}
        for via_code in group.members:
            if not spec.in_family(via_code.vocabulary):
                continue
            via = package.candidate_by_qualified_id(via_code.vocabulary, via_code.concept_code)
            if via is None:
                continue
            reached = targets_at(via, spec, granularity)
            if not reached:
                continue
            signature = frozenset((r.vocabulary, r.concept_code) for r in reached)
            for source in group.members:
                signatures.setdefault((source.vocabulary, source.concept_code), set()).add(
                    signature
                )
                links.extend(_links(source, via_code, reached))
        ambiguous.extend(
            member
            for member in group.members
            if len(signatures.get((member.vocabulary, member.concept_code), set())) > 1
        )
    return LinkSet(
        target_spec=spec.label,
        target_spec_sha256=spec.digest,
        granularity=granularity,
        links=links,
        ambiguous=[code.model_copy() for code in ambiguous],
    )


def _links(source: QualifiedCode, via: QualifiedCode, reached: list[Reached]) -> list[CodeLink]:
    links: list[CodeLink] = []
    for target in reached:
        statement = target.statement
        sources = (
            [str(statement.extensions["assertion_source"])]
            if statement is not None and statement.extensions.get("assertion_source")
            else []
        )
        links.append(
            CodeLink(
                source=QualifiedCode(
                    vocabulary=source.vocabulary,
                    concept_code=source.concept_code,
                    label=source.label,
                ),
                via=QualifiedCode(
                    vocabulary=via.vocabulary, concept_code=via.concept_code, label=via.label
                ),
                target=QualifiedCode(
                    vocabulary=target.vocabulary,
                    concept_code=target.concept_code,
                    label=target.label,
                ),
                kind=LinkKind.EQUIVALENCE if statement is None else LinkKind.PROJECTION,
                predicates=[statement.predicate] if statement is not None else [],
                assertion_sources=sources,
            )
        )
    return links
