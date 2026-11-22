"""Where a code leads at a target level: the one definition (§5.4).

Projection (the deterministic resolver), resolution (the concordant
resolver), and link derivation (the kernel) all ask the same question — what
does this target-family concept reach at the requested level? — and get the
answer from :func:`targets_at`, under the profile's target spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from biosemalign.enums import Authority, LinkKind
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    KnowledgePackage,
    KnowledgeStatement,
    statement_object_class,
)
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
    predicates: tuple[str, ...] = ()


def targets_at(
    candidate: CandidateConcept, spec: TargetSpec, granularity: str | None
) -> list[Reached]:
    """Concepts ``candidate`` reaches at ``granularity``.

    When ``granularity`` is ``None`` or ``candidate`` already sits at a
    terminal class for that level, it is its own target. Otherwise the spec's
    relationships are walked up to ``level.max_hops``, stopping at the first
    concept of an accepted target class along each path.
    """
    if granularity is None or spec.is_terminal(candidate.concept_class, granularity):
        return [
            Reached(
                candidate.vocabulary,
                candidate.candidate_id,
                candidate.preferred_label,
                candidate.concept_class,
            )
        ]
    level = spec.granularity(granularity)
    if level is None or not level.via:
        return []
    via = set(level.via)
    accepted = spec.accepted_classes(granularity)
    reached: dict[tuple[str, str], Reached] = {}
    frontier: list[tuple[str, tuple[str, ...]]] = [(candidate.candidate_id, ())]
    visited: set[str] = {candidate.candidate_id}

    for hop in range(1, level.max_hops + 1):
        next_frontier: list[tuple[str, tuple[str, ...]]] = []
        for node_id, path_preds in frontier:
            for statement in candidate.relationships:
                if statement.authority is Authority.LLM_INTERNAL_KNOWLEDGE:
                    continue
                step_subject = str(
                    statement.extensions.get("step_subject_id") or statement.subject_id
                )
                if step_subject != node_id or statement.predicate not in via:
                    continue
                vocabulary = statement.object_vocabulary or candidate.vocabulary
                raw_class = statement_object_class(statement) or spec.inferred_class(
                    statement.predicate
                )
                canon_class = spec.canonical_class(raw_class)
                # Preserve existing 1-hop fallback when a fixture edge omits target class.
                is_target = (
                    (canon_class in accepted) if canon_class is not None else (level.max_hops == 1)
                )
                full_preds = (*path_preds, statement.predicate)
                if is_target:
                    key = (vocabulary, statement.object_id)
                    if key not in reached:
                        reached[key] = Reached(
                            vocabulary,
                            statement.object_id,
                            statement.object_label,
                            raw_class,
                            statement,
                            full_preds,
                        )
                elif hop < level.max_hops and statement.object_id not in visited:
                    visited.add(statement.object_id)
                    next_frontier.append((statement.object_id, full_preds))
        frontier = next_frontier
        if not frontier:
            break
    return list(reached.values())


def derive_link_set(
    package: KnowledgePackage, spec: TargetSpec, granularity: str | None
) -> LinkSet:
    """The typed links of a package's matched codes and candidates.

    Every member of an equivalence group links through each target-family
    concept of its group that the package holds as a candidate. Any target-family
    candidate not already covered via a concordance group is also linked
    directly from itself so every mapping package carries typed target links.
    """
    links: list[CodeLink] = []
    ambiguous: list[QualifiedCode] = []
    linked_vias: set[tuple[str, str]] = set()
    if package.concordance is not None:
        for group in package.concordance.groups:
            signatures: dict[tuple[str, str], set[frozenset[tuple[str, str]]]] = {}
            for via_code in group.members:
                if not spec.in_family(via_code.vocabulary):
                    continue
                via = package.candidate_by_qualified_id(via_code.vocabulary, via_code.concept_code)
                if via is None:
                    continue
                linked_vias.add((via.vocabulary.casefold(), via.candidate_id))
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

    for candidate in package.candidates:
        if not spec.in_family(candidate.vocabulary):
            continue
        key = (candidate.vocabulary.casefold(), candidate.candidate_id)
        if key in linked_vias:
            continue
        reached = targets_at(candidate, spec, granularity)
        if not reached:
            continue
        linked_vias.add(key)
        code = QualifiedCode(
            vocabulary=candidate.vocabulary,
            concept_code=candidate.candidate_id,
            label=candidate.preferred_label,
        )
        links.extend(_links(code, code, reached))

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
                predicates=(
                    list(target.predicates or ((statement.predicate,) if statement else ()))
                ),
                assertion_sources=sources,
            )
        )
    return links
