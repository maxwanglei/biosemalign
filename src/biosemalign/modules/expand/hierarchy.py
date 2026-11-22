"""Hierarchy expander operator (`hierarchy`, §5.4, §5.6, WP6)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from biosemalign.enums import ConceptStatus, RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.providers.lexicon import LexiconEdge, LexiconProvider, LexiconStore
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import OpContext
from biosemalign.modules.registry import register
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet, ExpansionStep
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["HierarchyExpander", "hierarchy_expander_factory"]


class _HierarchyOptions(BioSemAlignModel):
    max_depth: int = Field(default=8, ge=1, le=32)
    include_ancestors: bool = False
    include_siblings: bool = False
    level: str | None = None
    classes: list[str] | None = None


class HierarchyExpander:
    """Walk terminology relationships downward (and optionally upward/siblings) from anchors."""

    op_id = "hierarchy"
    version = "0.1.0"
    uses_model = False

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        max_depth: int = 8,
        include_ancestors: bool = False,
        include_siblings: bool = False,
        level: str | None = None,
        classes: Sequence[str] | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.spec = profile.target_spec()
        self.max_depth = max_depth
        self.include_ancestors = include_ancestors
        self.include_siblings = include_siblings
        self.level = level
        self.classes = list(classes) if classes is not None else None

    @property
    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "max_depth": self.max_depth,
            "include_ancestors": self.include_ancestors,
            "include_siblings": self.include_siblings,
        }
        if self.level is not None:
            opts["level"] = self.level
        if self.classes is not None:
            opts["classes"] = list(self.classes)
        return opts

    def expand(
        self, sau: SemanticAlignmentUnit, anchors: Sequence[CandidateConcept]
    ) -> ExpansionSet:
        anchor_codes = [
            QualifiedCode(
                vocabulary=c.vocabulary,
                concept_code=c.candidate_id,
                label=c.preferred_label,
            )
            for c in anchors
        ]
        store = self.knowledge.lexicon
        if store is None or not anchors:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        effective_level = self.level if self.level is not None else sau.task.desired_granularity
        gran = self.spec.granularity(effective_level)
        walk_predicates = _permitted_predicates(
            [*self.spec.subsumption, *(gran.via if gran is not None else ())],
            self.profile,
        )
        subsumption_predicates = _permitted_predicates(self.spec.subsumption, self.profile)
        if not walk_predicates:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        limit = self.profile.limits.members
        allowed_classes: set[str] | None = (
            {self.spec.canonical_class(c) or c for c in self.classes}
            if self.classes is not None
            else None
        )
        anchor_keys = {(c.vocabulary.casefold(), c.candidate_id) for c in anchors}

        members: list[ExpansionMember] = []
        seen_keys: set[tuple[str, str]] = set(anchor_keys)
        truncated = False
        query = sau.mention.retrieval_text or sau.mention.text

        # 1. Downward traversal (incoming edges along subsumption + level.via).
        truncated = self._bfs_walk(
            store=store,
            anchors=anchors,
            anchor_codes=anchor_codes,
            predicates=walk_predicates,
            direction="incoming",
            query=query,
            limit=limit,
            allowed_classes=allowed_classes,
            seen_keys=seen_keys,
            out_members=members,
        )

        # 2. Optional ancestor walk (outgoing edges along subsumption + level.via).
        if not truncated and self.include_ancestors:
            truncated = self._bfs_walk(
                store=store,
                anchors=anchors,
                anchor_codes=anchor_codes,
                predicates=walk_predicates,
                direction="outgoing",
                query=query,
                limit=limit,
                allowed_classes=allowed_classes,
                seen_keys=seen_keys,
                out_members=members,
            )

        # 3. Optional sibling walk (1 hop out along subsumption -> 1 hop back in).
        if not truncated and self.include_siblings and subsumption_predicates:
            truncated = self._sibling_walk(
                store=store,
                anchors=anchors,
                anchor_codes=anchor_codes,
                subsumption=subsumption_predicates,
                query=query,
                limit=limit,
                allowed_classes=allowed_classes,
                seen_keys=seen_keys,
                out_members=members,
            )

        return ExpansionSet(
            expander=self.op_id,
            anchors=anchor_codes,
            members=members,
            truncated=truncated,
        )

    def _bfs_walk(
        self,
        *,
        store: LexiconStore,
        anchors: Sequence[CandidateConcept],
        anchor_codes: Sequence[QualifiedCode],
        predicates: list[str],
        direction: str,
        query: str,
        limit: int,
        allowed_classes: set[str] | None,
        seen_keys: set[tuple[str, str]],
        out_members: list[ExpansionMember],
    ) -> bool:
        # Queue elements: (root_anchor_code, current_code, depth, path_steps)
        queue: deque[tuple[QualifiedCode, QualifiedCode, int, list[ExpansionStep]]] = deque(
            (ac, ac, 0, []) for ac, _ in zip(anchor_codes, anchors, strict=True)
        )
        visited_bfs: set[tuple[str, str]] = {
            (c.vocabulary.casefold(), c.candidate_id) for c in anchors
        }

        while queue:
            root_anchor, current, depth, path = queue.popleft()
            if depth >= self.max_depth:
                continue
            if direction == "incoming":
                edges = store.incoming_relationships(
                    current.vocabulary, [current.concept_code], predicates
                )
            else:
                edges = store.relationships(current.vocabulary, [current.concept_code], predicates)

            next_by_vocab: dict[str, list[tuple[str, LexiconEdge]]] = {}
            for edge in edges:
                next_vocab = (
                    (edge.subject_vocabulary or current.vocabulary)
                    if direction == "incoming"
                    else edge.object_vocabulary
                )
                next_code = edge.subject if direction == "incoming" else edge.object_code
                next_key = (next_vocab.casefold(), next_code)
                if next_key in visited_bfs:
                    continue
                visited_bfs.add(next_key)
                next_by_vocab.setdefault(next_vocab, []).append((next_code, edge))

            for next_vocab, pairs in next_by_vocab.items():
                provider = self.knowledge.provider_for(next_vocab)
                if not isinstance(provider, LexiconProvider):
                    continue
                ordered_codes = [code for code, _ in pairs]
                edge_by_code = dict(pairs)
                candidates = provider.candidates_for_codes(
                    ordered_codes,
                    channel=RetrievalChannel.ONTOLOGY_NEIGHBORHOOD,
                    query=query,
                    provider_name=self.op_id,
                )
                cand_by_code = {c.candidate_id: c for c in candidates}
                for next_code in ordered_codes:
                    cand = cand_by_code.get(next_code)
                    if cand is None:
                        continue
                    edge = edge_by_code[next_code]
                    next_qc = QualifiedCode(
                        vocabulary=cand.vocabulary,
                        concept_code=cand.candidate_id,
                        label=cand.preferred_label,
                    )
                    step = ExpansionStep(
                        from_code=current,
                        predicate=edge.predicate,
                        direction="incoming" if direction == "incoming" else "outgoing",
                        to_code=next_qc,
                    )
                    new_path = [*path, step]
                    queue.append((root_anchor, next_qc, depth + 1, new_path))

                    cand_key = (cand.vocabulary.casefold(), cand.candidate_id)
                    if cand_key in seen_keys:
                        continue
                    if (
                        self.profile.target.require_active_concept
                        and cand.status is not ConceptStatus.ACTIVE
                    ):
                        continue
                    if (
                        allowed_classes is not None
                        and self.spec.canonical_class(cand.concept_class) not in allowed_classes
                    ):
                        continue
                    if len(out_members) >= limit:
                        return True
                    seen_keys.add(cand_key)
                    out_members.append(
                        ExpansionMember(
                            concept=cand,
                            anchor=root_anchor,
                            hops=depth + 1,
                            path=new_path,
                            via_operator=self.op_id,
                        )
                    )
        return False

    def _sibling_walk(
        self,
        *,
        store: LexiconStore,
        anchors: Sequence[CandidateConcept],
        anchor_codes: Sequence[QualifiedCode],
        subsumption: list[str],
        query: str,
        limit: int,
        allowed_classes: set[str] | None,
        seen_keys: set[tuple[str, str]],
        out_members: list[ExpansionMember],
    ) -> bool:
        for root_anchor, anchor in zip(anchor_codes, anchors, strict=True):
            parent_edges = store.relationships(
                anchor.vocabulary, [anchor.candidate_id], subsumption
            )
            for p_edge in parent_edges:
                p_vocab = p_edge.object_vocabulary
                p_code = p_edge.object_code
                parent_row = store.concepts(p_vocab, [p_code]).get(p_code)
                parent_qc = QualifiedCode(
                    vocabulary=p_vocab,
                    concept_code=p_code,
                    label=parent_row.concept_name if parent_row else None,
                )
                up_step = ExpansionStep(
                    from_code=root_anchor,
                    predicate=p_edge.predicate,
                    direction="outgoing",
                    to_code=parent_qc,
                )
                child_edges = store.incoming_relationships(p_vocab, [p_code], subsumption)
                by_vocab: dict[str, list[tuple[str, LexiconEdge]]] = {}
                for c_edge in child_edges:
                    c_vocab = c_edge.subject_vocabulary or p_vocab
                    if (c_vocab.casefold(), c_edge.subject) in seen_keys:
                        continue
                    by_vocab.setdefault(c_vocab, []).append((c_edge.subject, c_edge))
                for c_vocab, pairs in by_vocab.items():
                    provider = self.knowledge.provider_for(c_vocab)
                    if not isinstance(provider, LexiconProvider):
                        continue
                    ordered_codes = [code for code, _ in pairs]
                    c_edge_by_code = dict(pairs)
                    cands = {
                        c.candidate_id: c
                        for c in provider.candidates_for_codes(
                            ordered_codes,
                            channel=RetrievalChannel.ONTOLOGY_NEIGHBORHOOD,
                            query=query,
                            provider_name=self.op_id,
                        )
                    }
                    for code in ordered_codes:
                        cand = cands.get(code)
                        if cand is None:
                            continue
                        key = (cand.vocabulary.casefold(), cand.candidate_id)
                        if key in seen_keys:
                            continue
                        if (
                            self.profile.target.require_active_concept
                            and cand.status is not ConceptStatus.ACTIVE
                        ):
                            continue
                        if (
                            allowed_classes is not None
                            and self.spec.canonical_class(cand.concept_class) not in allowed_classes
                        ):
                            continue
                        if len(out_members) >= limit:
                            return True
                        seen_keys.add(key)
                        cand_qc = QualifiedCode(
                            vocabulary=cand.vocabulary,
                            concept_code=cand.candidate_id,
                            label=cand.preferred_label,
                        )
                        down_step = ExpansionStep(
                            from_code=parent_qc,
                            predicate=c_edge_by_code[code].predicate,
                            direction="incoming",
                            to_code=cand_qc,
                        )
                        out_members.append(
                            ExpansionMember(
                                concept=cand,
                                anchor=root_anchor,
                                hops=2,
                                path=[up_step, down_step],
                                via_operator=self.op_id,
                            )
                        )
        return False


def _permitted_predicates(predicates: Sequence[str], profile: TaskProfile) -> list[str]:
    ordered: list[str] = []
    for predicate in predicates:
        if predicate not in ordered and profile.knowledge.permits(predicate):
            ordered.append(predicate)
    return ordered


def hierarchy_expander_factory(context: OpContext, options: Mapping[str, Any]) -> HierarchyExpander:
    if context.knowledge is None:
        raise ConfigurationError(
            "operator 'hierarchy' expander needs the pipeline's knowledge router"
        )
    parsed = _HierarchyOptions.model_validate(options)
    return HierarchyExpander(
        context.knowledge,
        context.profile,
        max_depth=parsed.max_depth,
        include_ancestors=parsed.include_ancestors,
        include_siblings=parsed.include_siblings,
        level=parsed.level,
        classes=parsed.classes,
    )


register("expand", "hierarchy", hierarchy_expander_factory)
