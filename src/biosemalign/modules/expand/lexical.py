"""Lexical-neighbors expander operator (`lexical_neighbors`, §5.4, WP6)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from biosemalign.enums import ConceptStatus, RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.providers.lexicon import LexiconProvider
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import OpContext
from biosemalign.modules.registry import register
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet, ExpansionStep
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["LexicalNeighborsExpander", "lexical_neighbors_factory"]


class _LexicalNeighborsOptions(BioSemAlignModel):
    method: Literal["bm25", "trigram"] = "trigram"
    top_k: int = Field(default=10, ge=1, le=200)
    vocabularies: list[str] | None = None


class LexicalNeighborsExpander:
    """Gather lexically close candidate concepts around matched anchors and mention text."""

    op_id = "lexical_neighbors"
    version = "0.1.0"
    uses_model = False

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        method: Literal["bm25", "trigram"] = "trigram",
        top_k: int = 10,
        vocabularies: Sequence[str] | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.method = method
        self.top_k = top_k
        self.vocabularies = list(vocabularies) if vocabularies is not None else None

    @property
    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"method": self.method, "top_k": self.top_k}
        if self.vocabularies is not None:
            opts["vocabularies"] = list(self.vocabularies)
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
        if store is None:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        vocabs = list(self.vocabularies or [])
        if not vocabs:
            vocabs = [
                name
                for name in self.profile.knowledge.resources
                if self.knowledge.serves_locally(name)
            ]
        if not vocabs:
            vocabs = [sau.task.target_terminology]

        method: Literal["bm25", "trigram"] = self.method
        if not store.supports(method):
            fallback: Literal["bm25", "trigram"] = "bm25" if method == "trigram" else "trigram"
            if not store.supports(fallback):
                return ExpansionSet(expander=self.op_id, anchors=anchor_codes)
            method = fallback

        providers: list[LexiconProvider] = []
        for vocab in vocabs:
            provider = self.knowledge.provider_for(vocab)
            if isinstance(provider, LexiconProvider):
                providers.append(provider)
        if not providers:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        queries: list[tuple[QualifiedCode, str]] = []
        mention_query = sau.mention.retrieval_text or sau.mention.text
        default_anchor = (
            anchor_codes[0]
            if anchor_codes
            else QualifiedCode(vocabulary=providers[0].vocabulary, concept_code=sau.mention.text)
        )
        queries.append((default_anchor, mention_query))
        seen_queries = {mention_query.casefold()}
        for anchor, qc in zip(anchors, anchor_codes, strict=True):
            if anchor.preferred_label.casefold() not in seen_queries:
                seen_queries.add(anchor.preferred_label.casefold())
                queries.append((qc, anchor.preferred_label))

        seen_keys = {(c.vocabulary.casefold(), c.candidate_id) for c in anchors}
        limit = self.profile.limits.members
        members: list[ExpansionMember] = []
        truncated = False
        predicate = f"lexical_{method}"

        for anchor_qc, query in queries:
            hits = store.bm25(query, [p.vocabulary for p in providers], self.top_k, kind=method)
            for provider in providers:
                for cand in provider.candidates(
                    hits, channel=RetrievalChannel.LEXICAL, query=query
                ):
                    key = (cand.vocabulary.casefold(), cand.candidate_id)
                    if key in seen_keys:
                        continue
                    if (
                        self.profile.target.require_active_concept
                        and cand.status is not ConceptStatus.ACTIVE
                    ):
                        continue
                    if len(members) >= limit:
                        truncated = True
                        return ExpansionSet(
                            expander=self.op_id,
                            anchors=anchor_codes,
                            members=members,
                            truncated=truncated,
                        )
                    seen_keys.add(key)
                    to_code = QualifiedCode(
                        vocabulary=cand.vocabulary,
                        concept_code=cand.candidate_id,
                        label=cand.preferred_label,
                    )
                    members.append(
                        ExpansionMember(
                            concept=cand,
                            anchor=anchor_qc,
                            hops=1,
                            path=[
                                ExpansionStep(
                                    from_code=anchor_qc,
                                    predicate=predicate,
                                    direction="lexical",
                                    to_code=to_code,
                                )
                            ],
                            via_operator=self.op_id,
                        )
                    )

        return ExpansionSet(
            expander=self.op_id,
            anchors=anchor_codes,
            members=members,
            truncated=truncated,
        )


def lexical_neighbors_factory(
    context: OpContext, options: Mapping[str, Any]
) -> LexicalNeighborsExpander:
    if context.knowledge is None:
        raise ConfigurationError(
            "operator 'lexical_neighbors' needs the pipeline's knowledge router"
        )
    parsed = _LexicalNeighborsOptions.model_validate(options)
    return LexicalNeighborsExpander(
        context.knowledge,
        context.profile,
        method=parsed.method,
        top_k=parsed.top_k,
        vocabularies=parsed.vocabularies,
    )


register("expand", "lexical_neighbors", lexical_neighbors_factory)
