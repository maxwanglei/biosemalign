"""Lexicon and concordance operators (v0.2).

``lexicon`` retrieves from a local bundle by exact normalized name, by BM25
over tokenized names, or exact-then-BM25. ``cross_vocab_concordance`` fuses as
the default fuser does and attaches a concordance assessment over every exact
match. ``concordant_exact`` turns agreement into a proposal — handing the
agreed identity to the deterministic projection rules — and passes everything
it should not decide: collisions, conflicts, short acronyms, and names only
one non-authoritative source asserts.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Literal

from pydantic import Field

from biosemalign.adjudication.deterministic import DeterministicAdjudicator
from biosemalign.enums import ConcordanceState, RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.concordance import assess_concordance
from biosemalign.knowledge.lexicon.embeddings import (
    EmbeddingIndex,
    HashingEncoder,
    SentenceTransformerEncoder,
    TextEncoder,
)
from biosemalign.knowledge.providers.lexicon import LexiconHit, LexiconProvider, LexiconStore
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.ops.base import CandidateSet, OpContext, PassToNext
from biosemalign.ops.builtin import KnowledgePackageFuseOp
from biosemalign.ops.registry import register_op
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["ConcordanceFuseOp", "ConcordantExactProposeOp", "DenseRetrieveOp", "LexiconRetrieveOp"]

RetrievalMethod = Literal["exact", "bm25", "trigram"]
_DEFAULT_METHODS: tuple[RetrievalMethod, ...] = ("exact", "trigram")

_ISO_DATE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})")


# -- retrieval ---------------------------------------------------------------


class _LexiconOptions(BioSemAlignModel):
    vocabularies: list[str] | None = Field(
        default=None,
        description="Vocabularies to query; default every profile resource the bundle serves.",
    )
    methods: list[RetrievalMethod] = Field(
        default_factory=lambda: list(_DEFAULT_METHODS),
        min_length=1,
        description="Tried in order until one returns candidates.",
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
    respect_validity_dates: bool = Field(
        default=False,
        description="Drop concepts not valid on the record's date (context.date), when given.",
    )


class LexiconRetrieveOp:
    """Exact and ranked retrieval across the vocabularies of a local bundle.

    Methods run in order and stop at the first that finds anything, the way
    RxNav's channels escalate: exact names first, then BM25 over words or
    character trigrams for partial and misspelled names.
    """

    op_id = "lexicon"
    version = "0.1.0"
    requires_network = False

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        vocabularies: Sequence[str] | None = None,
        methods: Sequence[RetrievalMethod] = _DEFAULT_METHODS,
        top_k: int | None = None,
        respect_validity_dates: bool = False,
    ) -> None:
        if knowledge.lexicon is None:
            raise ConfigurationError(
                "the 'lexicon' retriever needs a lexicon bundle; set BIOSEMALIGN_LEXICON_PATH"
            )
        names = list(vocabularies or [])
        if not names:
            declared = [
                *profile.knowledge.primary_resources,
                *profile.knowledge.secondary_resources,
            ]
            names = [name for name in declared if knowledge.serves_locally(name)]
        if not names:
            raise ConfigurationError(
                "the 'lexicon' retriever has no vocabulary to query: none of the profile's "
                "resources is in the lexicon bundle"
            )
        providers: list[LexiconProvider] = []
        for name in names:
            provider = knowledge.provider_for(name)
            if not isinstance(provider, LexiconProvider):
                raise ConfigurationError(f"vocabulary {name!r} is not served by the lexicon bundle")
            providers.append(provider)
        unsupported = [m for m in methods if not knowledge.lexicon.supports(m)]
        if unsupported:
            raise ConfigurationError(
                f"the lexicon bundle has no index for {unsupported}; it supports "
                f"{knowledge.lexicon.manifest.indexes}"
            )
        self.store: LexiconStore = knowledge.lexicon
        self.providers = providers
        self.methods = tuple(dict.fromkeys(methods))
        self.top_k = top_k or profile.retrieval.maximum_candidates
        self.respect_validity_dates = respect_validity_dates

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(provider.vocabulary for provider in self.providers)

    @property
    def options(self) -> dict[str, Any]:
        return {
            "vocabularies": list(self.resources),
            "methods": list(self.methods),
            "top_k": self.top_k,
            "respect_validity_dates": self.respect_validity_dates,
        }

    def retrieve(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        text = sau.mention.retrieval_text or sau.mention.text
        vocabularies = list(self.resources)
        hits: list[LexiconHit] = []
        channel = RetrievalChannel.NORMALIZED_LOOKUP
        for method in self.methods:
            if method == "exact":
                hits, channel = (
                    self.store.exact(text, vocabularies),
                    RetrievalChannel.NORMALIZED_LOOKUP,
                )
            else:
                hits = self.store.bm25(text, vocabularies, self.top_k, kind=method)
                channel = RetrievalChannel.LEXICAL
            if hits:
                break
        candidates = [
            candidate
            for provider in self.providers
            for candidate in provider.candidates(hits, channel=channel, query=text)
        ]
        if self.respect_validity_dates:
            as_of = _parse_date(sau.context.date)
            if as_of is not None:
                candidates = [c for c in candidates if _valid_on(c, as_of)]
        return CandidateSet(retriever=self.op_id, candidates=candidates)


def _parse_date(value: str | None) -> date | None:
    match = _ISO_DATE.match((value or "").strip())
    if match is None:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _valid_on(candidate: CandidateConcept, as_of: date) -> bool:
    start = candidate.extensions.get("valid_start_date")
    end = candidate.extensions.get("valid_end_date")
    if isinstance(start, str) and date.fromisoformat(start) > as_of:
        return False
    return not (isinstance(end, str) and date.fromisoformat(end) < as_of)


class _DenseOptions(BioSemAlignModel):
    index: str = Field(description="Name of the bundle's embedding index.")
    encoder: Literal["hashing", "sentence_transformers"] = "hashing"
    dimension: int = Field(default=256, ge=8, description="Hashing encoder dimension.")
    vocabularies: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)


class DenseRetrieveOp:
    """Nearest names by embedding similarity, from a bundle's embedding index."""

    op_id = "dense"
    version = "0.1.0"
    requires_network = False

    def __init__(
        self,
        lexical: LexiconRetrieveOp,
        index: EmbeddingIndex,
        *,
        top_k: int,
    ) -> None:
        self.providers = lexical.providers
        self.index = index
        self.top_k = top_k

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(provider.vocabulary for provider in self.providers)

    @property
    def options(self) -> dict[str, Any]:
        manifest = self.index.manifest
        return {
            "index": manifest.name,
            "model_id": manifest.model_id,
            "weights_sha256": manifest.weights_sha256[:16],
            "vocabularies": list(self.resources),
            "top_k": self.top_k,
        }

    def retrieve(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        text = sau.mention.retrieval_text or sau.mention.text
        hits = self.index.search(text, list(self.resources), self.top_k)
        candidates = [
            candidate
            for provider in self.providers
            for candidate in provider.candidates(
                hits, channel=RetrievalChannel.EMBEDDING, query=text
            )
        ]
        return CandidateSet(retriever=self.op_id, candidates=candidates)


# -- fusion ------------------------------------------------------------------


class _ConcordanceOptions(BioSemAlignModel):
    equivalence_keys: list[str] | None = Field(
        default=None,
        description=(
            "Key types allowed to establish equivalence. Default: the identity of every "
            "vocabulary in the bundle, i.e. cross-references to a served vocabulary."
        ),
    )


class ConcordanceFuseOp:
    """The default fusion, plus a concordance assessment over exact matches."""

    op_id = "cross_vocab_concordance"
    version = "0.1.0"

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        equivalence_keys: Collection[str] | None = None,
    ) -> None:
        self.inner = KnowledgePackageFuseOp(knowledge, profile)
        self.store = knowledge.lexicon
        served = (
            [v.vocabulary_id for v in self.store.manifest.vocabularies]
            if self.store is not None
            else [profile.task.target_terminology]
        )
        self.equivalence_keys = sorted(set(equivalence_keys or served))
        self.source_aliases = dict(self.store.manifest.source_aliases) if self.store else {}

    @property
    def options(self) -> dict[str, Any]:
        return {"equivalence_keys": list(self.equivalence_keys)}

    def fuse(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage:
        package = self.inner.fuse(sau, candidate_sets)
        assessment = assess_concordance(
            (candidate for result in candidate_sets for candidate in result.candidates),
            equivalence_keys=self.equivalence_keys,
            key_lookup=self._keys if self.store is not None else None,
            source_aliases=self.source_aliases,
        )
        return package.model_copy(update={"concordance": assessment})

    def _keys(
        self, members: Iterable[tuple[str, str]]
    ) -> Mapping[tuple[str, str], set[tuple[str, str]]]:
        assert self.store is not None
        return self.store.keys(members)


# -- proposal ----------------------------------------------------------------


class _ConcordantOptions(BioSemAlignModel):
    authority_sources: list[str] = Field(
        default_factory=list,
        description="Sources whose single-source exact matches are trusted, e.g. ['RxNorm'].",
    )
    minimum_mention_length: int = Field(
        default=5,
        ge=1,
        description="Shorter mentions (acronyms such as 'MTX') always pass to the next tier.",
    )


class ConcordantExactProposeOp:
    """Propose only when exact matches agree on one target concept.

    The identity comes from concordance; relationship, granularity, and
    semantic loss come from the deterministic projection rules, unchanged.
    """

    op_id = "concordant_exact"
    version = "0.1.0"
    requires_network = False

    def __init__(
        self,
        adjudicator: DeterministicAdjudicator,
        profile: TaskProfile,
        *,
        authority_sources: Collection[str] = (),
        minimum_mention_length: int = 5,
    ) -> None:
        self.adjudicator = adjudicator
        self.target = profile.task.target_terminology
        self.authority_sources = frozenset(authority_sources)
        self.minimum_mention_length = minimum_mention_length

    @property
    def options(self) -> dict[str, Any]:
        return {
            "authority_sources": sorted(self.authority_sources),
            "minimum_mention_length": self.minimum_mention_length,
            "model": self.adjudicator.provenance().model_name,
        }

    def propose(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext:
        assessment = package.concordance
        if assessment is None:
            return PassToNext("no_concordance_assessment", "the fuser did not assess concordance")
        if len(sau.mention.normalized_text.replace(" ", "")) < self.minimum_mention_length:
            return PassToNext(
                "short_mention",
                f"mentions shorter than {self.minimum_mention_length} characters are too "
                "ambiguous to resolve by name alone",
            )
        passes = {
            ConcordanceState.NO_EXACT_MATCH: "no_exact_match",
            ConcordanceState.CROSS_VOCAB_COLLISION: "lexical_collision",
            ConcordanceState.EQUIVALENCE_CONFLICT: "equivalence_conflict",
        }
        if assessment.state in passes:
            return PassToNext(passes[assessment.state], _describe_groups(package))
        group = assessment.groups[0]
        if assessment.state is ConcordanceState.SINGLE_SOURCE_EXACT and not (
            set(group.sources) & self.authority_sources
        ):
            return PassToNext(
                "single_source",
                f"only {', '.join(group.sources)} asserts this name, which is not an "
                "authority source",
            )
        targets = [m for m in group.members if m.vocabulary.casefold() == self.target.casefold()]
        if len(targets) != 1:
            return PassToNext(
                "no_target_concept",
                f"the agreed entity has {len(targets)} {self.target} concepts",
            )
        identity = package.candidate_by_qualified_id(self.target, targets[0].concept_code)
        if identity is None:
            return PassToNext(
                "target_not_in_package",
                f"{self.target} {targets[0].concept_code} fell outside the bounded package",
            )
        return self.adjudicator.project(sau, identity)

    def provenance(self) -> ModelProvenance:
        return self.adjudicator.provenance()


def _describe_groups(package: KnowledgePackage) -> str | None:
    if package.concordance is None or not package.concordance.groups:
        return None
    return " | ".join(
        ", ".join(f"{m.vocabulary}:{m.concept_code}" for m in group.members)
        for group in package.concordance.groups
    )


# -- registration ------------------------------------------------------------


def _knowledge(context: OpContext, name: str) -> KnowledgeRouter:
    if context.knowledge is None:
        raise ConfigurationError(f"operator {name!r} needs the pipeline's knowledge router")
    return context.knowledge


def _lexicon(context: OpContext, options: Mapping[str, Any]) -> LexiconRetrieveOp:
    parsed = _LexiconOptions.model_validate(options)
    return LexiconRetrieveOp(
        _knowledge(context, "lexicon"),
        context.profile,
        vocabularies=parsed.vocabularies,
        methods=parsed.methods,
        top_k=parsed.top_k,
        respect_validity_dates=parsed.respect_validity_dates,
    )


def _concordance(context: OpContext, options: Mapping[str, Any]) -> ConcordanceFuseOp:
    parsed = _ConcordanceOptions.model_validate(options)
    return ConcordanceFuseOp(
        _knowledge(context, "cross_vocab_concordance"),
        context.profile,
        equivalence_keys=parsed.equivalence_keys,
    )


def _concordant_exact(context: OpContext, options: Mapping[str, Any]) -> ConcordantExactProposeOp:
    parsed = _ConcordantOptions.model_validate(options)
    adjudicator = context.adjudicator
    if not isinstance(adjudicator, DeterministicAdjudicator):
        # The projection rules are the deterministic adjudicator's, whatever
        # backend later tiers use.
        adjudicator = DeterministicAdjudicator(context.profile)
    return ConcordantExactProposeOp(
        adjudicator,
        context.profile,
        authority_sources=parsed.authority_sources,
        minimum_mention_length=parsed.minimum_mention_length,
    )


def _dense(context: OpContext, options: Mapping[str, Any]) -> DenseRetrieveOp:
    parsed = _DenseOptions.model_validate(options)
    knowledge = _knowledge(context, "dense")
    lexical = LexiconRetrieveOp(
        knowledge, context.profile, vocabularies=parsed.vocabularies, methods=["exact"]
    )
    encoder: TextEncoder
    if parsed.encoder == "hashing":
        encoder = HashingEncoder(parsed.dimension)
    else:
        model = context.settings.embedding_model_path if context.settings else None
        if model is None:
            raise ConfigurationError(
                "the sentence_transformers encoder needs BIOSEMALIGN_EMBEDDING_MODEL_PATH"
            )
        encoder = SentenceTransformerEncoder(model)
    index = EmbeddingIndex(lexical.store, parsed.index, encoder)
    return DenseRetrieveOp(
        lexical, index, top_k=parsed.top_k or context.profile.retrieval.maximum_candidates
    )


register_op("retrieve", "lexicon", _lexicon)
register_op("retrieve", "dense", _dense)
register_op("fuse", "cross_vocab_concordance", _concordance)
register_op("propose", "concordant_exact", _concordant_exact)
