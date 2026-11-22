"""Local lexicon bundle match operators (`exact`, `lexical`, `dense`)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Literal

from pydantic import Field

from biosemalign.enums import RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.input.context import context_view
from biosemalign.knowledge.lexicon.embeddings import (
    EmbeddingIndex,
    HashingEncoder,
    SentenceTransformerEncoder,
    TextEncoder,
)
from biosemalign.knowledge.providers.lexicon import LexiconHit, LexiconProvider, LexiconStore
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import CandidateSet, OpContext
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["DenseMatcher", "ExactMatcher", "LexicalMatcher"]

LexicalMethod = Literal["bm25", "trigram"]
_ISO_DATE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})")


def _resolve_providers(
    knowledge: KnowledgeRouter,
    profile: TaskProfile,
    vocabularies: Sequence[str] | None,
    *,
    op_name: str,
) -> tuple[LexiconStore, list[LexiconProvider]]:
    if knowledge.lexicon is None:
        raise ConfigurationError(
            f"the {op_name!r} matcher needs a lexicon bundle; set BIOSEMALIGN_LEXICON_PATH"
        )
    names = list(vocabularies or [])
    if not names:
        names = [name for name in profile.knowledge.resources if knowledge.serves_locally(name)]
    if not names:
        raise ConfigurationError(
            f"the {op_name!r} matcher has no vocabulary to query: none of the profile's "
            "resources is in the lexicon bundle"
        )
    providers: list[LexiconProvider] = []
    for name in names:
        provider = knowledge.provider_for(name)
        if not isinstance(provider, LexiconProvider):
            raise ConfigurationError(f"vocabulary {name!r} is not served by the lexicon bundle")
        providers.append(provider)
    return knowledge.lexicon, providers


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


class _BaseLexiconMatcher:
    """Shared local-bundle provider resolution and validity-date filtering."""

    version = "0.1.0"
    uses_model = False
    requires_network = False

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        op_id: str,
        vocabularies: Sequence[str] | None = None,
        top_k: int | None = None,
        respect_validity_dates: bool = False,
        required_index: str,
    ) -> None:
        store, providers = _resolve_providers(knowledge, profile, vocabularies, op_name=op_id)
        if not store.supports(required_index):
            raise ConfigurationError(
                f"the lexicon bundle has no index for {[required_index]}; it supports "
                f"{store.manifest.indexes}"
            )
        self.op_id = op_id
        self.profile = profile
        self.store = store
        self.providers = providers
        self.top_k = top_k or profile.limits.candidates
        self.respect_validity_dates = respect_validity_dates

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(provider.vocabulary for provider in self.providers)

    def _to_candidate_set(
        self,
        sau: SemanticAlignmentUnit,
        hits: list[LexiconHit],
        *,
        channel: RetrievalChannel,
        query: str,
    ) -> CandidateSet:
        candidates = [
            candidate
            for provider in self.providers
            for candidate in provider.candidates(hits, channel=channel, query=query)
        ]
        if self.respect_validity_dates:
            raw_date = context_view(sau, "match", self.profile).get("date")
            as_of = _parse_date(str(raw_date) if raw_date is not None else None)
            if as_of is not None:
                candidates = [c for c in candidates if _valid_on(c, as_of)]
        return CandidateSet(matcher=self.op_id, candidates=candidates)


class _ExactOptions(BioSemAlignModel):
    vocabularies: list[str] | None = Field(
        default=None,
        description="Vocabularies to query; default every profile resource the bundle serves.",
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
    respect_validity_dates: bool = Field(
        default=False,
        description="Drop concepts not valid on the record's date (context.date), when given.",
    )


class ExactMatcher(_BaseLexiconMatcher):
    """Exact normalized-name lookup across the local lexicon bundle."""

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        vocabularies: Sequence[str] | None = None,
        top_k: int | None = None,
        respect_validity_dates: bool = False,
    ) -> None:
        super().__init__(
            knowledge,
            profile,
            op_id="exact",
            vocabularies=vocabularies,
            top_k=top_k,
            respect_validity_dates=respect_validity_dates,
            required_index="exact",
        )

    @property
    def options(self) -> dict[str, Any]:
        return {
            "vocabularies": list(self.resources),
            "top_k": self.top_k,
            "respect_validity_dates": self.respect_validity_dates,
        }

    def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        text = sau.mention.retrieval_text or sau.mention.text
        hits = self.store.exact(text, list(self.resources))
        return self._to_candidate_set(
            sau, hits, channel=RetrievalChannel.NORMALIZED_LOOKUP, query=text
        )


class _LexicalOptions(BioSemAlignModel):
    vocabularies: list[str] | None = Field(
        default=None,
        description="Vocabularies to query; default every profile resource the bundle serves.",
    )
    method: LexicalMethod = Field(
        default="trigram",
        description="Ranked lexical index: 'bm25' (words) or 'trigram' (character n-grams).",
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
    respect_validity_dates: bool = Field(
        default=False,
        description="Drop concepts not valid on the record's date (context.date), when given.",
    )


class LexicalMatcher(_BaseLexiconMatcher):
    """Ranked BM25 or character-trigram lookup across the local lexicon bundle."""

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        vocabularies: Sequence[str] | None = None,
        method: LexicalMethod = "trigram",
        top_k: int | None = None,
        respect_validity_dates: bool = False,
    ) -> None:
        super().__init__(
            knowledge,
            profile,
            op_id="lexical",
            vocabularies=vocabularies,
            top_k=top_k,
            respect_validity_dates=respect_validity_dates,
            required_index=method,
        )
        self.method = method

    @property
    def options(self) -> dict[str, Any]:
        return {
            "vocabularies": list(self.resources),
            "method": self.method,
            "top_k": self.top_k,
            "respect_validity_dates": self.respect_validity_dates,
        }

    def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        text = sau.mention.retrieval_text or sau.mention.text
        hits = self.store.bm25(text, list(self.resources), self.top_k, kind=self.method)
        return self._to_candidate_set(sau, hits, channel=RetrievalChannel.LEXICAL, query=text)


class _DenseOptions(BioSemAlignModel):
    index: str = Field(description="Name of the bundle's embedding index.")
    encoder: Literal["hashing", "sentence_transformers"] = "hashing"
    dimension: int = Field(default=256, ge=8, description="Hashing encoder dimension.")
    vocabularies: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)


class DenseMatcher:
    """Nearest names by embedding similarity, from a bundle's embedding index."""

    op_id = "dense"
    version = "0.1.0"
    uses_model = False
    requires_network = False

    def __init__(
        self,
        providers: Sequence[LexiconProvider],
        index: EmbeddingIndex,
        *,
        top_k: int,
    ) -> None:
        self.providers = list(providers)
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

    def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        text = sau.mention.retrieval_text or sau.mention.text
        hits = self.index.search(text, list(self.resources), self.top_k)
        candidates = [
            candidate
            for provider in self.providers
            for candidate in provider.candidates(
                hits, channel=RetrievalChannel.EMBEDDING, query=text
            )
        ]
        return CandidateSet(matcher=self.op_id, candidates=candidates)


def _require_knowledge(context: OpContext, name: str) -> KnowledgeRouter:
    if context.knowledge is None:
        raise ConfigurationError(f"operator {name!r} needs the pipeline's knowledge router")
    return context.knowledge


def _exact_factory(context: OpContext, options: Mapping[str, Any]) -> ExactMatcher:
    parsed = _ExactOptions.model_validate(options)
    return ExactMatcher(
        _require_knowledge(context, "exact"),
        context.profile,
        vocabularies=parsed.vocabularies,
        top_k=parsed.top_k,
        respect_validity_dates=parsed.respect_validity_dates,
    )


def _lexical_factory(context: OpContext, options: Mapping[str, Any]) -> LexicalMatcher:
    parsed = _LexicalOptions.model_validate(options)
    return LexicalMatcher(
        _require_knowledge(context, "lexical"),
        context.profile,
        vocabularies=parsed.vocabularies,
        method=parsed.method,
        top_k=parsed.top_k,
        respect_validity_dates=parsed.respect_validity_dates,
    )


def _dense_factory(context: OpContext, options: Mapping[str, Any]) -> DenseMatcher:
    parsed = _DenseOptions.model_validate(options)
    knowledge = _require_knowledge(context, "dense")
    store, providers = _resolve_providers(
        knowledge, context.profile, parsed.vocabularies, op_name="dense"
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
    index = EmbeddingIndex(store, parsed.index, encoder)
    return DenseMatcher(providers, index, top_k=parsed.top_k or context.profile.limits.candidates)
