"""Standard linker operator (`standard`), unifying merge and concordance."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import CandidateSet, OpContext
from biosemalign.modules.link.concordance import assess_concordance
from biosemalign.modules.link.enrichment import enrich_candidates
from biosemalign.modules.link.fusion import RRF_K, fuse_candidate_sets
from biosemalign.modules.link.package_builder import build_knowledge_package
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["MergeStrategy", "StandardLinker", "_standard_factory"]

MergeStrategy = Literal["priority", "rrf"]


class _StandardOptions(BioSemAlignModel):
    merge: MergeStrategy = "priority"
    k: int = Field(default=RRF_K, ge=1, description="RRF constant; larger flattens rank weight.")
    concordance: bool = False
    equivalence_keys: list[str] | None = Field(
        default=None,
        description=(
            "Key types allowed to establish equivalence when concordance is true. "
            "Default: every vocabulary served by the bundle, or target terminology."
        ),
    )
    conflict_vocabularies: list[str] | None = Field(
        default=None,
        description=(
            "Vocabularies whose duplicate codes make a group a conflict, as one pool, "
            "e.g. ['RxNorm', 'RxNorm Extension']. Default: duplicates in any vocabulary."
        ),
    )


class StandardLinker:
    """Unified linker: deduplicates, merges (priority or RRF), enriches, and optionally assesses concordance."""

    op_id = "standard"
    version = "0.2.0"
    uses_model = False

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        *,
        merge: MergeStrategy = "priority",
        k: int = RRF_K,
        concordance: bool = False,
        equivalence_keys: Collection[str] | None = None,
        conflict_vocabularies: Collection[str] | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.merge = merge
        self.k = k
        self.concordance = concordance
        self.store = knowledge.lexicon
        served = (
            [v.vocabulary_id for v in self.store.manifest.vocabularies]
            if self.store is not None
            else [profile.target.terminology]
        )
        self.equivalence_keys = sorted(set(equivalence_keys or served)) if concordance else []
        self.conflict_vocabularies = (
            sorted(set(conflict_vocabularies))
            if concordance and conflict_vocabularies is not None
            else None
        )
        self.source_aliases = dict(self.store.manifest.source_aliases) if self.store else {}

    @property
    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"merge": self.merge, "concordance": self.concordance}
        if self.merge == "rrf":
            opts["k"] = self.k
        if self.concordance:
            opts["equivalence_keys"] = list(self.equivalence_keys)
            if self.conflict_vocabularies is not None:
                opts["conflict_vocabularies"] = list(self.conflict_vocabularies)
        return opts

    def link(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage:
        maximum = self.profile.limits.candidates
        target = self.profile.target.terminology
        fused = fuse_candidate_sets(
            candidate_sets,
            merge=self.merge,
            maximum=maximum,
            target_vocabulary=target,
            k=self.k,
        )

        enriched = enrich_candidates(
            fused, provider=self.knowledge.primary_provider(), profile=self.profile
        )
        present = {candidate.vocabulary.casefold() for candidate in fused}
        for vocabulary in self.profile.target.source_terminologies:
            if vocabulary.casefold() in present:
                enriched = enrich_candidates(
                    enriched, provider=self.knowledge.provider_for(vocabulary), profile=self.profile
                )

        package = build_knowledge_package(
            sau_id=sau.sau_id,
            candidates=enriched,
            profile=self.profile,
            resource_provenance=self.knowledge.resource_provenance(),
        )
        if not self.concordance:
            return package

        assessment = assess_concordance(
            (
                candidate
                for result in candidate_sets
                if result.origin == "match"
                for candidate in result.candidates
            ),
            equivalence_keys=self.equivalence_keys,
            key_lookup=self._keys if self.store is not None else None,
            source_aliases=self.source_aliases,
            conflict_vocabularies=self.conflict_vocabularies,
        )
        return package.model_copy(update={"concordance": assessment})

    def _keys(
        self, members: Iterable[tuple[str, str]]
    ) -> Mapping[tuple[str, str], set[tuple[str, str]]]:
        assert self.store is not None
        return self.store.keys(members)


def _standard_factory(context: OpContext, options: Mapping[str, Any]) -> StandardLinker:
    if context.knowledge is None:
        raise ConfigurationError("operator 'standard' needs the pipeline's knowledge router")
    parsed = _StandardOptions.model_validate(options)
    return StandardLinker(
        context.knowledge,
        context.profile,
        merge=parsed.merge,
        k=parsed.k,
        concordance=parsed.concordance,
        equivalence_keys=parsed.equivalence_keys,
        conflict_vocabularies=parsed.conflict_vocabularies,
    )
