"""RxNav match operator (`drug.rxnav`)."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from pydantic import Field

from biosemalign.domains.drug.providers.retriever import RETRIEVER_VERSION, CandidateRetriever
from biosemalign.domains.drug.providers.rxnav import RxNavProvider
from biosemalign.enums import RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import CandidateSet, OpContext
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["RxNavMatcher", "_rxnav_factory"]


class _RxNavOptions(BioSemAlignModel):
    exact_lookup: bool = True
    approximate_match: bool = True
    include_historical: bool = True
    channels: list[RetrievalChannel] | None = Field(
        default=None,
        description="Restrict matching to these channels; None runs every enabled channel.",
    )


class RxNavMatcher:
    """Multi-channel RxNav matcher."""

    op_id = "drug.rxnav"
    version = RETRIEVER_VERSION
    uses_model = False
    # Served from the release-bound cache in FROZEN mode; never requires a live call.
    requires_network = False

    def __init__(
        self,
        provider: RxNavProvider,
        profile: TaskProfile,
        *,
        exact_lookup: bool = True,
        approximate_match: bool = True,
        include_historical: bool = True,
        channels: Collection[RetrievalChannel] | None = None,
    ) -> None:
        self.retriever = CandidateRetriever(
            provider,
            profile,
            exact_lookup=exact_lookup,
            approximate_match=approximate_match,
            include_historical=include_historical,
            channels=channels,
        )

    @property
    def resources(self) -> tuple[str, ...]:
        return (str(self.retriever.provider.resource_name),)

    @property
    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "exact_lookup": self.retriever.exact_lookup,
            "approximate_match": self.retriever.approximate_match,
            "include_historical": self.retriever.include_historical,
        }
        if self.retriever.channels is not None:
            opts["channels"] = sorted(c.value for c in self.retriever.channels)
        return opts

    def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        return CandidateSet(matcher=self.op_id, candidates=self.retriever.retrieve(sau))


def _rxnav_factory(context: OpContext, options: Mapping[str, Any]) -> RxNavMatcher:
    if context.knowledge is None:
        raise ConfigurationError("operator 'drug.rxnav' needs the pipeline's knowledge router")
    parsed = _RxNavOptions.model_validate(options)
    provider = context.knowledge.provider_for("RxNorm")
    if not isinstance(provider, RxNavProvider):
        raise ConfigurationError(
            "RxNorm is served by the local lexicon bundle in this deployment; use 'exact' or "
            "'lexical' instead of 'drug.rxnav'"
        )
    return RxNavMatcher(
        provider,
        context.profile,
        exact_lookup=parsed.exact_lookup,
        approximate_match=parsed.approximate_match,
        include_historical=parsed.include_historical,
        channels=parsed.channels,
    )
