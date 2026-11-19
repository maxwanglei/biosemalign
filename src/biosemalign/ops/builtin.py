"""Built-in operators.

Most of these are thin adapters over the v0.1 stages, so the default workflow
is the v0.1 pipeline expressed as one tier. They register themselves under the
names profiles use; a third-party operator registers the same way.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, model_validator

from biosemalign.adjudication.backend import Adjudicator
from biosemalign.enums import DecisionType, RetrievalChannel
from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.candidate_fusion import RRF_K, fuse_candidates, fuse_ranked
from biosemalign.knowledge.candidate_retrieval import RETRIEVER_VERSION, CandidateRetriever
from biosemalign.knowledge.enrichment import enrich_candidates
from biosemalign.knowledge.package_builder import build_knowledge_package
from biosemalign.knowledge.providers.rxnorm import RxNavProvider
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.ops.base import CandidateSet, OpContext, PassToNext
from biosemalign.ops.registry import register_op
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.validation.drug_rules import validate_drug_semantics
from biosemalign.validation.policy_dsl import PolicyValidateOp, load_policy_pack

__all__ = [
    "AdapterTransformOp",
    "AdjudicatorProposeOp",
    "CleanerResult",
    "DrugSemanticsValidateOp",
    "ExtractStrengthOp",
    "KnowledgePackageFuseOp",
    "RankFusionFuseOp",
    "RxNavRetrieveOp",
    "UnicodeNormalizeOp",
]

_WHITESPACE = re.compile(r"\s+")
_STRENGTH = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|l|unit|units|iu|%)(?:\s*/\s*\d*(?:\.\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|ml|l|actuat|hr|h))?(?=\W|$)",
    re.IGNORECASE,
)


# -- transforms --------------------------------------------------------------


class UnicodeNormalizeOp:
    """Query with NFKC-normalized, whitespace-collapsed text.

    Full-width characters, ligatures, and non-breaking spaces from PDF or
    spreadsheet exports otherwise defeat exact lookup.
    """

    op_id = "unicode_normalize"
    version = "0.1.0"

    def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        base = sau.mention.retrieval_text or sau.mention.text
        canonical = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", base)).strip()
        return _with_retrieval_text(sau, canonical)


class ExtractStrengthOp:
    """Move a stated strength out of the query and into the ``strength`` slot.

    "TOPROL XL 50 MG" is queried as "TOPROL XL" while ``strength='50 MG'`` is
    kept as an attribute, so the decision records the strength rather than the
    transform erasing it. A strength the source already asserted is never
    overwritten.
    """

    op_id = "extract_strength"
    version = "0.1.0"

    def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        base = sau.mention.retrieval_text or sau.mention.text
        matches = list(_STRENGTH.finditer(base))
        if not matches:
            return sau
        remainder = _WHITESPACE.sub(" ", _STRENGTH.sub(" ", base)).strip()
        if not remainder:
            # A bare strength names no drug; leave the query untouched.
            return sau
        updated = _with_retrieval_text(sau, remainder)
        if updated.attributes.strength is None:
            strength = ", ".join(_WHITESPACE.sub(" ", m.group(0)).strip() for m in matches)
            updated = updated.model_copy(
                update={"attributes": updated.attributes.model_copy(update={"strength": strength})}
            )
        return updated


@dataclass(frozen=True, slots=True)
class CleanerResult:
    """What an external drug-name cleaner returns for one string."""

    canonical_text: str | None
    slots: Mapping[str, str] = field(default_factory=dict)


class AdapterTransformOp:
    """Wrap an external cleaner (Aioli, DiAna, a lab regex set) as a transform.

    The cleaner's canonical string becomes the retrieval query and its
    extracted slots fill empty attributes. The kernel's non-destructiveness
    check still applies, so a cleaner cannot rewrite the source mention or an
    attribute the source asserted.
    """

    def __init__(
        self,
        cleaner: Callable[[str], CleanerResult],
        *,
        op_id: str,
        version: str,
    ) -> None:
        self.cleaner = cleaner
        self.op_id = op_id
        self.version = version

    def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        result = self.cleaner(sau.mention.retrieval_text or sau.mention.text)
        updated = sau
        if result.canonical_text and result.canonical_text.strip():
            updated = _with_retrieval_text(updated, result.canonical_text.strip())
        fills = {
            name: value
            for name, value in result.slots.items()
            if name in type(updated.attributes).model_fields
            and name != "extensions"
            and getattr(updated.attributes, name) is None
        }
        if fills:
            updated = updated.model_copy(
                update={"attributes": updated.attributes.model_copy(update=fills)}
            )
        return updated


def _with_retrieval_text(sau: SemanticAlignmentUnit, text: str) -> SemanticAlignmentUnit:
    if text == sau.mention.text:
        return sau
    return sau.model_copy(
        update={"mention": sau.mention.model_copy(update={"retrieval_text": text})}
    )


# -- retrieval and fusion ----------------------------------------------------


class _RxNavOptions(BioSemAlignModel):
    channels: list[RetrievalChannel] | None = Field(
        default=None,
        description="Restrict retrieval to these channels; None runs every enabled channel.",
    )


class RxNavRetrieveOp:
    """The v0.1 multi-channel RxNav retriever as a workflow operator."""

    op_id = "rxnav"
    version = RETRIEVER_VERSION
    # Served from the release-bound cache in FROZEN mode; never requires a live call.
    requires_network = False

    def __init__(
        self,
        provider: RxNavProvider,
        profile: TaskProfile,
        *,
        channels: Collection[RetrievalChannel] | None = None,
    ) -> None:
        self.retriever = CandidateRetriever(provider, profile, channels=channels)

    @property
    def resources(self) -> tuple[str, ...]:
        return (str(self.retriever.provider.resource_name),)

    @property
    def options(self) -> dict[str, Any]:
        channels = self.retriever.channels
        return {} if channels is None else {"channels": sorted(c.value for c in channels)}

    def code(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        """The coding module's name for :meth:`retrieve`."""
        return self.retrieve(sau)

    def retrieve(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        return CandidateSet(retriever=self.op_id, candidates=self.retriever.retrieve(sau))


class KnowledgePackageFuseOp:
    """Deduplicate, rank, enrich, and bound candidates (the v0.1 fusion)."""

    op_id = "knowledge_package"
    version = "0.1.0"

    def __init__(self, knowledge: KnowledgeRouter, profile: TaskProfile) -> None:
        self.knowledge = knowledge
        self.profile = profile

    def link(
        self, sau: SemanticAlignmentUnit, code_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage:
        """The linking module's name for :meth:`fuse`."""
        return self.fuse(sau, code_sets)

    def fuse(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage:
        raw = [candidate for result in candidate_sets for candidate in result.candidates]
        fused = fuse_candidates(
            raw,
            maximum=self.profile.retrieval.maximum_candidates,
            target_vocabulary=self.profile.task.target_terminology,
        )
        return self.package(sau, fused)

    def package(self, sau: SemanticAlignmentUnit, fused: list[Any]) -> KnowledgePackage:
        """Enrich ranked candidates and bound them into a package.

        Candidates of the target terminology are enriched, and so are those of
        any permitted source terminology present, which a proposer may choose
        as the source identity and must be able to project.
        """
        enriched = enrich_candidates(
            fused, provider=self.knowledge.primary_provider(), profile=self.profile
        )
        present = {candidate.vocabulary.casefold() for candidate in fused}
        for vocabulary in self.profile.task.source_terminologies:
            if vocabulary.casefold() in present:
                enriched = enrich_candidates(
                    enriched, provider=self.knowledge.provider_for(vocabulary), profile=self.profile
                )
        return build_knowledge_package(
            sau_id=sau.sau_id,
            candidates=enriched,
            profile=self.profile,
            resource_provenance=self.knowledge.resource_provenance(),
        )


class _RankFusionOptions(BioSemAlignModel):
    k: int = Field(default=RRF_K, ge=1, description="RRF constant; larger flattens rank weight.")


class RankFusionFuseOp(KnowledgePackageFuseOp):
    """Reciprocal rank fusion of several retrievers' rankings (hybrid retrieval)."""

    op_id = "reciprocal_rank_fusion"

    def __init__(self, knowledge: KnowledgeRouter, profile: TaskProfile, *, k: int = RRF_K) -> None:
        super().__init__(knowledge, profile)
        self.k = k

    @property
    def options(self) -> dict[str, Any]:
        return {"k": self.k}

    def fuse(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage:
        fused = fuse_ranked(
            [result.candidates for result in candidate_sets],
            maximum=self.profile.retrieval.maximum_candidates,
            target_vocabulary=self.profile.task.target_terminology,
            k=self.k,
        )
        return self.package(sau, fused)


# -- proposal and validation -------------------------------------------------


class _AdjudicatorOptions(BioSemAlignModel):
    pass_on: list[DecisionType] = Field(
        default_factory=list,
        description="Proposal outcomes handed to the next tier instead of being finalized.",
    )
    pass_on_review_request: bool = Field(
        default=False,
        description="Hand proposals that request human review to the next tier.",
    )


class AdjudicatorProposeOp:
    """Expose an :class:`Adjudicator` as a proposer that can escalate.

    The adjudicator never learns it is in a cascade. Escalation is decided
    here, from the outcome it proposed, so an exact-match tier can hand its
    misses to a broader tier without either adjudicator changing.
    """

    version = "0.1.0"
    requires_network = False

    def __init__(
        self,
        adjudicator: Adjudicator,
        *,
        pass_on: Collection[DecisionType] = (),
        pass_on_review_request: bool = False,
    ) -> None:
        self.adjudicator = adjudicator
        self.pass_on = frozenset(pass_on)
        self.pass_on_review_request = pass_on_review_request
        # Startup verification compares this with retrieval.maximum_candidates.
        self.max_candidates: int | None = getattr(adjudicator, "max_candidates", None)
        # v0.1 never read adjudicator_name, so custom adjudicators may lack it.
        name = getattr(adjudicator, "adjudicator_name", None) or adjudicator.provenance().backend
        self.op_id = f"adjudicator:{name}"

    @property
    def options(self) -> dict[str, Any]:
        # The adjudicator's own rules or model version, so that a workflow
        # digest changes whenever what decides changes.
        options: dict[str, Any] = {"model": self.adjudicator.provenance().model_name}
        if self.pass_on:
            options["pass_on"] = sorted(decision.value for decision in self.pass_on)
        if self.pass_on_review_request:
            options["pass_on_review_request"] = True
        return options

    def resolve(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext:
        """The resolution module's name for :meth:`propose`."""
        return self.propose(sau, package)

    def propose(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext:
        proposal = self.adjudicator.adjudicate(sau, package)
        if proposal.decision in self.pass_on:
            return PassToNext(
                reason=f"decision:{proposal.decision.value}", detail=proposal.rationale
            )
        if self.pass_on_review_request and proposal.requested_human_review:
            return PassToNext(reason="review_requested", detail=proposal.rationale)
        return proposal

    def provenance(self) -> ModelProvenance:
        return self.adjudicator.provenance()


class DrugSemanticsValidateOp:
    """Combination integrity, salt conflation, and semantic-loss accounting."""

    op_id = "drug_semantics"
    version = "0.1.0"

    def __init__(self, profile: TaskProfile) -> None:
        self.profile = profile

    def validate(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]:
        return validate_drug_semantics(proposal, package, self.profile, sau=sau)


# -- registration ------------------------------------------------------------


def _require_knowledge(context: OpContext, name: str) -> KnowledgeRouter:
    if context.knowledge is None:
        raise ConfigurationError(f"operator {name!r} needs the pipeline's knowledge router")
    return context.knowledge


def _require_adjudicator(context: OpContext, name: str) -> Adjudicator:
    if context.adjudicator is None:
        raise ConfigurationError(f"operator {name!r} needs the pipeline's adjudicator")
    return context.adjudicator


def _no_options(name: str, options: Mapping[str, Any]) -> None:
    if options:
        raise ConfigurationError(f"operator {name!r} takes no options, got {sorted(options)}")


def _unicode_normalize(context: OpContext, options: Mapping[str, Any]) -> UnicodeNormalizeOp:
    _no_options("unicode_normalize", options)
    return UnicodeNormalizeOp()


def _extract_strength(context: OpContext, options: Mapping[str, Any]) -> ExtractStrengthOp:
    _no_options("extract_strength", options)
    return ExtractStrengthOp()


def _rxnav(context: OpContext, options: Mapping[str, Any]) -> RxNavRetrieveOp:
    parsed = _RxNavOptions.model_validate(options)
    provider = _require_knowledge(context, "rxnav").provider_for("RxNorm")
    if not isinstance(provider, RxNavProvider):
        raise ConfigurationError(
            "RxNorm is served by the local lexicon bundle in this deployment; use the "
            "'lexicon' retriever instead of 'rxnav'"
        )
    return RxNavRetrieveOp(provider, context.profile, channels=parsed.channels)


def _knowledge_package(context: OpContext, options: Mapping[str, Any]) -> KnowledgePackageFuseOp:
    _no_options("knowledge_package", options)
    return KnowledgePackageFuseOp(_require_knowledge(context, "knowledge_package"), context.profile)


def _rank_fusion(context: OpContext, options: Mapping[str, Any]) -> RankFusionFuseOp:
    parsed = _RankFusionOptions.model_validate(options)
    return RankFusionFuseOp(
        _require_knowledge(context, "reciprocal_rank_fusion"), context.profile, k=parsed.k
    )


def _adjudicator(context: OpContext, options: Mapping[str, Any]) -> AdjudicatorProposeOp:
    parsed = _AdjudicatorOptions.model_validate(options)
    return AdjudicatorProposeOp(
        _require_adjudicator(context, "adjudicator"),
        pass_on=parsed.pass_on,
        pass_on_review_request=parsed.pass_on_review_request,
    )


class _LLMOptions(_AdjudicatorOptions):
    max_candidates: int | None = Field(default=None, ge=1, le=50)


def _llm(context: OpContext, options: Mapping[str, Any]) -> AdjudicatorProposeOp:
    from biosemalign.adjudication.adjudicator import build_llm_adjudicator

    parsed = _LLMOptions.model_validate(options)
    knowledge = _require_knowledge(context, "llm")
    if context.settings is None:
        raise ConfigurationError("operator 'llm' needs the pipeline's settings")
    adjudicator = build_llm_adjudicator(
        context.profile,
        settings=context.settings,
        cache=knowledge.cache,
        max_candidates=parsed.max_candidates,
    )
    return AdjudicatorProposeOp(
        adjudicator, pass_on=parsed.pass_on, pass_on_review_request=parsed.pass_on_review_request
    )


def _drug_semantics(context: OpContext, options: Mapping[str, Any]) -> DrugSemanticsValidateOp:
    _no_options("drug_semantics", options)
    return DrugSemanticsValidateOp(context.profile)


class _PolicyOptions(BioSemAlignModel):
    path: str | None = Field(default=None, description="YAML policy pack file.")
    pack: dict[str, Any] | None = Field(default=None, description="An inline policy pack.")

    @model_validator(mode="after")
    def _exactly_one(self) -> _PolicyOptions:
        if (self.path is None) == (self.pack is None):
            raise ValueError("give exactly one of 'path' or 'pack'")
        return self


def _policy(context: OpContext, options: Mapping[str, Any]) -> PolicyValidateOp:
    parsed = _PolicyOptions.model_validate(options)
    source: str | dict[str, Any] = parsed.path if parsed.path is not None else parsed.pack or {}
    return PolicyValidateOp(
        load_policy_pack(source), target_terminology=context.profile.task.target_terminology
    )


register_op("transform", "unicode_normalize", _unicode_normalize)
register_op("transform", "extract_strength", _extract_strength)
register_op("retrieve", "rxnav", _rxnav)
register_op("fuse", "knowledge_package", _knowledge_package)
register_op("fuse", "reciprocal_rank_fusion", _rank_fusion)
register_op("propose", "adjudicator", _adjudicator)
register_op("propose", "llm", _llm)
register_op("validate", "drug_semantics", _drug_semantics)
register_op("validate", "policy", _policy)
