"""The alignment kernel: the only path from an SAU to a decision (v0.2).

Workflows are assembled from replaceable operators, so the guarantees that
make a decision trustworthy cannot live in any of them. The kernel applies
them around whatever workflow it is given:

* outcome-leakage filtering of every fused package, after the fuser ran;
* proposer isolation — proposers see copies, and blinded context is withheld;
* non-destructive transforms — source text and asserted context are immutable;
* mandatory validation (unretrieved identifiers, concept status) and routing;
* provenance, including which tier resolved the SAU and why earlier ones passed.

An operator can only return a proposal or a pass. Decisions are minted by the
router inside :meth:`AlignmentKernel.finalize`, and the resulting artifact
re-derives its own consistency on construction, so a decision that skipped
these steps does not validate as an artifact.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from biosemalign.enums import DecisionType, RelationshipType, TierOutcome
from biosemalign.exceptions import AdjudicationFailure, WorkflowViolation
from biosemalign.knowledge.candidate_retrieval import RETRIEVER_VERSION
from biosemalign.knowledge.package_builder import enforce_relationship_policy
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.logging import get_logger
from biosemalign.ops.base import PassToNext
from biosemalign.pipelines.workflow import Cascade, Tier
from biosemalign.profiles.schema import TaskProfile
from biosemalign.routing.router import DecisionRouter
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.manifest import TierTiming
from biosemalign.schemas.provenance import (
    MemoProvenance,
    ModelProvenance,
    Provenance,
    ResolutionTrace,
    ResourceProvenance,
    RunContext,
    TierAttempt,
)
from biosemalign.schemas.sau import ContextEnvelope, SemanticAlignmentUnit
from biosemalign.storage.crosswalk_memo import CrosswalkMemo, MemoEntry, memo_key
from biosemalign.util import stable_id
from biosemalign.validation.validator import Validator

__all__ = ["KERNEL_VERSION", "AlignmentKernel", "PackageCache", "Resolution"]

KERNEL_VERSION = "0.1.0"

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Resolution:
    """A workflow's outcome for one SAU, before validation and routing."""

    sau: SemanticAlignmentUnit
    package: KnowledgePackage
    proposal: AlignmentProposal
    trace: ResolutionTrace
    model: ModelProvenance | None


@dataclass(slots=True)
class _TierClock:
    attempts: int = 0
    resolved: int = 0
    seconds: float = field(default=0.0)


class PackageCache:
    """Bounded reuse of fused packages across equivalent SAUs in one batch.

    Keyed by tier and ``mapping_key``, which deliberately excludes row
    identity, so repeated product strings are retrieved once per tier.
    """

    def __init__(self, maximum: int = 1024) -> None:
        self.maximum = maximum
        self._entries: OrderedDict[tuple[str, str], KnowledgePackage] = OrderedDict()

    def get(self, tier: str, sau: SemanticAlignmentUnit) -> KnowledgePackage | None:
        key = (tier, sau.mapping_key)
        cached = self._entries.get(key)
        if cached is None:
            return None
        self._entries.move_to_end(key)
        return cached.model_copy(deep=True, update={"sau_id": sau.sau_id})

    def put(self, tier: str, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> None:
        self._entries[(tier, sau.mapping_key)] = package.model_copy(deep=True)
        if len(self._entries) > self.maximum:
            self._entries.popitem(last=False)


class AlignmentKernel:
    """Runs a workflow, then validates, routes, and stamps what it proposes."""

    version = KERNEL_VERSION

    def __init__(
        self,
        *,
        profile: TaskProfile,
        workflow: Cascade,
        knowledge: KnowledgeRouter,
        validator: Validator,
        router: DecisionRouter,
    ) -> None:
        self.profile = profile
        self.workflow = workflow
        self.knowledge = knowledge
        self.validator = validator
        self.router = router
        self.workflow_provenance = workflow.provenance(validator.rules)
        self._clocks: dict[str, _TierClock] = {tier.name: _TierClock() for tier in workflow.tiers}
        self.memo: CrosswalkMemo | None = None
        self.memo_resources: dict[str, str | None] = {}
        self.memo_hits = 0

    def tier_timings(self) -> dict[str, TierTiming]:
        """Per-tier work since the last reset, for the run manifest."""
        return {
            name: TierTiming(
                attempts=clock.attempts, resolved=clock.resolved, seconds=round(clock.seconds, 6)
            )
            for name, clock in self._clocks.items()
        }

    def reset_timings(self) -> None:
        for clock in self._clocks.values():
            clock.attempts, clock.resolved, clock.seconds = 0, 0, 0.0

    # -- workflow execution --------------------------------------------------

    def tier_knowledge(
        self,
        tier: Tier,
        sau: SemanticAlignmentUnit,
        cache: PackageCache | None = None,
    ) -> tuple[SemanticAlignmentUnit, KnowledgePackage]:
        """Transform the SAU for one tier and build its filtered package."""
        view = self._transformed(tier, sau)
        package = cache.get(tier.name, view) if cache is not None else None
        if package is None:
            candidate_sets = [retriever.retrieve(view) for retriever in tier.retrievers]
            package = tier.fuser.fuse(view, candidate_sets)
            if package.sau_id != view.sau_id:
                raise WorkflowViolation(
                    f"tier {tier.name!r}: fuser {tier.fuser.op_id!r} built a package for "
                    "a different SAU"
                )
            package = enforce_relationship_policy(package, self.profile)
            if cache is not None:
                cache.put(tier.name, view, package)
        return view, package

    def _transformed(self, tier: Tier, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        view = sau
        for transform in tier.transforms:
            transformed = transform.transform(view.model_copy(deep=True))
            _require_non_destructive(view, transformed, transform.op_id)
            view = transformed
        return view

    def resolve(
        self, sau: SemanticAlignmentUnit, *, cache: PackageCache | None = None
    ) -> Resolution:
        """Run the cascade until a tier proposes, or every tier has passed."""
        attempts: list[TierAttempt] = []
        withheld = self._withheld_fields(sau)
        view, package = sau, None
        for tier in self.workflow.tiers:
            clock = self._clocks[tier.name]
            started = time.perf_counter()
            try:
                view, package = self.tier_knowledge(tier, sau, cache)
                result = tier.proposer.propose(
                    self._proposer_view(view), package.model_copy(deep=True)
                )
            finally:
                clock.attempts += 1
                clock.seconds += time.perf_counter() - started
            if isinstance(result, PassToNext):
                attempts.append(
                    TierAttempt(
                        tier=tier.name,
                        outcome=TierOutcome.PASSED,
                        reason=result.reason,
                        detail=result.detail,
                    )
                )
                continue
            if not isinstance(result, AlignmentProposal):
                raise WorkflowViolation(
                    f"tier {tier.name!r}: proposer {tier.proposer.op_id!r} returned "
                    f"{type(result).__name__}; proposers may only return an "
                    "AlignmentProposal or PassToNext"
                )
            attempts.append(TierAttempt(tier=tier.name, outcome=TierOutcome.RESOLVED))
            clock.resolved += 1
            return Resolution(
                sau=view,
                package=package,
                proposal=result,
                trace=ResolutionTrace(
                    attempts=tuple(attempts), resolved_by_tier=tier.name, withheld_context=withheld
                ),
                model=tier.proposer.provenance(),
            )

        assert package is not None  # a Cascade always has at least one tier
        return Resolution(
            sau=view,
            package=package,
            proposal=self._exhausted(view, attempts),
            trace=ResolutionTrace(attempts=tuple(attempts), withheld_context=withheld),
            model=ModelProvenance(backend="kernel", model_name=f"cascade@{KERNEL_VERSION}"),
        )

    def align(
        self,
        sau: SemanticAlignmentUnit,
        run: RunContext,
        *,
        cache: PackageCache | None = None,
    ) -> AlignmentArtifact:
        """Resolve one SAU through the workflow — or the memo — and finalize it."""
        key = None
        if self.memo is not None:
            key = memo_key(
                sau,
                workflow_id=self.workflow_provenance.workflow_id,
                resources=self.memo_resources,
            )
            entry = self.memo.get(key)
            if entry is not None:
                artifact = self._replay(sau, entry, key, run)
                self.memo_hits += 1
                return artifact
        resolution = self.resolve(sau.model_copy(deep=True), cache=cache)
        artifact = self.finalize(
            sau=resolution.sau,
            package=resolution.package,
            proposal=resolution.proposal,
            run=run,
            model=resolution.model,
            resolution=resolution.trace,
        )
        if self.memo is not None and key is not None:
            self.memo.put(
                key,
                MemoEntry(
                    package=resolution.package,
                    proposal=resolution.proposal,
                    trace=resolution.trace,
                    model=resolution.model,
                    run_id=run.run_id,
                    stored_at=CrosswalkMemo.now(),
                ),
            )
        return artifact

    def _replay(
        self, sau: SemanticAlignmentUnit, entry: MemoEntry, key: str, run: RunContext
    ) -> AlignmentArtifact:
        """Finalize a memoized resolution for a new occurrence of the same input."""
        tier_name = entry.trace.resolved_by_tier or entry.trace.attempts[-1].tier
        tier = next(t for t in self.workflow.tiers if t.name == tier_name)
        view = self._transformed(tier, sau.model_copy(deep=True))
        package = entry.package.model_copy(deep=True, update={"sau_id": sau.sau_id})
        proposal = entry.proposal.model_copy(deep=True, update={"sau_id": sau.sau_id})
        return self.finalize(
            sau=view,
            package=package,
            proposal=proposal,
            run=run,
            model=entry.model,
            resolution=entry.trace,
            memo=MemoProvenance(key=key, original_run_id=entry.run_id, stored_at=entry.stored_at),
        )

    # -- finalization --------------------------------------------------------

    def finalize(
        self,
        *,
        sau: SemanticAlignmentUnit,
        package: KnowledgePackage,
        proposal: AlignmentProposal,
        run: RunContext,
        model: ModelProvenance | None,
        resolution: ResolutionTrace | None,
        memo: MemoProvenance | None = None,
    ) -> AlignmentArtifact:
        """Validate, route, and assemble the auditable artifact."""
        if package.sau_id != sau.sau_id:
            raise ValueError("knowledge package identity does not match the SAU")
        validation = self.validator.validate(proposal, package, sau)
        decision = self.router.route(
            sau=sau,
            package=package,
            proposal=proposal,
            validation=validation,
            provenance=self.provenance(
                run,
                model=model,
                resolution=resolution,
                memo=memo,
                # Reused evidence was gathered when it was first resolved; its
                # resources, not this run's (untouched) ones, describe it.
                resources=package.resource_provenance if memo is not None else None,
            ),
        )
        return AlignmentArtifact(
            artifact_id=stable_id(
                "ART",
                {
                    "run_id": run.run_id,
                    "sau": sau.model_dump(mode="json"),
                    "knowledge_package": package.model_dump(mode="json"),
                    "proposal": proposal.model_dump(mode="json"),
                    "validation": validation.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                },
                length=32,
            ),
            run_id=run.run_id,
            mapping_key=sau.mapping_key,
            sau=sau,
            knowledge_package=package,
            proposal=proposal,
            validation=validation,
            decision=decision,
        )

    def provenance(
        self,
        run: RunContext,
        *,
        model: ModelProvenance | None,
        resolution: ResolutionTrace | None,
        memo: MemoProvenance | None = None,
        resources: list[ResourceProvenance] | None = None,
    ) -> Provenance:
        """Fill in the run's component versions and freeze it (§17.3)."""
        run.retriever_version = RETRIEVER_VERSION
        run.validation_rules_version = self.validator.version
        run.routing_policy_version = self.router.version
        run.model = model
        run.workflow = self.workflow_provenance

        provenance = run.freeze()
        # The router owns the providers, so it is the only thing that knows
        # which resources were actually touched and at which release.
        provenance.resources = (
            list(resources) if resources is not None else self.knowledge.resource_provenance()
        )
        provenance.resolution = resolution
        provenance.memo = memo
        return provenance

    # -- internals -----------------------------------------------------------

    def _withheld_fields(self, sau: SemanticAlignmentUnit) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.profile.knowledge.blinded_context_fields
            if getattr(sau.context, name) != _blank(name)
        )

    def _proposer_view(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        """An isolated copy with blinded context removed."""
        view = sau.model_copy(deep=True)
        blinded = self.profile.knowledge.blinded_context_fields
        if not blinded:
            return view
        context = view.context.model_copy(update={name: _blank(name) for name in blinded})
        return view.model_copy(update={"context": context})

    def _exhausted(
        self, sau: SemanticAlignmentUnit, attempts: list[TierAttempt]
    ) -> AlignmentProposal:
        """Every tier passed: abstain, and ask a person rather than drop the SAU."""
        if not self.profile.permits_relation(RelationshipType.UNSUPPORTED_MAPPING):
            raise AdjudicationFailure(
                f"profile {self.profile.profile_name} cannot represent the abstention an "
                "exhausted cascade requires (UNSUPPORTED_MAPPING)"
            )
        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=DecisionType.ABSTAINED,
            relationship_to_source=RelationshipType.UNSUPPORTED_MAPPING,
            preserved_attributes=sau.attributes.model_dump(
                exclude_none=True, exclude={"extensions"}
            ),
            rationale="every workflow tier passed: "
            + "; ".join(f"{attempt.tier}: {attempt.reason}" for attempt in attempts),
            requested_human_review=True,
        )


def _blank(field_name: str) -> object:
    return ContextEnvelope.model_fields[field_name].get_default(call_default_factory=True)


def _require_non_destructive(
    before: SemanticAlignmentUnit, after: SemanticAlignmentUnit, op_id: str
) -> None:
    """Allow a transform to set the retrieval query and fill empty attributes only."""
    if not isinstance(after, SemanticAlignmentUnit):
        raise WorkflowViolation(f"transform {op_id!r} returned {type(after).__name__}, not an SAU")
    exclude: dict[str, Any] = {"mention": {"retrieval_text"}, "attributes": True}
    if before.model_dump(exclude=exclude) != after.model_dump(exclude=exclude):
        raise WorkflowViolation(
            f"transform {op_id!r} changed the source mention, identity, or context; transforms "
            "may only set mention.retrieval_text and fill empty attributes"
        )
    previous = before.attributes.model_dump(exclude_none=True, exclude={"extensions"})
    current = after.attributes.model_dump(exclude_none=True, exclude={"extensions"})
    overwritten = sorted(name for name, value in previous.items() if current.get(name) != value)
    lost_extensions = sorted(
        key
        for key, value in before.attributes.extensions.items()
        if after.attributes.extensions.get(key) != value
    )
    if overwritten or lost_extensions:
        raise WorkflowViolation(
            f"transform {op_id!r} overwrote or removed asserted attributes: "
            f"{overwritten + lost_extensions}"
        )
