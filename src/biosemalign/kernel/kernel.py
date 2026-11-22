"""The alignment kernel: the only path from an SAU to a decision (§5.5).

Workflows are assembled from replaceable operators, so the guarantees that
make a decision trustworthy cannot live in any of them. The kernel applies
them around whatever workflow it is given:

* outcome-leakage filtering of every package, after the linker ran;
* resolver isolation — resolvers see copies, and blinded context is withheld;
* non-destructive cleaners — source text and asserted context are immutable;
* mandatory validation (unretrieved identifiers, concept status) and routing;
* provenance, including which tier resolved the SAU and why earlier ones passed.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from biosemalign.enums import DecisionType, RelationshipType, TierOutcome
from biosemalign.exceptions import AdjudicationFailure, WorkflowViolation
from biosemalign.kernel.guards import (
    isolated_view,
    mark_candidate_set_from_model,
    mark_expansion_member_from_model,
    merge_model_cleaned_sau,
    require_non_destructive,
    resolver_view,
    withheld_fields,
)
from biosemalign.kernel.tiers import Cascade, Tier
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.logging import get_logger
from biosemalign.modules.base import CandidateSet, PassToNext
from biosemalign.modules.link.package_builder import enforce_relationship_policy
from biosemalign.modules.link.targets import derive_link_set
from biosemalign.modules.model import ModelService, capture_model_calls
from biosemalign.profiles.schema import TaskProfile
from biosemalign.routing.router import DecisionRouter
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.codes import QualifiedCode, RefinementOutcome
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.manifest import TierTiming
from biosemalign.schemas.provenance import (
    MemoProvenance,
    ModelCallRecord,
    ModelProvenance,
    Provenance,
    ResolutionTrace,
    ResourceProvenance,
    RunContext,
    TierAttempt,
)
from biosemalign.schemas.sau import SemanticAlignmentUnit, canonical_preserved_attributes
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
    model_calls: tuple[ModelCallRecord, ...] = ()


@dataclass(slots=True)
class _TierClock:
    attempts: int = 0
    resolved: int = 0
    seconds: float = field(default=0.0)


class PackageCache:
    """Bounded reuse of linked packages across equivalent SAUs in one batch.

    Keyed by tier and ``mapping_key``, which deliberately excludes row
    identity, so repeated product strings are matched once per tier.
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
        model_services: Sequence[ModelService] = (),
    ) -> None:
        self.profile = profile
        self.workflow = workflow
        self.knowledge = knowledge
        self.validator = validator
        self.router = router
        self._explicit_model_services = tuple(model_services)
        self.target_spec = profile.target_spec()
        self.workflow_provenance = workflow.provenance(validator.checks, self.target_spec)
        self._clocks: dict[str, _TierClock] = {tier.name: _TierClock() for tier in workflow.tiers}
        self.memo: CrosswalkMemo | None = None
        self.memo_resources: dict[str, str | None] = {}
        self.memo_hits = 0

    def _all_model_services(self) -> list[ModelService]:
        """Collect all unique :class:`ModelService` instances reachable from the workflow."""
        services: list[ModelService] = []
        seen: set[int] = set()

        def add(candidate: object) -> None:
            if isinstance(candidate, ModelService) and id(candidate) not in seen:
                seen.add(id(candidate))
                services.append(candidate)

        for svc in self._explicit_model_services:
            add(svc)
        add(getattr(self.validator, "model_service", None))
        for tier in self.workflow.tiers:
            for op in [*tier.cleaners, *tier.matchers, tier.expander, tier.linker, tier.resolver]:
                if op is not None:
                    add(getattr(op, "model_service", None))
        for check in self.validator.checks:
            add(getattr(check, "model_service", None))
        return services

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
        """Clean the SAU for one tier and build its filtered package."""
        view = self._cleaned(tier, sau)
        package = cache.get(tier.name, view) if cache is not None else None
        if package is None:
            candidate_sets: list[CandidateSet] = []
            for matcher in tier.matchers:
                matcher_uses_model = bool(matcher.uses_model)
                matcher_sau = (
                    isolated_view(view, "match", self.profile) if matcher_uses_model else view
                )
                try:
                    cs = matcher.match(matcher_sau)
                except AdjudicationFailure:
                    if matcher_uses_model:
                        # M6: failed model reply in match yields no suggestion.
                        cs = CandidateSet(matcher=matcher.op_id, candidates=[])
                    else:
                        raise
                if view.mention.retrieval_text_from_model or matcher_uses_model:
                    cs = mark_candidate_set_from_model(cs)
                candidate_sets.append(cs)

            refinement: RefinementOutcome | None = None
            if tier.expander is not None:
                anchors = [c for cs in candidate_sets for c in cs.candidates]
                expander_uses_model = bool(tier.expander.uses_model)
                expander_sau = (
                    isolated_view(view, "expand", self.profile)
                    if expander_uses_model
                    else view.model_copy(deep=True)
                )
                try:
                    expansion = tier.expander.expand(expander_sau, anchors)
                except AdjudicationFailure:
                    if expander_uses_model:
                        # M6: failed model reply in expand yields no suggestion.
                        expansion = ExpansionSet(expander=tier.expander.op_id, anchors=[])
                    else:
                        raise
                if not isinstance(expansion, ExpansionSet):
                    raise WorkflowViolation(
                        f"tier {tier.name!r}: expander {tier.expander.op_id!r} returned "
                        f"{type(expansion).__name__}, not an ExpansionSet"
                    )
                surviving: list[ExpansionMember] = [
                    member
                    for member in expansion.members
                    if all(
                        (
                            step.predicate not in self.profile.knowledge.prohibited_relationships
                            if step.direction == "lexical"
                            else self.profile.knowledge.permits(step.predicate)
                        )
                        for step in member.path
                    )
                ]
                anchors_from_model = any(
                    ev.from_model_suggestion
                    for anchor in anchors
                    for ev in anchor.retrieval_evidence
                )
                if (
                    expander_uses_model
                    or view.mention.retrieval_text_from_model
                    or anchors_from_model
                ):
                    surviving = [mark_expansion_member_from_model(m) for m in surviving]
                refinement = _recompute_refinement(
                    expansion.refinement,
                    surviving,
                    removed=len(surviving) != len(expansion.members),
                    truncated=expansion.truncated,
                )
                if refinement is None:
                    if surviving:
                        candidate_sets = [
                            *candidate_sets,
                            CandidateSet(
                                matcher=tier.expander.op_id,
                                candidates=[member.concept for member in surviving],
                                origin="expand",
                            ),
                        ]
                elif refinement.status == "unique" and surviving:
                    candidate_sets = [
                        *candidate_sets,
                        CandidateSet(
                            matcher=tier.expander.op_id,
                            candidates=[member.concept for member in surviving],
                            origin="refinement",
                        ),
                    ]
            linker_sau = (
                isolated_view(view, "link", self.profile) if tier.linker.uses_model else view
            )
            package = tier.linker.link(linker_sau, candidate_sets)
            if package.sau_id != view.sau_id:
                raise WorkflowViolation(
                    f"tier {tier.name!r}: linker {tier.linker.op_id!r} built a package for "
                    "a different SAU"
                )
            package = enforce_relationship_policy(package, self.profile)
            # The typed links are the kernel's reading of what the linker put
            # in the package, taken after the leakage filter so that no link
            # rests on an edge the profile prohibits.
            package_updates: dict[str, Any] = {
                "links": derive_link_set(package, self.target_spec, self.profile.target.level)
            }
            if (
                refinement is not None
                and refinement.status == "unique"
                and refinement.selected is not None
                and package.candidate_by_qualified_id(
                    refinement.selected.vocabulary, refinement.selected.concept_code
                )
                is None
            ):
                refinement = RefinementOutcome(
                    status="none",
                    anchor=refinement.anchor,
                    slots_checked=refinement.slots_checked,
                    candidates=[],
                    selected=None,
                    detail=(
                        f"refined candidate {refinement.selected.vocabulary}:"
                        f"{refinement.selected.concept_code} excluded from bounded package; "
                        f"{refinement.detail}"
                    ),
                )
            if refinement is not None:
                package_updates["refinement"] = refinement
            package = package.model_copy(update=package_updates)
            if cache is not None:
                cache.put(tier.name, view, package)
        return view, package

    def _cleaned(self, tier: Tier, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        view = sau
        for cleaner in tier.cleaners:
            uses_model = bool(cleaner.uses_model)
            before = isolated_view(view, "clean", self.profile) if uses_model else view
            try:
                cleaned = cleaner.clean(before.model_copy(deep=True))
            except AdjudicationFailure:
                if uses_model:
                    # M6: failed or unusable model reply in clean yields no suggestion.
                    continue
                raise
            require_non_destructive(before, cleaned, cleaner.op_id)
            if not uses_model:
                view = cleaned
                continue
            view = merge_model_cleaned_sau(view, cleaned)
        return view

    def resolve(
        self, sau: SemanticAlignmentUnit, *, cache: PackageCache | None = None
    ) -> Resolution:
        """Run the cascade until a tier resolves, or every tier has passed."""
        attempts: list[TierAttempt] = []
        withheld = withheld_fields(sau, self.profile)
        services = self._all_model_services()
        all_calls: list[ModelCallRecord] = []
        view, package = sau, None
        for tier in self.workflow.tiers:
            clock = self._clocks[tier.name]
            started = time.perf_counter()
            with capture_model_calls(services) as tier_calls:
                try:
                    view, package = self.tier_knowledge(tier, sau, cache)
                    resolver_sau = resolver_view(view, self.profile)
                    try:
                        raw_result = tier.resolver.resolve(
                            resolver_sau, package.model_copy(deep=True)
                        )
                    except AdjudicationFailure:
                        if tier.resolver.uses_model:
                            raw_result = self._model_abstention(resolver_sau)
                        else:
                            raise
                    result = tier.apply_escalation(raw_result)
                finally:
                    clock.attempts += 1
                    clock.seconds += time.perf_counter() - started
                    for rec in tier_calls:
                        all_calls.append(
                            rec.model_copy(update={"tier": tier.name}) if rec.tier is None else rec
                        )
            coded = _coding(package)
            if isinstance(result, PassToNext):
                attempts.append(
                    TierAttempt(
                        tier=tier.name,
                        outcome=TierOutcome.PASSED,
                        reason=result.reason,
                        detail=result.detail,
                        **coded,
                    )
                )
                continue
            if not isinstance(result, AlignmentProposal):
                raise WorkflowViolation(
                    f"tier {tier.name!r}: resolver {tier.resolver.op_id!r} returned "
                    f"{type(result).__name__}; resolvers may only return an "
                    "AlignmentProposal or PassToNext"
                )
            attempts.append(TierAttempt(tier=tier.name, outcome=TierOutcome.RESOLVED, **coded))
            clock.resolved += 1
            return Resolution(
                sau=view,
                package=package,
                proposal=result,
                trace=ResolutionTrace(
                    attempts=tuple(attempts), resolved_by_tier=tier.name, withheld_context=withheld
                ),
                model=tier.resolver.provenance(),
                model_calls=tuple(all_calls),
            )

        assert package is not None  # a Cascade always has at least one tier
        return Resolution(
            sau=view,
            package=package,
            proposal=self._exhausted(view, attempts),
            trace=ResolutionTrace(attempts=tuple(attempts), withheld_context=withheld),
            model=ModelProvenance(backend="kernel", model_name=f"cascade@{KERNEL_VERSION}"),
            model_calls=tuple(all_calls),
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
            model_calls=resolution.model_calls,
        )
        if self.memo is not None and key is not None:
            self.memo.put(
                key,
                MemoEntry(
                    package=resolution.package,
                    proposal=resolution.proposal,
                    trace=resolution.trace,
                    model=resolution.model,
                    model_calls=list(resolution.model_calls),
                    cleaned_mention=resolution.sau.mention,
                    cleaned_attributes=resolution.sau.attributes,
                    run_id=run.run_id,
                    stored_at=CrosswalkMemo.now(),
                ),
            )
        return artifact

    def _replay(
        self, sau: SemanticAlignmentUnit, entry: MemoEntry, key: str, run: RunContext
    ) -> AlignmentArtifact:
        """Finalize a memoized resolution for a new occurrence of the same input."""
        if entry.cleaned_mention is not None and entry.cleaned_attributes is not None:
            view = sau.model_copy(
                deep=True,
                update={
                    "mention": entry.cleaned_mention.model_copy(deep=True),
                    "attributes": entry.cleaned_attributes.model_copy(deep=True),
                },
            )
        else:
            tier_name = entry.trace.resolved_by_tier or entry.trace.attempts[-1].tier
            tier = next(t for t in self.workflow.tiers if t.name == tier_name)
            view = sau.model_copy(deep=True)
            for cleaner in tier.cleaners:
                if bool(cleaner.uses_model):
                    continue
                before = view
                cleaned = cleaner.clean(before.model_copy(deep=True))
                require_non_destructive(before, cleaned, cleaner.op_id)
                view = cleaned
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
            model_calls=entry.model_calls,
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
        model_calls: Sequence[ModelCallRecord] = (),
    ) -> AlignmentArtifact:
        """Validate, route, and assemble the auditable artifact."""
        if package.sau_id != sau.sau_id:
            raise ValueError("knowledge package identity does not match the SAU")
        services = self._all_model_services()
        with capture_model_calls(services) as check_calls:
            validation = self.validator.validate(proposal, package, sau)
        combined_calls = [*model_calls, *check_calls]
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
                model_calls=combined_calls,
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
        model_calls: Sequence[ModelCallRecord] = (),
    ) -> Provenance:
        """Fill in the run's component versions and freeze it (§17.3)."""
        matcher_versions = "+".join(
            dict.fromkeys(op.version for tier in self.workflow.tiers for op in tier.matchers)
        )
        run.retriever_version = matcher_versions or None
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
        provenance.model_calls = list(model_calls)
        return provenance

    # -- internals -----------------------------------------------------------

    def _model_abstention(self, sau: SemanticAlignmentUnit) -> AlignmentProposal:
        """M6 fallback when a model-backed resolver fails with an unusable reply."""
        relation = (
            RelationshipType.UNSUPPORTED_MAPPING
            if self.profile.permits_relation(RelationshipType.UNSUPPORTED_MAPPING)
            else next(iter(self.profile.task.relation_types), RelationshipType.UNSUPPORTED_MAPPING)
        )
        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=DecisionType.ABSTAINED,
            relationship_to_source=relation,
            preserved_attributes=canonical_preserved_attributes(sau.attributes),
            rationale="the model reply could not be used; a reviewer must decide",
            requested_human_review=True,
        )

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
            preserved_attributes=canonical_preserved_attributes(sau.attributes),
            rationale="every workflow tier passed: "
            + "; ".join(f"{attempt.tier}: {attempt.reason}" for attempt in attempts),
            requested_human_review=True,
        )


def _recompute_refinement(
    refinement: RefinementOutcome | None,
    surviving: list[ExpansionMember],
    *,
    removed: bool,
    truncated: bool = False,
) -> RefinementOutcome | None:
    """Adjust refinement if G2 relationship filtering removed expanded members."""
    if refinement is None or not removed or refinement.status in {"none", "not_applicable"}:
        return refinement
    codes = [
        QualifiedCode(
            vocabulary=m.concept.vocabulary,
            concept_code=m.concept.candidate_id,
            label=m.concept.preferred_label,
        )
        for m in surviving
    ]
    if not codes:
        return RefinementOutcome(
            status="none",
            anchor=refinement.anchor,
            slots_checked=refinement.slots_checked,
            candidates=[],
            selected=None,
            detail=f"refined candidates filtered by relationship policy; {refinement.detail}",
        )
    if len(codes) == 1 and not (truncated and refinement.status == "multiple"):
        return RefinementOutcome(
            status="unique",
            anchor=refinement.anchor,
            slots_checked=refinement.slots_checked,
            candidates=codes,
            selected=codes[0],
            detail=refinement.detail,
        )
    return RefinementOutcome(
        status="multiple",
        anchor=refinement.anchor,
        slots_checked=refinement.slots_checked,
        candidates=codes,
        selected=None,
        detail=refinement.detail,
    )


def _coding(package: KnowledgePackage) -> dict[str, Any]:
    """What a tier's package coded: its concordance state and every exact match."""
    if package.concordance is None:
        return {}
    return {
        "concordance_state": package.concordance.state,
        "exact_matches": tuple(group.model_copy(deep=True) for group in package.concordance.groups),
    }
