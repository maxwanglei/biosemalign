"""Per-SAU coding workflow and cascade runner (§5.4, §9 WP5)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from biosemalign.enums import ResourceMode, RetrievalChannel
from biosemalign.exceptions import AdjudicationFailure, ConfigurationError, WorkflowViolation
from biosemalign.kernel.guards import (
    isolated_view,
    mark_candidate_set_from_model,
    merge_model_cleaned_sau,
    require_non_destructive,
)
from biosemalign.kernel.tiers import ParallelUnion, operator_provenance
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules import OpContext, build
from biosemalign.modules.base import CandidateSet, Cleaner, Linker, Matcher
from biosemalign.modules.link.package_builder import enforce_relationship_policy
from biosemalign.modules.link.targets import derive_link_set
from biosemalign.modules.model import ModelService, capture_model_calls
from biosemalign.modules.registry import default_matcher_for_domain
from biosemalign.profiles.schema import OperatorConfig, TaskProfile, TierConfig
from biosemalign.schemas.coding import CodedConcept, CodingArtifact, CodingTierAttempt
from biosemalign.schemas.knowledge import CandidateConcept, RetrievalEvidence
from biosemalign.schemas.links import CodeLink, LinkSet
from biosemalign.schemas.provenance import (
    ModelCallRecord,
    Provenance,
    RunContext,
    TierProvenance,
    WorkflowProvenance,
)
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.settings import Settings
from biosemalign.targets.spec import TargetSpec
from biosemalign.util import stable_digest

__all__ = [
    "CodingCascade",
    "CodingTier",
    "CodingWorkflow",
    "coding_workflow_from_profile",
]


@dataclass(frozen=True, slots=True)
class CodingTier:
    """One tier of a coding cascade: ``clean? -> match (-> link, optional)``."""

    name: str
    match: Matcher | ParallelUnion
    cleaners: tuple[Cleaner, ...] = ()
    linker: Linker | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cleaners", tuple(self.cleaners))
        checks: list[tuple[str, object, type]] = [
            *(("matcher", op, Matcher) for op in self.matchers),
            *(("cleaner", op, Cleaner) for op in self.cleaners),
        ]
        if self.linker is not None:
            checks.append(("linker", self.linker, Linker))
        for role, op, protocol in checks:
            if not isinstance(op, protocol):
                raise ConfigurationError(
                    f"coding tier {self.name!r}: {role} {op!r} does not implement "
                    f"{protocol.__name__}"
                )

    @property
    def matchers(self) -> tuple[Matcher, ...]:
        if isinstance(self.match, ParallelUnion):
            return self.match.matchers
        return (self.match,)

    def provenance(self) -> TierProvenance:
        ops = [
            *(operator_provenance("clean", op) for op in self.cleaners),
            *(operator_provenance("match", op) for op in self.matchers),
        ]
        if self.linker is not None:
            ops.append(operator_provenance("link", self.linker))
        return TierProvenance(name=self.name, operators=tuple(ops))


@dataclass(frozen=True, slots=True, init=False)
class CodingCascade:
    """Coding tiers tried in order until one matches at least one code."""

    tiers: tuple[CodingTier, ...]

    def __init__(self, tiers: Iterable[CodingTier]) -> None:
        object.__setattr__(self, "tiers", tuple(tiers))
        if not self.tiers:
            raise ConfigurationError("a coding cascade needs at least one tier")
        names = [tier.name for tier in self.tiers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigurationError(f"coding cascade tier names must be unique: {duplicates}")

    def verify(self, profile: TaskProfile, settings: Settings) -> None:
        """Fail at startup on configurations that cannot run safely."""
        if settings.mode is ResourceMode.FROZEN:
            live: list[str] = []
            for tier in self.tiers:
                ops: list[object] = [*tier.cleaners, *tier.matchers]
                if tier.linker is not None:
                    ops.append(tier.linker)
                live.extend(
                    f"{tier.name}/{getattr(op, 'op_id', '?')}"
                    for op in ops
                    if getattr(op, "requires_network", False)
                )
            if live:
                raise ConfigurationError(
                    "FROZEN mode forbids operators that require live network access: "
                    + ", ".join(live)
                )
        declared = list(profile.knowledge.resources)
        matchers = [op for tier in self.tiers for op in tier.matchers]
        if declared and all(hasattr(op, "resources") for op in matchers):
            queried = {
                str(name).casefold() for op in matchers for name in getattr(op, "resources", ())
            }
            unqueried = [name for name in declared if name.casefold() not in queried]
            if unqueried:
                raise ConfigurationError(
                    f"profile resources {unqueried} are declared but no coding matcher "
                    "queries them; their evidence would be silently omitted"
                )

    def verify_local_only(self, knowledge: KnowledgeRouter | None = None) -> None:
        """Refuse any network, non-local-provider, or model-backed operator (§5.4)."""
        disallowed: list[str] = []
        for tier in self.tiers:
            ops: list[object] = [*tier.cleaners, *tier.matchers]
            if tier.linker is not None:
                ops.append(tier.linker)
            for op in ops:
                op_id = getattr(op, "op_id", repr(op))
                if getattr(op, "requires_network", False):
                    disallowed.append(f"{tier.name}/{op_id} (requires network)")
                elif bool(getattr(op, "uses_model", False)):
                    disallowed.append(f"{tier.name}/{op_id} (uses model)")
            if knowledge is not None:
                for matcher in tier.matchers:
                    for resource in getattr(matcher, "resources", ()):
                        if not knowledge.serves_locally(str(resource)):
                            disallowed.append(
                                f"{tier.name}/{matcher.op_id} (non-local resource "
                                f"{resource!r}; requires network or remote cache)"
                            )
        if disallowed:
            raise ConfigurationError(
                "interactive code_text permits local non-model operators only; found: "
                + ", ".join(disallowed)
            )

    def provenance(self, target_spec: TargetSpec | None = None) -> WorkflowProvenance:
        """The coding workflow's identity."""
        tiers = tuple(tier.provenance() for tier in self.tiers)
        payload: dict[str, object] = {
            "workflow": "coding",
            "tiers": [tier.model_dump(mode="json") for tier in tiers],
        }
        if target_spec is not None:
            payload["target_spec"] = {"label": target_spec.label, "sha256": target_spec.digest}
        return WorkflowProvenance(
            workflow_id=stable_digest(payload, length=16),
            tiers=tiers,
            validators=(),
            target_spec=target_spec.label if target_spec is not None else None,
            target_spec_sha256=target_spec.digest if target_spec is not None else None,
        )


def coding_workflow_from_profile(context: OpContext) -> CodingCascade:
    """Assemble a :class:`CodingCascade` from a profile, or the default exact coding tier."""
    tiers = context.profile.tiers
    if tiers is None:
        tiers = _default_coding_tiers(context)
    return CodingCascade(_build_coding_tier(config, context) for config in tiers)


def _default_coding_tiers(context: OpContext) -> list[TierConfig]:
    local = context.knowledge is not None and context.knowledge.serves_locally(
        context.profile.target.terminology
    )
    if local and context.knowledge is not None and context.knowledge.lexicon is not None:
        return [TierConfig(name="exact", match=[OperatorConfig(op="exact")])]
    default_matcher = default_matcher_for_domain(context.profile.domain)
    return [TierConfig(name="exact", match=[OperatorConfig(op=default_matcher)])]


def _build_coding_tier(config: TierConfig, context: OpContext) -> CodingTier:
    matchers: list[Matcher] = [build("match", op.op, context, op.options) for op in config.match]
    linker = (
        build("link", config.link.op, context, config.link.options)
        if config.link is not None
        else None
    )
    return CodingTier(
        name=config.name,
        cleaners=tuple(build("clean", op.op, context, op.options) for op in config.clean),
        match=matchers[0] if len(matchers) == 1 else ParallelUnion(matchers),
        linker=linker,
    )


class CodingWorkflow:
    """Runs a :class:`CodingCascade` per SAU and stamps an auditable :class:`CodingArtifact`."""

    def __init__(
        self,
        *,
        profile: TaskProfile,
        cascade: CodingCascade,
        knowledge: KnowledgeRouter,
        model_services: Sequence[ModelService] = (),
    ) -> None:
        self.profile = profile
        self.cascade = cascade
        self.knowledge = knowledge
        self._explicit_model_services = tuple(model_services)
        self.target_spec = profile.target_spec()
        self.workflow_provenance = cascade.provenance(self.target_spec)

    def _all_model_services(self) -> list[ModelService]:
        services: list[ModelService] = []
        seen: set[int] = set()

        def add(candidate: object) -> None:
            if isinstance(candidate, ModelService) and id(candidate) not in seen:
                seen.add(id(candidate))
                services.append(candidate)

        for svc in self._explicit_model_services:
            add(svc)
        for tier in self.cascade.tiers:
            ops: list[object] = [*tier.cleaners, *tier.matchers]
            if tier.linker is not None:
                ops.append(tier.linker)
            for op in ops:
                add(getattr(op, "model_service", None))
        return services

    def _clean_tier(
        self, tier: CodingTier, sau: SemanticAlignmentUnit
    ) -> tuple[SemanticAlignmentUnit, list[str]]:
        view = sau
        cleaners_applied: list[str] = []
        for cleaner in tier.cleaners:
            uses_model = bool(cleaner.uses_model)
            before = isolated_view(view, "clean", self.profile) if uses_model else view
            try:
                cleaned = cleaner.clean(before.model_copy(deep=True))
            except AdjudicationFailure:
                if uses_model:
                    continue
                raise
            require_non_destructive(before, cleaned, cleaner.op_id)
            next_view = merge_model_cleaned_sau(view, cleaned) if uses_model else cleaned
            if (
                next_view.mention.retrieval_text != view.mention.retrieval_text
                or next_view.attributes != view.attributes
            ):
                cleaners_applied.append(cleaner.op_id)
            view = next_view
        return view, cleaners_applied

    def code(self, sau: SemanticAlignmentUnit, run: RunContext) -> CodingArtifact:
        """Attach every matched code to ``sau`` without making a mapping decision."""
        services = self._all_model_services()
        all_calls: list[ModelCallRecord] = []
        attempts: list[CodingTierAttempt] = []
        original_sau = sau.model_copy(deep=True)
        final_sau = original_sau

        for tier in self.cascade.tiers:
            with capture_model_calls(services) as tier_calls:
                try:
                    view, cleaners_applied = self._clean_tier(
                        tier, original_sau.model_copy(deep=True)
                    )
                    query_text = view.mention.retrieval_text or view.mention.text
                    candidate_sets: list[CandidateSet] = []
                    for matcher in tier.matchers:
                        matcher_uses_model = bool(matcher.uses_model)
                        matcher_sau = (
                            isolated_view(view, "match", self.profile)
                            if matcher_uses_model
                            else view
                        )
                        try:
                            cs = matcher.match(matcher_sau)
                        except AdjudicationFailure:
                            if matcher_uses_model:
                                cs = CandidateSet(matcher=matcher.op_id, candidates=[])
                            else:
                                raise
                        if view.mention.retrieval_text_from_model or matcher_uses_model:
                            cs = mark_candidate_set_from_model(cs)
                        candidate_sets.append(cs)
                    merged = _merge_matched_candidates(candidate_sets)
                    links: LinkSet | None = None
                    canonical_by_key: dict[tuple[str, str], CandidateConcept] = {}
                    if merged and tier.linker is not None:
                        linker_sau = (
                            isolated_view(view, "link", self.profile)
                            if tier.linker.uses_model
                            else view
                        )
                        package = tier.linker.link(linker_sau, candidate_sets)
                        if package.sau_id != view.sau_id:
                            raise WorkflowViolation(
                                f"coding tier {tier.name!r}: linker {tier.linker.op_id!r} built "
                                "a package for a different SAU"
                            )
                        package = enforce_relationship_policy(package, self.profile)
                        links = derive_link_set(
                            package, self.target_spec, self.profile.target.level
                        )
                        canonical_by_key = {
                            (c.vocabulary.casefold(), c.candidate_id): c for c in package.candidates
                        }
                finally:
                    for rec in tier_calls:
                        all_calls.append(
                            rec.model_copy(update={"tier": tier.name}) if rec.tier is None else rec
                        )

            if not merged:
                attempts.append(
                    CodingTierAttempt(
                        tier=tier.name,
                        query_text=query_text,
                        cleaners_applied=cleaners_applied,
                        matched_codes=0,
                        coded=False,
                    )
                )
                final_sau = view
                continue

            codes = _build_coded_concepts(
                merged,
                canonical_by_key=canonical_by_key,
                links=links,
                retrieval_text_from_model=view.mention.retrieval_text_from_model,
            )
            attempts.append(
                CodingTierAttempt(
                    tier=tier.name,
                    query_text=query_text,
                    cleaners_applied=cleaners_applied,
                    matched_codes=len(codes),
                    coded=True,
                )
            )
            provenance = self._provenance(run, model_calls=all_calls)
            return _assemble_artifact(
                run=run,
                sau=view,
                codes=codes,
                uncoded=False,
                coded_by_tier=tier.name,
                cleaned_by=cleaners_applied,
                attempts=attempts,
                links=links,
                provenance=provenance,
            )

        provenance = self._provenance(run, model_calls=all_calls)
        return _assemble_artifact(
            run=run,
            sau=final_sau,
            codes=[],
            uncoded=True,
            coded_by_tier=None,
            cleaned_by=[],
            attempts=attempts,
            links=None,
            provenance=provenance,
        )

    def _provenance(
        self, run: RunContext, *, model_calls: Sequence[ModelCallRecord] = ()
    ) -> Provenance:
        matcher_versions = "+".join(
            dict.fromkeys(op.version for tier in self.cascade.tiers for op in tier.matchers)
        )
        run.retriever_version = matcher_versions or None
        run.validation_rules_version = None
        run.routing_policy_version = None
        run.model = None
        run.workflow = self.workflow_provenance
        frozen = run.freeze()
        frozen.resources = self.knowledge.resource_provenance()
        frozen.resolution = None
        frozen.memo = None
        frozen.model_calls = list(model_calls)
        return frozen


def _merge_matched_candidates(candidate_sets: Sequence[CandidateSet]) -> list[CandidateConcept]:
    """Deduplicate matched candidates across matchers without truncating any code."""
    by_key: dict[tuple[str, str], CandidateConcept] = {}
    for cs in candidate_sets:
        for candidate in cs.candidates:
            key = (candidate.vocabulary.casefold(), candidate.candidate_id)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate.model_copy(deep=True)
                continue
            channels = list(dict.fromkeys([*existing.channels, *candidate.channels]))
            evidence = _dedupe_evidence(
                [*existing.retrieval_evidence, *candidate.retrieval_evidence]
            )
            by_key[key] = existing.model_copy(
                update={"channels": channels, "retrieval_evidence": evidence}
            )
    return sorted(by_key.values(), key=lambda c: (c.vocabulary.casefold(), c.candidate_id))


def _dedupe_evidence(items: Sequence[RetrievalEvidence]) -> list[RetrievalEvidence]:
    seen: list[RetrievalEvidence] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _build_coded_concepts(
    candidates: Sequence[CandidateConcept],
    *,
    canonical_by_key: dict[tuple[str, str], CandidateConcept],
    links: LinkSet | None,
    retrieval_text_from_model: bool,
) -> list[CodedConcept]:
    links_by_code: dict[tuple[str, str], list[CodeLink]] = {}
    if links is not None:
        for link in links.links:
            links_by_code.setdefault(
                (link.source.vocabulary.casefold(), link.source.concept_code), []
            ).append(link)

    codes: list[CodedConcept] = []
    for candidate in candidates:
        key = (candidate.vocabulary.casefold(), candidate.candidate_id)
        enriched = canonical_by_key.get(key)
        preferred = enriched if enriched is not None else candidate
        evidence = list(candidate.retrieval_evidence)
        if enriched is not None:
            evidence = _dedupe_evidence([*evidence, *enriched.retrieval_evidence])
        channels: list[RetrievalChannel] = list(
            dict.fromkeys([*candidate.channels, *(enriched.channels if enriched else [])])
        )
        codes.append(
            CodedConcept(
                vocabulary=preferred.vocabulary,
                concept_code=preferred.candidate_id,
                label=preferred.preferred_label,
                concept_class=preferred.concept_class,
                status=preferred.status,
                channels=channels,
                retrieval_evidence=evidence,
                target_links=list(links_by_code.get(key, ())),
                from_model_suggestion=(
                    retrieval_text_from_model or any(ev.from_model_suggestion for ev in evidence)
                ),
            )
        )
    return codes


def _assemble_artifact(
    *,
    run: RunContext,
    sau: SemanticAlignmentUnit,
    codes: list[CodedConcept],
    uncoded: bool,
    coded_by_tier: str | None,
    cleaned_by: list[str],
    attempts: list[CodingTierAttempt],
    links: LinkSet | None,
    provenance: Provenance,
) -> CodingArtifact:
    artifact_id = CodingArtifact.compute_artifact_id(
        run_id=run.run_id,
        sau=sau,
        codes=codes,
        uncoded=uncoded,
        coded_by_tier=coded_by_tier,
        cleaned_by=cleaned_by,
        attempts=attempts,
        links=links,
        provenance=provenance,
    )
    return CodingArtifact(
        artifact_id=artifact_id,
        run_id=run.run_id,
        mapping_key=sau.mapping_key,
        sau=sau,
        codes=codes,
        uncoded=uncoded,
        coded_by_tier=coded_by_tier,
        cleaned_by=cleaned_by,
        attempts=attempts,
        links=links,
        provenance=provenance,
    )
