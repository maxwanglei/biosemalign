"""Composable operators, cascades, and the kernel's non-bypassable guarantees."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.support import (
    ALPHA,
    MISSPELLING,
    NO_MATCH,
    PRODUCT,
    UNSUPPORTED,
    make_pipeline,
)

from biosemalign.enums import (
    Authority,
    DecisionType,
    RetrievalChannel,
    RoutingStatus,
    TierOutcome,
)
from biosemalign.exceptions import AdjudicationFailure, ConfigurationError, WorkflowViolation
from biosemalign.ops import (
    AdapterTransformOp,
    AdjudicatorProposeOp,
    CandidateSet,
    CleanerResult,
    ExtractStrengthOp,
    KnowledgePackageFuseOp,
    OpContext,
    RxNavRetrieveOp,
    UnicodeNormalizeOp,
    available_ops,
    build_op,
    register_op,
)
from biosemalign.pipelines.workflow import Cascade, ParallelUnion, Tier
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.knowledge import KnowledgePackage, KnowledgeStatement
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit


def two_tier_profile(**exact_options: Any) -> TaskProfile:
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["workflow"] = {
        "tiers": [
            {
                "name": "exact",
                "retrievers": [{"op": "rxnav", "options": {"channels": ["normalized_lookup"]}}],
                "proposer": {
                    "op": "adjudicator",
                    "options": {"pass_on": ["NO_VALID_MAPPING"], **exact_options},
                },
            },
            {"name": "full"},
        ]
    }
    return TaskProfile.model_validate(payload)


def reload(profile: TaskProfile) -> TaskProfile:
    """Validate a mutated profile the way the loader would."""
    return TaskProfile.model_validate(profile.model_dump(mode="json"))


class RecordingProposer:
    """Wraps the default proposer and remembers what it was shown."""

    op_id = "recording"
    version = "test"
    requires_network = False

    def __init__(self, inner: AdjudicatorProposeOp) -> None:
        self.inner = inner
        self.seen: list[SemanticAlignmentUnit] = []

    def propose(self, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> Any:
        self.seen.append(sau)
        return self.inner.propose(sau, package)

    def provenance(self) -> ModelProvenance:
        return self.inner.provenance()


def default_tier(context: OpContext, **overrides: Any) -> Tier:
    assert context.knowledge is not None and context.adjudicator is not None
    fields: dict[str, Any] = {
        "name": "only",
        "retrieval": RxNavRetrieveOp(context.knowledge.primary_provider(), context.profile),
        "fuser": KnowledgePackageFuseOp(context.knowledge, context.profile),
        "proposer": AdjudicatorProposeOp(context.adjudicator),
    }
    fields.update(overrides)
    return Tier(**fields)


# -- default workflow --------------------------------------------------------


def test_default_workflow_is_one_tier_and_records_its_resolution(
    artifact: AlignmentArtifact,
) -> None:
    provenance = artifact.decision.provenance
    assert provenance.workflow is not None
    assert [tier.name for tier in provenance.workflow.tiers] == ["default"]
    assert [op.op_id for op in provenance.workflow.tiers[0].operators] == [
        "rxnav",
        "knowledge_package",
        "adjudicator:deterministic",
    ]
    assert [op.op_id for op in provenance.workflow.validators] == ["drug_semantics"]
    assert provenance.resolution is not None
    assert provenance.resolution.resolved_by_tier == "default"
    assert artifact.validation.rules_version == "0.1.0+drug_semantics@0.1.0"


def test_workflow_id_is_stable_and_changes_with_configuration(tmp_path: Path) -> None:
    with make_pipeline(tmp_path / "a") as first, make_pipeline(tmp_path / "b") as second:
        assert first.kernel.workflow_provenance == second.kernel.workflow_provenance
    with make_pipeline(tmp_path / "c", profile=two_tier_profile()) as cascade:
        assert (
            cascade.kernel.workflow_provenance.workflow_id
            != first.kernel.workflow_provenance.workflow_id
        )


# -- cascades ----------------------------------------------------------------


def test_cascade_escalates_misses_and_attributes_the_resolving_tier(tmp_path: Path) -> None:
    with make_pipeline(tmp_path, profile=two_tier_profile()) as pipeline:
        exact = pipeline.align_artifact(pipeline.make_request(ALPHA))
        escalated = pipeline.align_artifact(pipeline.make_request(MISSPELLING))

    assert exact.decision.provenance.resolution.resolved_by_tier == "exact"
    assert exact.decision.routing_status is RoutingStatus.AUTO_ACCEPT

    trace = escalated.decision.provenance.resolution
    assert trace.resolved_by_tier == "full"
    assert [(a.tier, a.outcome) for a in trace.attempts] == [
        ("exact", TierOutcome.PASSED),
        ("full", TierOutcome.RESOLVED),
    ]
    assert trace.attempts[0].reason == "decision:NO_VALID_MAPPING"
    assert escalated.decision.selected_concepts[0].concept_id == "1001"
    channels = {
        c for candidate in escalated.knowledge_package.candidates for c in candidate.channels
    }
    # The exact-only tier found nothing; the full tier's approximate channel did.
    assert channels == {RetrievalChannel.TERMINOLOGY_APPROXIMATE_CURRENT}


def test_exhausted_cascade_abstains_for_human_review(tmp_path: Path) -> None:
    profile = two_tier_profile()
    profile.workflow.tiers[1].proposer.options = {"pass_on": ["NO_VALID_MAPPING"]}
    with make_pipeline(tmp_path, profile=reload(profile)) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(NO_MATCH, route="oral"))
        record = next(pipeline.align_batch_records([{"drug": NO_MATCH}], value_column="drug"))

    trace = artifact.decision.provenance.resolution
    assert trace.resolved_by_tier is None
    assert [a.outcome for a in trace.attempts] == [TierOutcome.PASSED, TierOutcome.PASSED]
    assert artifact.decision.decision is DecisionType.ABSTAINED
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert artifact.proposal.preserved_attributes == {"route": "oral"}
    assert "every workflow tier passed" in artifact.proposal.rationale
    assert artifact.decision.provenance.model.backend == "kernel"
    assert record.status == "success"


def test_parallel_union_fuses_every_retriever(tmp_path: Path) -> None:
    def workflow(context: OpContext) -> Cascade:
        assert context.knowledge is not None
        provider = context.knowledge.primary_provider()
        union = ParallelUnion(
            [
                RxNavRetrieveOp(
                    provider, context.profile, channels=[RetrievalChannel.EXACT_LOOKUP]
                ),
                RxNavRetrieveOp(
                    provider,
                    context.profile,
                    channels=[RetrievalChannel.TERMINOLOGY_APPROXIMATE_CURRENT],
                ),
            ]
        )
        return Cascade([default_tier(context, retrieval=union)])

    with make_pipeline(tmp_path, workflow=workflow) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(PRODUCT))
    operators = artifact.decision.provenance.workflow.tiers[0].operators
    retrievers = [op for op in operators if op.kind == "retrieve"]
    assert [op.options["channels"] for op in retrievers] == [
        ("exact_lookup",),
        ("terminology_approximate_current",),
    ]
    assert artifact.decision.selected_concepts[0].concept_id == "1001"


def test_profile_cascade_matches_programmatic_cascade(tmp_path: Path) -> None:
    def workflow(context: OpContext) -> Cascade:
        tier = default_tier(context, name="default")
        return Cascade([tier])

    with (
        make_pipeline(tmp_path / "profile") as from_profile,
        make_pipeline(tmp_path / "code", workflow=workflow) as from_code,
    ):
        assert (
            from_profile.kernel.workflow_provenance.workflow_id
            == from_code.kernel.workflow_provenance.workflow_id
        )


# -- kernel guarantees -------------------------------------------------------


def test_proposer_cannot_return_a_decision(tmp_path: Path) -> None:
    class Forger:
        op_id = "forger"
        version = "test"
        requires_network = False

        def propose(self, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> Any:
            return {"decision": "MAPPED", "routing_status": "AUTO_ACCEPT"}

        def provenance(self) -> ModelProvenance:
            return ModelProvenance(backend="forger")

    with make_pipeline(
        tmp_path, workflow=lambda context: Cascade([default_tier(context, proposer=Forger())])
    ) as pipeline:
        with pytest.raises(WorkflowViolation, match="may only return"):
            pipeline.align_value(ALPHA)
        record = next(pipeline.align_batch_records([{"drug": ALPHA}], value_column="drug"))
    assert record.status == "error"
    assert record.error_code == "WorkflowViolation"


def test_kernel_strips_prohibited_predicates_a_custom_fuser_adds(tmp_path: Path) -> None:
    class LeakyFuser:
        op_id = "leaky"
        version = "test"

        def __init__(self, inner: KnowledgePackageFuseOp) -> None:
            self.inner = inner

        def fuse(
            self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
        ) -> KnowledgePackage:
            package = self.inner.fuse(sau, candidate_sets)
            leak = KnowledgeStatement(
                subject_id="1001",
                predicate="has_adverse_effect",
                object_id="outcome",
                source_resource="RxNorm",
                authority=Authority.TARGET_TERMINOLOGY,
            )
            package.candidates[0].relationships.append(leak)
            package.kg_guidance.append(leak)
            return package

    def workflow(context: OpContext) -> Cascade:
        assert context.knowledge is not None
        inner = KnowledgePackageFuseOp(context.knowledge, context.profile)
        return Cascade([default_tier(context, fuser=LeakyFuser(inner))])

    with make_pipeline(tmp_path, workflow=workflow) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(ALPHA))
    package = artifact.knowledge_package
    predicates = {s.predicate for c in package.candidates for s in c.relationships}
    predicates |= {s.predicate for s in package.kg_guidance}
    assert "has_adverse_effect" not in predicates
    assert artifact.decision.selected_concepts[0].concept_id == "1001"


def test_transforms_cannot_rewrite_the_source_mention(tmp_path: Path) -> None:
    class Rewriter:
        op_id = "rewriter"
        version = "test"

        def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
            mention = sau.mention.model_copy(update={"text": "something else"})
            return sau.model_copy(update={"mention": mention})

    pipeline = make_pipeline(
        tmp_path,
        workflow=lambda context: Cascade([default_tier(context, transforms=(Rewriter(),))]),
    )
    with pipeline, pytest.raises(WorkflowViolation, match="changed the source mention"):
        pipeline.align_value(ALPHA)


def test_transforms_cannot_overwrite_asserted_attributes(tmp_path: Path) -> None:
    cleaner = AdapterTransformOp(
        lambda text: CleanerResult(canonical_text=text, slots={"route": "topical"}),
        op_id="cleaner",
        version="test",
    )
    with make_pipeline(
        tmp_path, workflow=lambda context: Cascade([default_tier(context, transforms=(cleaner,))])
    ) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(ALPHA, route="oral"))
    # The cleaner's slot was ignored rather than overwriting what the source said.
    assert artifact.decision.preserved_attributes == {"route": "oral"}

    class Overwriter:
        op_id = "overwriter"
        version = "test"

        def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
            attributes = sau.attributes.model_copy(update={"route": "topical"})
            return sau.model_copy(update={"attributes": attributes})

    pipeline = make_pipeline(
        tmp_path / "overwrite",
        workflow=lambda context: Cascade([default_tier(context, transforms=(Overwriter(),))]),
    )
    with pipeline, pytest.raises(WorkflowViolation, match="overwrote or removed"):
        pipeline.align_value(ALPHA, route="oral")


def test_extract_strength_moves_strength_into_a_slot(synthetic_pipeline: Any) -> None:
    sau = synthetic_pipeline.build_sau(synthetic_pipeline.make_request("ExampleAlpha 50 MG"))[0]
    transformed = ExtractStrengthOp().transform(sau)
    assert transformed.mention.text == "ExampleAlpha 50 MG"
    assert transformed.mention.retrieval_text == "ExampleAlpha"
    assert transformed.attributes.strength == "50 MG"

    bare = synthetic_pipeline.build_sau(synthetic_pipeline.make_request("50 MG"))[0]
    assert ExtractStrengthOp().transform(bare) == bare
    assert ExtractStrengthOp().transform(transformed).attributes.strength == "50 MG"


#: "Example Alpha" in full-width letters with a no-break space, as PDF exports produce.
FULLWIDTH = "\uff25\uff58\uff41\uff4d\uff50\uff4c\uff45\u00a0Alpha"


def test_unicode_normalize_canonicalizes_the_query_only(synthetic_pipeline: Any) -> None:
    sau = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(FULLWIDTH))[0]
    transformed = UnicodeNormalizeOp().transform(sau)
    assert transformed.mention.retrieval_text == "Example Alpha"
    assert transformed.mention.text == sau.mention.text
    plain = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(ALPHA))[0]
    assert UnicodeNormalizeOp().transform(plain) == plain


def test_transformed_query_reaches_retrieval_and_slot_reaches_the_decision(
    tmp_path: Path,
) -> None:
    def workflow(context: OpContext) -> Cascade:
        return Cascade([default_tier(context, transforms=(ExtractStrengthOp(),))])

    with make_pipeline(tmp_path, workflow=workflow) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("ExampleAlpha 50 MG"))
    assert artifact.sau.mention.retrieval_text == "ExampleAlpha"
    assert artifact.decision.preserved_attributes == {"strength": "50 MG"}
    assert artifact.decision.selected_concepts[0].concept_id == "1001"


def test_blinded_context_is_withheld_from_proposers(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.knowledge.blinded_context_fields = ["indication"]
    recorder: list[RecordingProposer] = []

    def workflow(context: OpContext) -> Cascade:
        assert context.adjudicator is not None
        proposer = RecordingProposer(AdjudicatorProposeOp(context.adjudicator))
        recorder.append(proposer)
        return Cascade([default_tier(context, proposer=proposer)])

    with make_pipeline(tmp_path, profile=reload(profile), workflow=workflow) as pipeline:
        artifact = pipeline.align_artifact(
            pipeline.make_request(ALPHA, indication="studied outcome", route="oral")
        )
        unblinded = pipeline.align_artifact(pipeline.make_request(ALPHA, route="oral"))
    assert recorder[0].seen[0].context.indication is None
    assert recorder[0].seen[0].context.route == "oral"
    assert artifact.sau.context.indication == "studied outcome"
    assert artifact.decision.provenance.resolution.withheld_context == ("indication",)
    assert unblinded.decision.provenance.resolution.withheld_context == ()


def test_only_unmirrored_context_can_be_blinded() -> None:
    profile = load_profile("faers_drug_normalization")
    payload = profile.model_dump(mode="json")
    payload["knowledge"]["blinded_context_fields"] = ["route"]
    with pytest.raises(ValidationError, match="cannot be blinded"):
        TaskProfile.model_validate(payload)


# -- startup verification and configuration ----------------------------------


def test_frozen_mode_rejects_operators_that_require_the_network(tmp_path: Path) -> None:
    class LiveRetriever:
        op_id = "live"
        version = "test"
        requires_network = True

        def retrieve(self, sau: SemanticAlignmentUnit) -> CandidateSet:
            raise AssertionError("never reached")

    with pytest.raises(ConfigurationError, match="FROZEN mode forbids"):
        make_pipeline(
            tmp_path,
            workflow=lambda context: Cascade([default_tier(context, retrieval=LiveRetriever())]),
        )


def test_proposer_candidate_limit_is_checked_at_startup(tmp_path: Path) -> None:
    def workflow(context: OpContext) -> Cascade:
        assert context.adjudicator is not None
        proposer = AdjudicatorProposeOp(context.adjudicator)
        proposer.max_candidates = 5  # type: ignore[attr-defined]
        return Cascade([default_tier(context, proposer=proposer)])

    with pytest.raises(ConfigurationError, match="accepts at most 5 candidates"):
        make_pipeline(tmp_path, workflow=workflow)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("retrievers", [{"op": "nonexistent"}], "no retrieve operator named"),
        ("retrievers", [{"op": "rxnav", "options": {"chanels": []}}], "invalid options"),
        ("proposer", {"op": "adjudicator", "options": {"pass_on": ["NOT_A_DECISION"]}}, "invalid"),
        ("fuser", {"op": "knowledge_package", "options": {"x": 1}}, "takes no options"),
    ],
)
def test_misconfigured_operators_fail_at_construction(
    tmp_path: Path, section: str, value: Any, message: str
) -> None:
    profile = load_profile("faers_drug_normalization")
    payload = profile.model_dump(mode="json")
    payload["workflow"] = {"tiers": [{"name": "only", section: value}]}
    with pytest.raises(ConfigurationError, match=message):
        make_pipeline(tmp_path, profile=TaskProfile.model_validate(payload))


def test_workflow_profiles_reject_duplicate_and_empty_tiers() -> None:
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["workflow"] = {"tiers": [{"name": "a"}, {"name": "a"}]}
    with pytest.raises(ValidationError, match="unique"):
        TaskProfile.model_validate(payload)
    payload["workflow"] = {"tiers": []}
    with pytest.raises(ValidationError):
        TaskProfile.model_validate(payload)


def test_combinators_validate_their_operators(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="at least one tier"):
        Cascade([])
    with pytest.raises(ConfigurationError, match="at least one retriever"):
        ParallelUnion([])
    with make_pipeline(tmp_path) as pipeline:
        tier = pipeline.workflow.tiers[0]
        with pytest.raises(ConfigurationError, match="does not implement"):
            Tier(name="bad", retrieval=tier.retrieval, fuser=tier.fuser, proposer=object())  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="unique"):
            Cascade([tier, tier])


def test_validation_rules_are_configurable_but_core_checks_remain(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.validation.rules = []
    with make_pipeline(tmp_path, profile=reload(profile)) as pipeline:
        assert pipeline.validator.version == "0.1.0"
        artifact = pipeline.align_artifact(pipeline.make_request(ALPHA))
    assert artifact.decision.provenance.workflow.validators == ()

    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["validation"]["rules"] = ["no_such_rule"]
    with pytest.raises(ConfigurationError, match="no validate operator named"):
        make_pipeline(tmp_path / "unknown", profile=TaskProfile.model_validate(payload))


def test_registry_accepts_third_party_operators_and_checks_protocols() -> None:
    class Shouter:
        op_id = "shout"
        version = "test"

        def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
            return sau

    register_op("transform", "test_shout", lambda context, options: Shouter())
    register_op("retrieve", "test_not_a_retriever", lambda context, options: Shouter())
    context = OpContext(profile=load_profile("faers_drug_normalization"))
    assert "test_shout" in available_ops("transform")
    assert isinstance(build_op("transform", "test_shout", context), Shouter)
    with pytest.raises(ConfigurationError, match="does not implement the RetrieveOp"):
        build_op("retrieve", "test_not_a_retriever", context)
    with pytest.raises(ConfigurationError, match="needs the pipeline's knowledge router"):
        build_op("retrieve", "rxnav", context)
    with pytest.raises(ConfigurationError, match="needs the pipeline's adjudicator"):
        build_op("propose", "adjudicator", context)
    with pytest.raises(ValueError, match="unknown operator kind"):
        register_op("bogus", "x", lambda context, options: Shouter())  # type: ignore[arg-type]


def test_pass_on_review_request_escalates(tmp_path: Path) -> None:
    profile = two_tier_profile(pass_on_review_request=True)
    with make_pipeline(tmp_path, profile=reload(profile)) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(UNSUPPORTED))
    attempts = artifact.decision.provenance.resolution.attempts
    assert [(a.tier, a.outcome, a.reason) for a in attempts] == [
        ("exact", TierOutcome.PASSED, "review_requested"),
        ("full", TierOutcome.RESOLVED, None),
    ]
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW


# -- artifact self-verification ---------------------------------------------


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute the content hash so only the targeted consistency check can fail."""
    from biosemalign.util import stable_id

    payload["artifact_id"] = stable_id(
        "ART",
        {
            name: payload[name]
            for name in ("run_id", "sau", "knowledge_package", "proposal", "validation", "decision")
        },
        length=32,
    )
    return payload


def test_forged_tier_attribution_fails_artifact_validation(tmp_path: Path) -> None:
    with make_pipeline(tmp_path, profile=two_tier_profile()) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(ALPHA))
    payload = artifact.model_dump(mode="json")
    payload["decision"]["provenance"]["resolution"]["attempts"][0]["tier"] = "full"
    payload["decision"]["provenance"]["resolution"]["resolved_by_tier"] = "full"
    with pytest.raises(ValidationError, match="do not follow the recorded workflow order"):
        AlignmentArtifact.model_validate(_resign(payload))

    exhausted = artifact.model_dump(mode="json")
    exhausted["decision"]["provenance"]["resolution"] = {
        "attempts": [
            {"tier": "exact", "outcome": "PASSED", "reason": "x"},
            {"tier": "full", "outcome": "PASSED", "reason": "y"},
        ],
        "resolved_by_tier": None,
    }
    with pytest.raises(ValidationError, match="must abstain and request human review"):
        AlignmentArtifact.model_validate(_resign(exhausted))

    partial = artifact.model_dump(mode="json")
    partial["decision"]["provenance"]["resolution"] = {
        "attempts": [{"tier": "exact", "outcome": "PASSED", "reason": "x"}],
        "resolved_by_tier": None,
    }
    with pytest.raises(ValidationError, match="attempt for every tier"):
        AlignmentArtifact.model_validate(_resign(partial))


def test_resolution_trace_is_internally_coherent() -> None:
    from biosemalign.schemas.provenance import ResolutionTrace

    with pytest.raises(ValidationError, match="only the final tier"):
        ResolutionTrace(
            attempts=[
                {"tier": "a", "outcome": "RESOLVED"},
                {"tier": "b", "outcome": "RESOLVED"},
            ],
            resolved_by_tier="b",
        )
    with pytest.raises(ValidationError, match="must name the tier"):
        ResolutionTrace(attempts=[{"tier": "a", "outcome": "RESOLVED"}], resolved_by_tier="b")
    with pytest.raises(ValidationError, match="cannot name a resolving tier"):
        ResolutionTrace(attempts=[{"tier": "a", "outcome": "PASSED"}], resolved_by_tier="a")


def test_cached_packages_are_reused_per_tier(tmp_path: Path) -> None:
    with make_pipeline(tmp_path, profile=two_tier_profile()) as pipeline:
        records = list(
            pipeline.align_batch_records(
                [{"drug": MISSPELLING}, {"drug": MISSPELLING}], value_column="drug"
            )
        )
    first, second = (record.artifact for record in records)
    assert first.sau.sau_id != second.sau.sau_id
    assert second.knowledge_package.sau_id == second.sau.sau_id
    assert (
        first.decision.provenance.resolution.attempts
        == second.decision.provenance.resolution.attempts
    )


def test_kernel_rejects_operators_that_break_identity_or_type(tmp_path: Path) -> None:
    class Misaddressed:
        op_id = "misaddressed"
        version = "test"

        def __init__(self, inner: KnowledgePackageFuseOp) -> None:
            self.inner = inner

        def fuse(
            self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
        ) -> KnowledgePackage:
            return self.inner.fuse(sau, candidate_sets).model_copy(update={"sau_id": "other"})

    class NotAnSAU:
        op_id = "not_an_sau"
        version = "test"

        def transform(self, sau: SemanticAlignmentUnit) -> Any:
            return {"mention": "text"}

    def misaddressed(context: OpContext) -> Cascade:
        assert context.knowledge is not None
        inner = KnowledgePackageFuseOp(context.knowledge, context.profile)
        return Cascade([default_tier(context, fuser=Misaddressed(inner))])

    pipeline = make_pipeline(tmp_path / "fuser", workflow=misaddressed)
    with pipeline, pytest.raises(WorkflowViolation, match="package for a different SAU"):
        pipeline.align_value(ALPHA)

    pipeline = make_pipeline(
        tmp_path / "transform",
        workflow=lambda context: Cascade([default_tier(context, transforms=(NotAnSAU(),))]),
    )
    with pipeline, pytest.raises(WorkflowViolation, match="not an SAU"):
        pipeline.align_value(ALPHA)


def test_kernel_finalize_requires_matching_package_identity(
    synthetic_pipeline: Any,
) -> None:
    run = synthetic_pipeline.new_run()
    sau = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(ALPHA), run)[0]
    package = synthetic_pipeline.retrieve_knowledge(sau)
    proposal = synthetic_pipeline.adjudicate(sau, package)
    with pytest.raises(ValueError, match="identity does not match"):
        synthetic_pipeline.kernel.finalize(
            sau=sau,
            package=package.model_copy(update={"sau_id": "other"}),
            proposal=proposal,
            run=run,
            model=None,
            resolution=None,
        )


def test_exhausted_cascade_requires_a_representable_abstention(tmp_path: Path) -> None:
    profile = two_tier_profile()
    profile.workflow.tiers[1].proposer.options = {"pass_on": ["NO_VALID_MAPPING"]}
    profile.adjudication.relation_types = [
        relation
        for relation in profile.adjudication.relation_types
        if relation.value != "UNSUPPORTED_MAPPING"
    ]

    class AlwaysPass:
        op_id = "always_pass"
        version = "test"
        requires_network = False

        def propose(self, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> Any:
            from biosemalign.ops import PassToNext

            return PassToNext("no opinion")

        def provenance(self) -> ModelProvenance:
            return ModelProvenance(backend="always_pass")

    pipeline = make_pipeline(
        tmp_path,
        profile=reload(profile),
        workflow=lambda context: Cascade([default_tier(context, proposer=AlwaysPass())]),
    )
    with pipeline, pytest.raises(AdjudicationFailure, match="cannot represent the abstention"):
        pipeline.align_value(ALPHA)


def test_pipeline_rejects_a_cascade_bound_to_another_pipeline(tmp_path: Path) -> None:
    with make_pipeline(tmp_path / "builder") as builder:
        cascade = builder.workflow
    with pytest.raises(ConfigurationError, match="not a built Cascade"):
        make_pipeline(tmp_path / "prebuilt", workflow=cascade)


def test_adapter_transform_fills_only_known_empty_slots(synthetic_pipeline: Any) -> None:
    cleaner = AdapterTransformOp(
        lambda text: CleanerResult(
            canonical_text="  ExampleAlpha  ",
            slots={"strength": "5 MG", "not_a_slot": "x", "extensions": "x"},
        ),
        op_id="cleaner",
        version="test",
    )
    sau = synthetic_pipeline.build_sau(synthetic_pipeline.make_request("EXAMPLEALPHA 5MG"))[0]
    transformed = cleaner.transform(sau)
    assert transformed.mention.retrieval_text == "ExampleAlpha"
    assert transformed.attributes.strength == "5 MG"
    assert transformed.attributes.extensions == {}


def test_package_cache_is_bounded(synthetic_pipeline: Any) -> None:
    from biosemalign.pipelines.kernel import PackageCache

    cache = PackageCache(maximum=1)
    first = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(ALPHA))[0]
    second = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(PRODUCT))[0]
    cache.put("t", first, synthetic_pipeline.retrieve_knowledge(first))
    cache.put("t", second, synthetic_pipeline.retrieve_knowledge(second))
    assert cache.get("t", first) is None
    assert cache.get("t", second) is not None
