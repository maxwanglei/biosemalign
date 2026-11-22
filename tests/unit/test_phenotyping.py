"""Unit tests for the WP6 phenotyping workflow, expanders, selectors, and guarantees."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from tests.scenarios_support import appendix_a_phenotyping_profile, build_appendix_a_bundle
from tests.support import synthetic_settings

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.enums import RelationshipType, RoutingStatus, ValidationStatus
from biosemalign.exceptions import AdjudicationFailure, ConfigurationError, WorkflowViolation
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.modules import build
from biosemalign.modules.expand.hierarchy import HierarchyExpander
from biosemalign.modules.expand.lexical import LexicalNeighborsExpander
from biosemalign.modules.select.hierarchy import HierarchySelector
from biosemalign.profiles.schema import TaskProfile
from biosemalign.run_manifest import build_run_manifest
from biosemalign.schemas.expansion import ExpansionSet
from biosemalign.schemas.phenotyping import ConceptSetMember, ConceptSetProposal, ConceptSetReview
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.storage.cache import ResponseCache
from biosemalign.testing import ScriptedModelExpander, ScriptedModelSelector, scripted_model_service
from biosemalign.util import utc_now
from biosemalign.workflows.phenotyping import (
    PhenotypingWorkflow,
    apply_concept_set_review,
    export_concept_set,
)


@pytest.fixture(scope="module")
def pheno_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    return build_appendix_a_bundle(tmp_path_factory.mktemp("pheno_wf"))


def test_hierarchy_and_lexical_expanders_ancestors_siblings_and_limits(
    pheno_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = pheno_bundle
    profile = appendix_a_phenotyping_profile()
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        run = pipeline.new_run()
        # Expand from D2 with ancestors + siblings enabled.
        sau = pipeline.build_sau(pipeline.make_request("severe toy fever"), run)[0]
        anchor = pipeline.knowledge.provider_for("ToyDisease").lookup("D2")
        assert anchor is not None

        hier = HierarchyExpander(
            pipeline.knowledge,
            profile,
            include_ancestors=True,
            include_siblings=True,
        )
        exp = hier.expand(sau, [anchor])
        proposal = HierarchySelector(pipeline.knowledge, profile).select(sau, exp)
        by_code = {m.concept_code: m for m in proposal.members}
        assert by_code["D2"].relation is RelationshipType.EXACT
        assert by_code["D1"].relation is RelationshipType.TARGET_BROADER_THAN_SOURCE
        assert by_code["D1"].scope == "broad"
        assert by_code["D3"].relation is RelationshipType.RELATED_NOT_EQUIVALENT
        assert by_code["D3"].scope == "broad"

        # Lexical neighbors expander finds lexically close ToyDisease concepts.
        lex_exp = LexicalNeighborsExpander(pipeline.knowledge, profile, method="trigram", top_k=5)
        lex_set = lex_exp.expand(sau, [anchor])
        assert {m.concept.candidate_id for m in lex_set.members} >= {"D1", "D3"}
        lex_prop = HierarchySelector(pipeline.knowledge, profile).select(sau, lex_set)
        lex_by_code = {m.concept_code: m for m in lex_prop.members}
        assert lex_by_code["D1"].relation is RelationshipType.RELATED_NOT_EQUIVALENT
        assert lex_by_code["D1"].scope == "broad"

        # Limits.members bounds expansion and sets truncated=True.
        bounded_profile = profile.model_copy(
            update={"limits": profile.limits.model_copy(update={"members": 1})}
        )
        bounded = HierarchyExpander(pipeline.knowledge, bounded_profile).expand(
            pipeline.build_sau(pipeline.make_request("toy fever"), pipeline.new_run())[0],
            [pipeline.knowledge.provider_for("ToyDisease").lookup("D1")],  # type: ignore[list-item]
        )
        assert len(bounded.members) == 1
        assert bounded.truncated is True

        # Registered via operator registry.
        op_ctx = pipeline._op_context
        assert isinstance(build("expand", "hierarchy", op_ctx, {}), HierarchyExpander)
        assert isinstance(
            build("expand", "lexical_neighbors", op_ctx, {"method": "bm25"}),
            LexicalNeighborsExpander,
        )
        assert isinstance(build("select", "hierarchy", op_ctx, {}), HierarchySelector)


class _RogueSelector:
    op_id = "rogue_selector"
    version = "0.1.0"
    uses_model = False
    requires_network = False
    options: ClassVar[dict[str, Any]] = {}

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(backend="test", model_name="rogue@0.1.0")

    def select(self, sau: SemanticAlignmentUnit, expansion: ExpansionSet) -> ConceptSetProposal:
        del expansion
        return ConceptSetProposal(
            sau_id=sau.sau_id,
            idea=sau.mention.text,
            target_terminology=sau.task.target_terminology,
            level=sau.task.desired_granularity,
            members=[
                ConceptSetMember(
                    vocabulary="ToyDisease",
                    concept_code="G1",
                    label="toy fever disorders",
                    relation=RelationshipType.RELATED_NOT_EQUIVALENT,
                    scope="broad",
                    reason="Injected outside expansion.",
                )
            ],
            requested_human_review=True,
        )


def test_phenotyping_guarantees_g2_g4_m6_and_manifest(
    pheno_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = pheno_bundle
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    # G2: prohibited relationship removes members after expand.
    payload = appendix_a_phenotyping_profile().model_dump(mode="json")
    payload["knowledge"]["allowed_relationships"] = []
    payload["knowledge"]["prohibited_relationships"] = ["is_a"]
    g2_profile = TaskProfile.model_validate(payload)
    with BioSemAlign(
        g2_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as g2_pipe:
        g2_art = g2_pipe.propose_concept_set(g2_pipe.make_request("toy fever"))
        assert g2_art.expansion.members == []
        assert [m.concept_code for m in g2_art.decision.members] == ["D1"]
        assert verify_artifact(g2_art, profile=g2_profile).valid

    # G4: rogue selector naming code outside expansion is rejected & sanitized.
    profile = appendix_a_phenotyping_profile()
    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipe:
        base = pipe.phenotyping_runner
        assert base is not None
        rogue_wf = PhenotypingWorkflow(
            profile=profile,
            knowledge=pipe.knowledge,
            coding_cascade=base.coding_cascade,
            expanders=base.expanders,
            selector=_RogueSelector(),
        )
        run = pipe.new_run()
        sau = pipe.build_sau(pipe.make_request("toy fever"), run)[0]
        rogue_art = rogue_wf.propose(sau, run)
        assert rogue_art.validation.status is ValidationStatus.REJECTED
        assert any(f.code == "MEMBER_OUTSIDE_EXPANSION" for f in rogue_art.validation.findings)
        assert rogue_art.decision.members == []
        assert rogue_art.decision.routing_status is RoutingStatus.HUMAN_REVIEW
        assert verify_artifact(rogue_art, profile=profile).valid

        # M6: failing model selector produces empty set routed for human review.
        failing_svc = scripted_model_service(AdjudicationFailure("bad json"))
        model_wf = PhenotypingWorkflow(
            profile=profile,
            knowledge=pipe.knowledge,
            coding_cascade=base.coding_cascade,
            expanders=base.expanders,
            selector=ScriptedModelSelector(pipe.knowledge, profile, failing_svc),
            model_services=(failing_svc,),
        )
        run2 = pipe.new_run()
        sau2 = pipe.build_sau(pipe.make_request("toy fever"), run2)[0]
        m6_art = model_wf.propose(sau2, run2)
        assert any(f.code == "MODEL_SELECTOR_UNAVAILABLE" for f in m6_art.validation.findings)
        assert m6_art.decision.members == []
        assert m6_art.decision.routing_status is RoutingStatus.HUMAN_REVIEW

        # Unapproved review refuses export (G8).
        declined = ConceptSetReview(
            review_id="rev-no",
            artifact_id=m6_art.artifact_id,
            reviewer="reviewer",
            approved=False,
        )
        declined_art = apply_concept_set_review(m6_art, declined, knowledge=pipe.knowledge)
        with pytest.raises(WorkflowViolation, match="approval"):
            export_concept_set(declined_art, name="never")

        # G10 + M4: model-backed expander propagates from_model_suggestion warning.
        exp_svc = scripted_model_service({"queries": ["recurrent toy fever"]})
        exp_wf = PhenotypingWorkflow(
            profile=profile,
            knowledge=pipe.knowledge,
            coding_cascade=base.coding_cascade,
            expanders=(ScriptedModelExpander(pipe.knowledge, profile, exp_svc),),
            selector=HierarchySelector(pipe.knowledge, profile),
            model_services=(exp_svc,),
        )
        run3 = pipe.new_run()
        sau3 = pipe.build_sau(pipe.make_request("toy fever"), run3)[0]
        exp_art = exp_wf.propose(sau3, run3)
        d3 = next(m for m in exp_art.decision.members if m.concept_code == "D3")
        assert d3.from_model_suggestion is True
        assert any(f.code == "MODEL_SUGGESTED_MEMBER" for f in exp_art.validation.findings)

        manifest_obj = build_run_manifest(
            pipeline=pipe,
            run_id="pheno-run",
            started_at=utc_now(),
            input_sha256="b" * 64,
            output_paths=[],
            record_counts={"proposed": 1},
        )
        assert manifest_obj.workflow is not None
        assert manifest_obj.model is not None

        with pytest.raises(ConfigurationError, match="workflow='phenotyping'"):
            BioSemAlign(
                profile,
                settings=settings,
                cache=ResponseCache(cache_dir, mode=settings.mode),
                workflow=lambda ctx: ctx,  # type: ignore[arg-type,return-value]
            )
