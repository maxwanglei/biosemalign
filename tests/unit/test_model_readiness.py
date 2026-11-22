"""Unit and architecture tests for WP8 model readiness (§5.10, M1–M9, G9–G10)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest
import respx
from tests.scenarios_support import (
    appendix_a_drug_profile,
    appendix_a_llm_profile,
    build_appendix_a_bundle,
)
from tests.support import synthetic_settings

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.enums import (
    DecisionType,
    RelationshipType,
    ResourceMode,
    RoutingStatus,
    ValidationStatus,
)
from biosemalign.evaluation import GoldMapping, evaluate_predictions
from biosemalign.exceptions import AdjudicationFailure, CacheMiss
from biosemalign.kernel import Cascade, Tier
from biosemalign.modules import OpContext, build
from biosemalign.modules.model import load_prompt
from biosemalign.profiles.schema import TaskProfile
from biosemalign.run_manifest import build_run_manifest
from biosemalign.schemas.decision import SelectedConcept
from biosemalign.storage.cache import ResponseCache
from biosemalign.testing import (
    ScriptedModelCheck,
    ScriptedModelCleaner,
    ScriptedModelExpander,
    ScriptedModelMatcher,
    scripted_model_service,
)
from biosemalign.util import utc_now

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "biosemalign"


def test_no_operator_bypasses_model_service() -> None:
    """AST enforcement: only ModelService calls generate_structured; only llm_backend uses httpx for LLM."""
    checked_roots = ("modules", "domains", "kernel", "workflows", "testing")
    direct_generate_calls: list[str] = []
    direct_llm_http: list[str] = []

    for sub in checked_roots:
        root = SRC_ROOT / sub
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(SRC_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "generate_structured"
                    and rel != "modules/model.py"
                ):
                    direct_generate_calls.append(f"{rel}:{node.lineno}")
                if (
                    isinstance(node, ast.Import)
                    and any(alias.name == "httpx" for alias in node.names)
                    and rel
                    not in {
                        "modules/resolve/llm_backend.py",
                        "domains/drug/providers/rxnav.py",
                    }
                ):
                    direct_llm_http.append(f"{rel}:{node.lineno}")

    assert direct_generate_calls == []
    assert direct_llm_http == []


def test_scripted_model_matcher_and_expander_enforce_m5_m6_and_g10(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    payload = appendix_a_drug_profile(
        desired_granularity="ingredient", include_cleaned_tier=False
    ).model_dump(mode="json")
    payload["task"]["relation_types"] = [
        *payload["task"]["relation_types"],
        "RELATED_NOT_EQUIVALENT",
    ]
    payload["context"] = {
        "fields": [
            {"name": "route", "kind": "attribute", "use": ["key", "match", "expand"]},
            {"name": "indication", "kind": "context", "use": [], "blinded": True},
        ]
    }
    profile = TaskProfile.model_validate(payload)

    # 1. ScriptedModelMatcher finds T100 via model suggestion -> G10 forces HUMAN_REVIEW.
    matcher_svc = scripted_model_service({"queries": ["toyprolol"]}, model_name="match-model")

    def match_wf(ctx: OpContext) -> Cascade:
        return Cascade(
            [
                Tier(
                    name="m_match",
                    match=ScriptedModelMatcher(ctx.knowledge, ctx.profile, matcher_svc),
                    linker=build("link", "standard", ctx, {}),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    cache_dir = tmp_path / "cache1"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=match_wf,
        model_service=matcher_svc,
    ) as pl:
        art = pl.align_artifact(
            pl.make_request("ABBREV", route="oral", indication="secret", extra="noise")
        )
    matcher_prompt = matcher_svc.backend.messages[0][1]["content"]  # type: ignore[attr-defined]
    assert "route: oral" in matcher_prompt
    assert "secret" not in matcher_prompt
    assert "noise" not in matcher_prompt
    t100 = art.knowledge_package.candidate_by_qualified_id("RxNorm", "T100")
    assert t100 is not None and all(e.from_model_suggestion for e in t100.retrieval_evidence)
    assert art.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert art.decision.provenance.workflow is not None
    assert art.decision.provenance.workflow.tiers[0].operators[0].uses_model is True
    assert art.decision.provenance.model_calls[0].operator_kind == "match"
    assert verify_artifact(art, profile=profile).valid

    # 2. ScriptedModelExpander adds model-suggested member -> marked & flagged HUMAN_REVIEW,
    #    and G2 strips prohibited relationships if emitted.
    exp_svc = scripted_model_service({"queries": ["toyprolol"]}, model_name="expand-model")

    def exp_wf(ctx: OpContext) -> Cascade:
        return Cascade(
            [
                Tier(
                    name="m_expand",
                    match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    expander=ScriptedModelExpander(ctx.knowledge, ctx.profile, exp_svc),
                    linker=build("link", "standard", ctx, {}),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=exp_wf,
        model_service=exp_svc,
    ) as pl2:
        art_exp = pl2.align_artifact(
            pl2.make_request("toyprol", route="oral", indication="secret", extra="noise")
        )
    exp_prompt = exp_svc.backend.messages[0][1]["content"]  # type: ignore[attr-defined]
    assert "route: oral" in exp_prompt
    assert "secret" not in exp_prompt
    assert "noise" not in exp_prompt
    t100_exp = art_exp.knowledge_package.candidate_by_qualified_id("RxNorm", "T100")
    assert t100_exp is not None and all(
        e.from_model_suggestion for e in t100_exp.retrieval_evidence
    )
    assert art_exp.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert art_exp.decision.provenance.model_calls[0].operator_kind == "expand"
    assert verify_artifact(art_exp, profile=profile).valid

    # 3. M6: unusable replies in clean / expand / check never crash batch.
    fail_clean = scripted_model_service(AdjudicationFailure("bad clean"))
    fail_exp = scripted_model_service(AdjudicationFailure("bad expand"))
    fail_chk = scripted_model_service(AdjudicationFailure("bad check"))

    def m6_wf(ctx: OpContext) -> Cascade:
        return Cascade(
            [
                Tier(
                    name="m6",
                    cleaners=(ScriptedModelCleaner(ctx.profile, fail_clean),),
                    match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    expander=ScriptedModelExpander(ctx.knowledge, ctx.profile, fail_exp),
                    linker=build("link", "standard", ctx, {}),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=m6_wf,
    ) as pl3:
        pl3.validator = type(pl3.validator)(
            profile,
            provider=pl3.knowledge.primary_provider(),
            checks=[ScriptedModelCheck(profile, fail_chk)],
            model_service=fail_chk,
        )
        pl3.kernel.validator = pl3.validator
        art_m6 = pl3.align_artifact(pl3.make_request("toyprolol"))

    assert art_m6.decision.decision is DecisionType.MAPPED
    assert any(f.code == "MODEL_CHECK_UNAVAILABLE" for f in art_m6.validation.findings)
    assert art_m6.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert [c.operator_kind for c in art_m6.decision.provenance.model_calls] == [
        "clean",
        "expand",
        "check",
    ]


def test_g10_refuses_auto_accept_even_if_resolver_claims_exact(tmp_path: Path) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    profile = appendix_a_drug_profile(desired_granularity="ingredient", include_cleaned_tier=False)
    cleaner_svc = scripted_model_service({"queries": ["toyprolol"]})

    class _ExactClaimResolver:
        op_id = "claim_exact"
        version = "0.1.0"
        uses_model = False
        requires_network = False

        def resolve(self, sau, package):  # type: ignore[no-untyped-def]
            from biosemalign.schemas.decision import AlignmentProposal

            cand = package.candidate_by_qualified_id("RxNorm", "T100")
            assert cand is not None
            sel = SelectedConcept(
                concept_id=cand.candidate_id,
                label=cand.preferred_label,
                vocabulary=cand.vocabulary,
                concept_class=cand.concept_class,
                status=cand.status,
                relationship_to_source=RelationshipType.EXACT,
            )
            return AlignmentProposal(
                sau_id=sau.sau_id,
                decision=DecisionType.MAPPED,
                selected_concepts=[sel],
                source_level_concept=sel,
                analysis_level_concept=sel,
                relationship_to_source=RelationshipType.EXACT,
                rationale="claimed exact",
            )

        def provenance(self):  # type: ignore[no-untyped-def]
            from biosemalign.schemas.provenance import ModelProvenance

            return ModelProvenance(backend="claim", model_name="claim@0.1.0")

    def wf(ctx: OpContext) -> Cascade:
        return Cascade(
            [
                Tier(
                    name="model_clean",
                    cleaners=(ScriptedModelCleaner(ctx.profile, cleaner_svc),),
                    match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    linker=build("link", "standard", ctx, {}),
                    resolver=_ExactClaimResolver(),
                )
            ]
        )

    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=wf,
        model_service=cleaner_svc,
    ) as pl:
        art = pl.align_artifact(pl.make_request("TPL"))
    assert art.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert art.validation.status is not ValidationStatus.REJECTED


@respx.mock
def test_worked_examples_change_prompt_digest_and_cache_key(tmp_path: Path) -> None:
    ex_file = tmp_path / "examples.txt"
    ex_file.write_text("Input: X -> Candidate 1\n", encoding="utf-8")
    p1 = load_prompt("choose_candidate", domain="drug", examples_path=ex_file)
    ex_file.write_text("Input: X -> Candidate 2\n", encoding="utf-8")
    p2 = load_prompt("choose_candidate", domain="drug", examples_path=ex_file)
    assert p1.digest != p2.digest

    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    endpoint = "http://localhost:8000/v1"
    agree = {
        "candidate": 1,
        "abstain_reason": "none",
        "relationship": "TARGET_BROADER_THAN_SOURCE",
        "semantic_loss": ["brand_identity", "route", "salt_form", "strength"],
        "needs_review": False,
        "rationale": "branded tablet",
    }
    respx.post(f"{endpoint}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(agree)}, "finish_reason": "stop"}]
            },
        )
    )
    payload = appendix_a_llm_profile().model_dump(mode="json")
    payload["tiers"][0]["resolve"] = {"op": "llm", "options": {"examples_path": str(ex_file)}}
    cache_dir = tmp_path / "cache"
    live_settings = synthetic_settings(
        cache_dir,
        mode=ResourceMode.CACHE_PREFER,
        deployment_environment="local",
        lexicon_path=bundle,
        lexicon_sha256=manifest.bundle_sha256,
        llm_base_url=endpoint,
        llm_model="local-model",
        llm_model_revision="rev-1",
    )
    with BioSemAlign(
        TaskProfile.model_validate(payload),
        settings=live_settings,
        cache=ResponseCache(cache_dir, mode=ResourceMode.CACHE_PREFER),
    ) as pl:
        pl.align_artifact(pl.make_request("TOYBRAND XL 50 MG", route="oral"))

    ex_file.write_text("Input: X -> Candidate 3\n", encoding="utf-8")
    frozen_settings = synthetic_settings(
        cache_dir,
        mode=ResourceMode.FROZEN,
        deployment_environment="local",
        lexicon_path=bundle,
        lexicon_sha256=manifest.bundle_sha256,
        llm_base_url=endpoint,
        llm_model="local-model",
        llm_model_revision="rev-1",
    )
    with (
        BioSemAlign(
            TaskProfile.model_validate(payload),
            settings=frozen_settings,
            cache=ResponseCache(cache_dir, mode=ResourceMode.FROZEN),
        ) as pl_frozen,
        pytest.raises(CacheMiss),
    ):
        pl_frozen.align_artifact(pl_frozen.make_request("TOYBRAND XL 50 MG", route="oral"))


def test_evaluation_report_tracks_model_agreement_per_tier(tmp_path: Path) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    profile = appendix_a_llm_profile()
    agree = {
        "candidate": 1,
        "abstain_reason": "none",
        "relationship": "TARGET_BROADER_THAN_SOURCE",
        "semantic_loss": ["brand_identity", "route", "salt_form", "strength"],
        "needs_review": False,
        "rationale": "ok",
    }
    disagree = {**agree, "semantic_loss": ["strength"], "rationale": "wrong loss"}
    service = scripted_model_service([agree, disagree])
    from biosemalign.modules.resolve.leashed import LeashedLLMResolver

    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        resolver=LeashedLLMResolver(profile, service),
        model_service=service,
    ) as pl:
        records = list(
            pl.align_batch_records(
                [
                    {"drug": "TOYBRAND XL 50 MG", "route": "oral", "record_id": "c1"},
                    {"drug": "TOYBRAND XL 50 MG", "route": "oral", "record_id": "c2"},
                ],
                value_column="drug",
                id_columns=["record_id"],
                context_columns=["route"],
            )
        )
        run_manifest = build_run_manifest(
            pipeline=pl,
            run_id="eval-run",
            started_at=utc_now(),
            input_sha256="eval-in",
            output_paths=[],
            record_counts={"artifacts": 2},
        )
    assert run_manifest.model_calls == 2
    assert run_manifest.model_cache_hits == 0
    a1, a2 = (rec.artifact for rec in records)
    assert a1 is not None and a2 is not None

    gold = [
        GoldMapping(case_id="c1", source_value="TOYBRAND XL 50 MG", expected_concept_ids=["T100"]),
        GoldMapping(case_id="c2", source_value="TOYBRAND XL 50 MG", expected_concept_ids=["T100"]),
    ]
    from biosemalign.evaluation import EvaluationPrediction

    report = evaluate_predictions(
        gold, [EvaluationPrediction.from_artifact(a1), EvaluationPrediction.from_artifact(a2)]
    )
    assert "llm_tier" in report.model_agreement
    metrics = report.model_agreement["llm_tier"]
    assert metrics.calls == 2
    assert metrics.evaluated_calls == 2
    assert metrics.agreements == 1
    assert metrics.disagreements == 1
    assert metrics.agreement_rate == 0.5
    assert report.as_dict()["model_agreement"]["llm_tier"]["agreement_rate"] == 0.5
