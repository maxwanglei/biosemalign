"""Regression tests for the 15 reported audit issues."""

from __future__ import annotations

import csv
import importlib.metadata
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.scenarios_support import (
    appendix_a_drug_profile,
    appendix_a_llm_profile,
    appendix_a_phenotyping_profile,
    appendix_a_pipeline,
    appendix_a_toy_disease_hierarchy_profile,
    appendix_a_toy_disease_profile,
    build_appendix_a_bundle,
)
from tests.support import synthetic_settings

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.enums import (
    DecisionType,
    RelationshipType,
    RetrievalChannel,
    RoutingStatus,
    ValidationStatus,
)
from biosemalign.kernel import Cascade, Tier
from biosemalign.kernel.guards import resolver_view
from biosemalign.knowledge.lexicon.build import LexiconDeclaration, build_lexicon
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules import OpContext, build, register
from biosemalign.modules.base import CandidateSet, Linker
from biosemalign.modules.clean import AdapterCleaner, CleanerResult
from biosemalign.modules.model import ModelService
from biosemalign.modules.registry import default_matcher_for_domain, unregister
from biosemalign.modules.resolve.deterministic import DeterministicResolver
from biosemalign.modules.resolve.leashed import LeashedLLMResolver
from biosemalign.plugins import ensure_plugins_loaded, without_domain
from biosemalign.profiles.schema import (
    ContextFieldConfig,
    LimitsConfig,
    TaskProfile,
)
from biosemalign.run_manifest import build_run_manifest
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.phenotyping import ConceptSetArtifact, ConceptSetReview, MemberReviewAction
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.storage.cache import ResponseCache
from biosemalign.targets import builtin_specs, register_target_spec, unregister_target_spec
from biosemalign.testing import (
    ScriptedLLMBackend,
    ScriptedModelCleaner,
    ScriptedModelMatcher,
    scripted_model_service,
)
from biosemalign.util import lexical_similarity, utc_now
from biosemalign.workflows.coding import CodingWorkflow, code_terms, coding_workflow_from_profile
from biosemalign.workflows.coding.workflow import (
    CodingCascade,
    CodingTier,
    _default_coding_tiers,
)
from biosemalign.workflows.mapping import _default_tiers, workflow_from_profile
from biosemalign.workflows.phenotyping import apply_concept_set_review, export_concept_set


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_custom_bundle(
    root: Path,
    *,
    vocabularies: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    concept_fields: Sequence[str] = (
        "vocabulary_id",
        "concept_code",
        "concept_name",
        "concept_class",
        "status",
    ),
    synonyms: list[dict[str, Any]] = (),
    relationships: list[dict[str, Any]] = (),
) -> tuple[Path, Any]:
    src_dir = root / "src"
    tables = src_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "concepts.csv", concepts, concept_fields)
    if synonyms:
        _write_csv(tables / "synonyms.csv", synonyms, ("vocabulary_id", "concept_code", "synonym"))
    if relationships:
        _write_csv(
            tables / "relationships.csv",
            relationships,
            ("vocabulary_id_1", "concept_code_1", "predicate", "vocabulary_id_2", "concept_code_2"),
        )
    bundle_dir = root / "bundle"
    manifest = build_lexicon(
        tables,
        bundle_dir,
        declaration=LexiconDeclaration.model_validate({"vocabularies": vocabularies}),
        overwrite=True,
    )
    return bundle_dir, manifest


def test_issue_01_plugin_entry_points_and_builtin_specs_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_plugins_loaded(reload_entry_points=True)
    scan_calls = 0
    real_entry_points = importlib.metadata.entry_points

    def counting_entry_points(*args: Any, **kwargs: Any) -> Any:
        nonlocal scan_calls
        scan_calls += 1
        return real_entry_points(*args, **kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", counting_entry_points)

    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    profile = appendix_a_drug_profile()
    pipeline = appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile)
    try:
        pipeline.align_artifact(pipeline.make_request("toyprolol"))
        pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG"))
        ensure_plugins_loaded()
        _ = profile.target_spec()
        assert scan_calls == 0

        ensure_plugins_loaded(reload_entry_points=True)
        assert scan_calls == 1
    finally:
        pipeline.close()

    first = builtin_specs()
    second = builtin_specs()
    assert first is second

    custom_spec = first["toydisease"].model_copy(update={"spec_id": "audit_custom"})
    register_target_spec(custom_spec)
    try:
        third = builtin_specs()
        assert third is not first
        assert "audit_custom" in third
    finally:
        unregister_target_spec("audit_custom")

    fourth = builtin_specs()
    assert "audit_custom" not in fourth


def test_issue_02_plain_hierarchy_expansion_does_not_outrank_exact_anchor(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    base = appendix_a_toy_disease_profile().model_dump(mode="json")
    base["knowledge"] = {
        "resources": ["ToyDisease"],
        "allowed_relationships": ["is_a", "in_group"],
        "prohibited_relationships": [],
    }
    base["tiers"] = [
        {
            "name": "exact_expand",
            "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
            "expand": {"op": "hierarchy", "options": {}},
        }
    ]
    profile = TaskProfile.model_validate(base)
    pipeline = appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile)
    try:
        artifact = pipeline.align_artifact(pipeline.make_request("toy fever"))
    finally:
        pipeline.close()

    candidate_ids = [c.candidate_id for c in artifact.knowledge_package.candidates]
    assert candidate_ids[0] == "D1"
    assert set(candidate_ids) == {"D1", "D2", "D3"}
    assert artifact.proposal.decision == DecisionType.MAPPED
    assert artifact.proposal.relationship_to_source == RelationshipType.EXACT
    assert artifact.proposal.selected_concepts[0].concept_id == "D1"
    assert artifact.decision.routing_status == RoutingStatus.AUTO_ACCEPT


def test_issue_03_g10_model_mark_preserved_across_refinement_and_phenotyping(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    drug_profile = appendix_a_drug_profile(include_cleaned_tier=False)
    clean_svc = scripted_model_service({"queries": ["toybrand xl 50 mg"]}, model_name="clean-mod")
    settings = synthetic_settings(
        tmp_path / "cache_map", lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    def map_workflow(ctx: OpContext) -> Cascade:
        return Cascade(
            [
                Tier(
                    name="m_clean",
                    cleaners=(
                        ScriptedModelCleaner(ctx.profile, clean_svc),
                        build("clean", "drug.extract_strength", ctx, {}),
                    ),
                    match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    expander=build("expand", "refine_by_slots", ctx, {}),
                    linker=build("link", "standard", ctx, {}),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    map_pipeline = BioSemAlign(
        drug_profile,
        settings=settings,
        cache=ResponseCache(tmp_path / "cache_map", mode=settings.mode),
        workflow=map_workflow,
        model_service=clean_svc,
    )
    try:
        map_artifact = map_pipeline.align_artifact(map_pipeline.make_request("TOYBRAND-XL 50MG"))
    finally:
        map_pipeline.close()

    assert map_artifact.sau.mention.retrieval_text_from_model is True
    assert map_artifact.proposal.source_level_concept is not None
    assert map_artifact.proposal.source_level_concept.concept_id == "T200"
    assert map_artifact.proposal.selected_concepts[0].concept_id == "T100"
    t200 = map_artifact.knowledge_package.candidate_by_qualified_id("RxNorm", "T200")
    assert t200 is not None
    assert any(ev.from_model_suggestion for ev in t200.retrieval_evidence)
    assert map_artifact.decision.routing_status == RoutingStatus.HUMAN_REVIEW

    # Phenotyping: ScriptedModelCleaner and ScriptedModelMatcher preserve model marks through HierarchyExpander.
    pheno_clean_svc = scripted_model_service({"queries": ["toy fever"]}, model_name="pheno-clean")
    pheno_match_svc = scripted_model_service({"queries": ["toy fever"]}, model_name="pheno-match")
    register(
        "clean",
        "audit.model_cleaner",
        lambda ctx, opts: ScriptedModelCleaner(ctx.profile, pheno_clean_svc),
    )
    register(
        "match",
        "audit.model_matcher",
        lambda ctx, opts: ScriptedModelMatcher(
            ctx.knowledge, ctx.profile, pheno_match_svc, vocabularies=["ToyDisease"]
        ),
    )
    try:
        base_pheno = appendix_a_phenotyping_profile().model_dump(mode="json")

        clean_pheno_payload = {
            **base_pheno,
            "tiers": [
                {
                    "name": "model_clean_tier",
                    "clean": [{"op": "audit.model_cleaner", "options": {}}],
                    "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
                    "expand": [{"op": "hierarchy", "options": {}}],
                    "select": {"op": "hierarchy", "options": {}},
                }
            ],
        }
        p1 = appendix_a_pipeline(
            tmp_path / "p1",
            bundle,
            manifest,
            profile=TaskProfile.model_validate(clean_pheno_payload),
        )
        try:
            cs1 = p1.propose_concept_set(p1.make_request("TFVR"))
        finally:
            p1.close()
        assert {m.concept_code for m in cs1.proposal.members} == {"D1", "D2", "D3"}
        assert all(m.from_model_suggestion for m in cs1.proposal.members)
        assert any(
            r.startswith("MODEL_SUGGESTED_MEMBER:") and "ToyDisease:D1" in r
            for r in cs1.decision.routing_reasons
        )
        assert any(
            r.startswith("MODEL_SUGGESTED_MEMBER:") and "ToyDisease:D2" in r
            for r in cs1.decision.routing_reasons
        )
        assert any(
            r.startswith("MODEL_SUGGESTED_MEMBER:") and "ToyDisease:D3" in r
            for r in cs1.decision.routing_reasons
        )
        assert cs1.decision.routing_status == RoutingStatus.HUMAN_REVIEW

        match_pheno_payload = {
            **base_pheno,
            "tiers": [
                {
                    "name": "model_match_tier",
                    "clean": [],
                    "match": [{"op": "audit.model_matcher", "options": {}}],
                    "expand": [{"op": "hierarchy", "options": {}}],
                    "select": {"op": "hierarchy", "options": {}},
                }
            ],
        }
        p2 = appendix_a_pipeline(
            tmp_path / "p2",
            bundle,
            manifest,
            profile=TaskProfile.model_validate(match_pheno_payload),
        )
        try:
            cs2 = p2.propose_concept_set(p2.make_request("TFVR"))
        finally:
            p2.close()
        assert {m.concept_code for m in cs2.proposal.members} == {"D1", "D2", "D3"}
        assert all(m.from_model_suggestion for m in cs2.proposal.members)
        assert any(
            r.startswith("MODEL_SUGGESTED_MEMBER:") and "ToyDisease:D1" in r
            for r in cs2.decision.routing_reasons
        )
        assert any(
            r.startswith("MODEL_SUGGESTED_MEMBER:") and "ToyDisease:D2" in r
            for r in cs2.decision.routing_reasons
        )
        assert any(
            r.startswith("MODEL_SUGGESTED_MEMBER:") and "ToyDisease:D3" in r
            for r in cs2.decision.routing_reasons
        )
        assert cs2.decision.routing_status == RoutingStatus.HUMAN_REVIEW
    finally:
        unregister("clean", "audit.model_cleaner")
        unregister("match", "audit.model_matcher")


def test_issue_04_g3_resolver_isolation_scrubs_source_row_and_stage_methods(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    payload = appendix_a_drug_profile().model_dump(mode="json")
    for field in payload["context"]["fields"]:
        if field["name"] == "indication":
            field["blinded"] = True
            field["use"] = []
    payload["tiers"][0]["resolve"] = {"op": "rules", "options": {}}
    profile = TaskProfile.model_validate(payload)

    class _CapturingResolver:
        op_id = "audit.capturing"
        version = "0.1.0"
        uses_model = False
        requires_network = False

        def __init__(self, prof: TaskProfile) -> None:
            self._delegate = DeterministicResolver(prof)
            self.seen_saus: list[SemanticAlignmentUnit] = []

        def provenance(self) -> ModelProvenance:
            return self._delegate.provenance()

        def resolve(
            self, sau: SemanticAlignmentUnit, package: KnowledgePackage
        ) -> AlignmentProposal:
            self.seen_saus.append(sau)
            return self._delegate.resolve(sau, package)

    capturing = _CapturingResolver(profile)
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    pipeline = BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        resolver=capturing,
    )
    try:
        req = pipeline.make_request(
            "toyprolol", indication="hypercalcemia", analyte="calcium", route="oral"
        )
        [sau] = pipeline.build_sau(req)
        direct_view = resolver_view(sau, profile)
        assert direct_view.context.indication is None
        assert "indication" not in direct_view.source.row
        assert "indi_pt" not in direct_view.source.row
        assert "analyte" not in direct_view.source.row
        assert "analyte" not in direct_view.source.extensions
        assert direct_view.source.row.get("route") == "oral"

        pkg = pipeline.build_package(sau)
        _ = pipeline.resolve(sau, pkg)
        artifact = pipeline.align_artifact(req)
    finally:
        pipeline.close()

    assert len(capturing.seen_saus) == 2
    for seen in capturing.seen_saus:
        assert seen.context.indication is None
        assert "indication" not in seen.source.row
        assert "analyte" not in seen.source.row
        assert "analyte" not in seen.source.extensions
        assert seen.source.row.get("route") == "oral"

    assert artifact.sau.source.row.get("indication") == "hypercalcemia"
    assert artifact.sau.source.row.get("analyte") == "calcium"


def test_issue_05_phenotyping_review_tamper_detection_and_export_digest(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    profile = appendix_a_phenotyping_profile()
    pipeline = appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile)
    try:
        artifact = pipeline.propose_concept_set(pipeline.make_request("toy fever"))
        assert pipeline.knowledge is not None
        review_excl = ConceptSetReview(
            review_id="REV-EXCL-D3",
            artifact_id=artifact.artifact_id,
            reviewer="clinician",
            approved=True,
            actions=[
                MemberReviewAction(
                    action="exclude",
                    vocabulary="ToyDisease",
                    concept_code="D3",
                    reason="exclude recurrent",
                )
            ],
        )
        reviewed_excl = apply_concept_set_review(
            artifact, review_excl, knowledge=pipeline.knowledge
        )
        assert [m.concept_code for m in reviewed_excl.approved_members] == ["D1", "D2"]

        tampered = reviewed_excl.model_dump(mode="json")
        d3_member = next(m for m in artifact.decision.members if m.concept_code == "D3").model_dump(
            mode="json"
        )
        tampered["approved_members"].append(d3_member)
        with pytest.raises(ValidationError):
            ConceptSetArtifact.model_validate(tampered)
        assert not verify_artifact(tampered, profile=profile).valid

        review_all = ConceptSetReview(
            review_id="REV-ALL",
            artifact_id=artifact.artifact_id,
            reviewer="clinician",
            approved=True,
            actions=[],
        )
        reviewed_all = apply_concept_set_review(artifact, review_all, knowledge=pipeline.knowledge)
    finally:
        pipeline.close()

    exported_excl = export_concept_set(reviewed_excl, name="toy_fever_excl")
    exported_all = export_concept_set(reviewed_all, name="toy_fever_all")
    assert exported_excl.artifact_digest == reviewed_excl.compute_digest()
    assert exported_excl.artifact_digest != artifact.artifact_id
    assert exported_excl.artifact_digest != exported_all.artifact_digest
    assert verify_artifact(exported_excl).valid


def test_issue_06_semantic_checks_match_resolver_context_purpose(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")

    for use_list in (["key", "resolve"], ["key", "check"]):
        payload = appendix_a_drug_profile().model_dump(mode="json")
        for field in payload["context"]["fields"]:
            if field["name"] == "route":
                field["use"] = use_list
        profile = TaskProfile.model_validate(payload)
        pipeline = appendix_a_pipeline(
            tmp_path / "_".join(use_list), bundle, manifest, profile=profile
        )
        try:
            art = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG", route="oral"))
        finally:
            pipeline.close()
        assert art.validation.status != ValidationStatus.REJECTED
        codes = {f.code for f in art.validation.findings}
        assert "UNEXPECTED_SEMANTIC_LOSS" not in codes
        assert "UNREPORTED_SEMANTIC_LOSS" not in codes

    hier_payload = appendix_a_toy_disease_hierarchy_profile().model_dump(mode="json")
    hier_payload["context"]["fields"] = [
        ContextFieldConfig(name="route", kind="attribute", use=["key", "resolve"]).model_dump(
            mode="json"
        )
    ]
    hier_profile = TaskProfile.model_validate(hier_payload)
    hier_pipeline = appendix_a_pipeline(tmp_path / "hier", bundle, manifest, profile=hier_profile)
    try:
        hier_art = hier_pipeline.align_artifact(
            hier_pipeline.make_request("toy fever", route="oral")
        )
    finally:
        hier_pipeline.close()
    assert hier_art.validation.status != ValidationStatus.REJECTED
    hier_codes = {f.code for f in hier_art.validation.findings}
    assert "UNEXPECTED_SEMANTIC_LOSS" not in hier_codes
    assert "UNREPORTED_SEMANTIC_LOSS" not in hier_codes


def test_issue_07_memo_hit_replays_cleaned_view_without_calling_model(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    profile = appendix_a_drug_profile(include_cleaned_tier=False)
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    backend = ScriptedLLMBackend([{"queries": ["toyprolol"]}])
    svc = ModelService(
        backend=backend,
        model_name="scripted-clean",
        model_revision="1",
    )

    def clean_workflow(ctx: OpContext) -> Cascade:
        return Cascade(
            [
                Tier(
                    name="m_clean",
                    cleaners=(ScriptedModelCleaner(ctx.profile, svc),),
                    match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    linker=build("link", "standard", ctx, {}),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    pipeline = BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=clean_workflow,
        model_service=svc,
    )
    pipeline.attach_memo(tmp_path / "memo.sqlite")
    try:
        first = pipeline.align_artifact(pipeline.make_request("TOY-PROLOL"))
        second = pipeline.align_artifact(pipeline.make_request("TOY-PROLOL"))
    finally:
        pipeline.close()

    assert len(backend.messages) == 1
    assert first.proposal.decision == DecisionType.MAPPED
    assert second.proposal.decision == DecisionType.MAPPED
    assert first.sau.mention.retrieval_text == "toyprolol"
    assert second.sau.mention.retrieval_text == "toyprolol"
    assert second.sau.mention.retrieval_text_from_model is True
    assert second.decision.routing_status == RoutingStatus.HUMAN_REVIEW
    assert second.decision.provenance.memo is not None
    assert len(second.decision.provenance.model_calls) == 1
    assert verify_artifact(second, profile=profile).valid


def test_issue_08_undeclared_date_ignored_and_declared_date_separates_batch_cache(
    tmp_path: Path,
) -> None:
    bundle, manifest = _build_custom_bundle(
        tmp_path / "custom",
        vocabularies=[{"vocabulary_id": "ToyVocab", "release": "1", "license": "synthetic"}],
        concepts=[
            {
                "vocabulary_id": "ToyVocab",
                "concept_code": "V1",
                "concept_name": "WidgetX",
                "concept_class": "Entity",
                "status": "active",
                "valid_start_date": "2000-01-01",
                "valid_end_date": "2010-12-31",
            }
        ],
        concept_fields=(
            "vocabulary_id",
            "concept_code",
            "concept_name",
            "concept_class",
            "status",
            "valid_start_date",
            "valid_end_date",
        ),
    )

    def make_profile(*, declare_date: bool) -> TaskProfile:
        fields = (
            [ContextFieldConfig(name="date", kind="context", use=["match"]).model_dump(mode="json")]
            if declare_date
            else []
        )
        return TaskProfile.model_validate(
            {
                "profile_name": f"date_test_{declare_date}",
                "profile_version": "0.1.0",
                "domain": None,
                "workflow": "mapping",
                "task": {"operation": "test", "mapping_intent": "exact", "relation_types": []},
                "target": {"terminology": "ToyVocab", "spec": None, "level": None},
                "knowledge": {"resources": ["ToyVocab"]},
                "context": {"fields": fields},
                "tiers": [
                    {
                        "name": "exact_date",
                        "match": [
                            {
                                "op": "exact",
                                "options": {
                                    "vocabularies": ["ToyVocab"],
                                    "respect_validity_dates": True,
                                },
                            }
                        ],
                    }
                ],
            }
        )

    p_undeclared = appendix_a_pipeline(
        tmp_path / "undecl", bundle, manifest, profile=make_profile(declare_date=False)
    )
    try:
        rows_u = list(
            p_undeclared.align_batch_records(
                [
                    {"id": "1", "term": "WidgetX", "date": "2024-01-01"},
                    {"id": "2", "term": "WidgetX", "date": "2005-01-01"},
                ],
                value_column="term",
                id_columns=["id"],
                context_columns=["date"],
            )
        )
    finally:
        p_undeclared.close()
    assert all(
        r.artifact is not None and r.artifact.proposal.decision == DecisionType.MAPPED
        for r in rows_u
    )

    p_declared = appendix_a_pipeline(
        tmp_path / "decl", bundle, manifest, profile=make_profile(declare_date=True)
    )
    try:
        rows_d = list(
            p_declared.align_batch_records(
                [
                    {"id": "1", "term": "WidgetX", "date": "2005-01-01"},
                    {"id": "2", "term": "WidgetX", "date": "2024-01-01"},
                ],
                value_column="term",
                id_columns=["id"],
                context_columns=["date"],
            )
        )
    finally:
        p_declared.close()
    assert rows_d[0].artifact is not None and rows_d[1].artifact is not None
    assert rows_d[0].artifact.sau.mapping_key != rows_d[1].artifact.sau.mapping_key
    assert rows_d[0].artifact.proposal.decision == DecisionType.MAPPED
    assert rows_d[1].artifact.proposal.decision == DecisionType.NO_VALID_MAPPING


def test_issue_09_unique_refinement_survives_rrf_candidate_limit_and_custom_drop(
    tmp_path: Path,
) -> None:
    concepts: list[dict[str, Any]] = [
        {
            "vocabulary_id": "RxNorm",
            "concept_code": "IN1",
            "concept_name": "betax",
            "concept_class": "IN",
            "status": "active",
        },
        {
            "vocabulary_id": "RxNorm",
            "concept_code": "BN1",
            "concept_name": "BetaBrand",
            "concept_class": "BN",
            "status": "active",
        },
        {
            "vocabulary_id": "RxNorm",
            "concept_code": "SBD1",
            "concept_name": "BetaBrand 50 MG Oral Tablet",
            "concept_class": "SBD",
            "status": "active",
        },
    ]
    for i in range(10):
        concepts.append(
            {
                "vocabulary_id": "RxNorm",
                "concept_code": f"N{i}",
                "concept_name": f"NoiseDrug {i}",
                "concept_class": "IN",
                "status": "active",
            }
        )
    bundle, manifest = _build_custom_bundle(
        tmp_path / "rrf_bundle",
        vocabularies=[{"vocabulary_id": "RxNorm", "release": "1", "license": "synthetic"}],
        concepts=concepts,
        relationships=[
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD1",
                "predicate": "has_tradename",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "BN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD1",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "BN1",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
        ],
    )

    class _ScriptedNoiseMatcher:
        version = "0.1"
        uses_model = False
        requires_network = False

        def __init__(self, op_id: str, knowledge: KnowledgeRouter, include_bn: bool) -> None:
            self.op_id = op_id
            self.knowledge = knowledge
            self.include_bn = include_bn

        def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
            store = self.knowledge.lexicon
            prov = self.knowledge.provider_for("RxNorm")
            assert store is not None and prov is not None
            cands: list[CandidateConcept] = []
            if self.include_bn:
                cands.extend(
                    prov.candidates(
                        store.exact("BetaBrand", ["RxNorm"]),
                        channel=RetrievalChannel.EXACT_LOOKUP,
                        query="BetaBrand",
                    )
                )
            for i in range(10):
                hits = store.exact(f"NoiseDrug {i}", ["RxNorm"])
                cands.extend(
                    prov.candidates(
                        hits, channel=RetrievalChannel.EXACT_LOOKUP, query=f"NoiseDrug {i}"
                    )
                )
            return CandidateSet(matcher=self.op_id, candidates=cands)

    class _DroppingLinker:
        op_id = "audit.dropper"
        version = "0.1"
        uses_model = False
        requires_network = False

        def __init__(self, inner: Linker) -> None:
            self._inner = inner

        def link(
            self,
            sau: SemanticAlignmentUnit,
            candidate_sets: Sequence[CandidateSet],
        ) -> KnowledgePackage:
            pkg = self._inner.link(sau, candidate_sets)
            return pkg.model_copy(
                update={"candidates": [c for c in pkg.candidates if c.candidate_id != "SBD1"]}
            )

    profile = appendix_a_drug_profile().model_copy(
        update={"limits": LimitsConfig(candidates=10, members=100)}
    )
    settings = synthetic_settings(
        tmp_path / "c1", lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    from biosemalign.kernel.tiers import ParallelUnion

    def wf_rrf_two(ctx: OpContext) -> Cascade:
        assert ctx.knowledge is not None
        return Cascade(
            [
                Tier(
                    name="rrf_refine",
                    cleaners=(build("clean", "drug.extract_strength", ctx, {}),),
                    match=ParallelUnion(
                        (
                            _ScriptedNoiseMatcher("m1", ctx.knowledge, True),
                            _ScriptedNoiseMatcher("m2", ctx.knowledge, False),
                        )
                    ),
                    expander=build("expand", "refine_by_slots", ctx, {}),
                    linker=build("link", "standard", ctx, {"merge": "rrf"}),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    p1 = BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(tmp_path / "c1", mode=settings.mode),
        workflow=wf_rrf_two,
    )
    try:
        art = p1.align_artifact(p1.make_request("BetaBrand 50 MG"))
    finally:
        p1.close()
    assert len(art.knowledge_package.candidates) == 10
    assert art.knowledge_package.candidate_by_qualified_id("RxNorm", "SBD1") is not None
    assert art.knowledge_package.refinement is not None
    assert art.knowledge_package.refinement.status == "unique"
    assert art.proposal.source_level_concept is not None
    assert art.proposal.source_level_concept.concept_id == "SBD1"

    def wf_drop(ctx: OpContext) -> Cascade:
        assert ctx.knowledge is not None
        return Cascade(
            [
                Tier(
                    name="drop_refine",
                    cleaners=(build("clean", "drug.extract_strength", ctx, {}),),
                    match=_ScriptedNoiseMatcher("m1", ctx.knowledge, True),
                    expander=build("expand", "refine_by_slots", ctx, {}),
                    linker=_DroppingLinker(build("link", "standard", ctx, {"merge": "rrf"})),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    s2 = synthetic_settings(
        tmp_path / "c2", lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    p2 = BioSemAlign(
        profile,
        settings=s2,
        cache=ResponseCache(tmp_path / "c2", mode=s2.mode),
        workflow=wf_drop,
    )
    try:
        art_drop = p2.align_artifact(p2.make_request("BetaBrand 50 MG"))
    finally:
        p2.close()
    assert art_drop.knowledge_package.refinement is not None
    assert art_drop.knowledge_package.refinement.status == "none"
    assert art_drop.knowledge_package.refinement.selected is None


def test_issue_10_refinement_status_computed_before_member_limit_truncation(
    tmp_path: Path,
) -> None:
    bundle, manifest = _build_custom_bundle(
        tmp_path / "multi_sbd",
        vocabularies=[{"vocabulary_id": "RxNorm", "release": "1", "license": "synthetic"}],
        concepts=[
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "IN1",
                "concept_name": "toyprolol",
                "concept_class": "IN",
                "status": "active",
            },
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "BN1",
                "concept_name": "Toybrand",
                "concept_class": "BN",
                "status": "active",
            },
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "SBD1",
                "concept_name": "Toybrand 50 MG Oral Tablet",
                "concept_class": "SBD",
                "status": "active",
            },
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "SBD2",
                "concept_name": "Toybrand 50 MG Oral Capsule",
                "concept_class": "SBD",
                "status": "active",
            },
        ],
        relationships=[
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD1",
                "predicate": "has_tradename",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "BN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD2",
                "predicate": "has_tradename",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "BN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD1",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD2",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "BN1",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
        ],
    )
    payload = appendix_a_drug_profile(include_cleaned_tier=True).model_dump(mode="json")
    payload["limits"] = {"candidates": 10, "members": 1, "enrichment_batch_size": 3}
    payload["tiers"] = [payload["tiers"][1]]
    profile = TaskProfile.model_validate(payload)
    pipeline = appendix_a_pipeline(tmp_path / "p", bundle, manifest, profile=profile)
    try:
        art = pipeline.align_artifact(pipeline.make_request("Toybrand 50 MG"))
    finally:
        pipeline.close()

    assert art.knowledge_package.refinement is not None
    assert art.knowledge_package.refinement.status == "multiple"
    assert art.knowledge_package.refinement.selected is None
    assert art.proposal.source_level_concept is not None
    assert art.proposal.source_level_concept.concept_id == "BN1"


def test_issue_11_bulk_coding_passes_raw_sample_text_to_case_sensitive_cleaners(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    csv_path = tmp_path / "terms.csv"
    _write_csv(csv_path, [{"term": "TOPROL-XL"}], ("term",))

    cleaner = AdapterCleaner(
        lambda text: CleanerResult(canonical_text="toyprolol" if "TOPROL-XL" in text else None),
        op_id="test.upper_brand",
        version="1.0",
    )
    report = code_terms(
        bundle,
        csv_path,
        tmp_path / "bulk_out",
        text_column="term",
        vocabularies=["RxNorm"],
        target="RxNorm",
        granularity="ingredient",
        cleaners=[cleaner],
    )
    assert report.coded_terms == 1
    assert report.coded_terms_cleaned == 1

    # Compare against per-SAU CodingWorkflow with the same cleaner.
    profile = appendix_a_drug_profile(include_cleaned_tier=False).model_copy(
        update={"workflow": "coding"}
    )
    settings = synthetic_settings(
        tmp_path / "cache", lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    pipeline = BioSemAlign(
        profile, settings=settings, cache=ResponseCache(tmp_path / "cache", mode=settings.mode)
    )
    try:
        [sau] = pipeline.build_sau(pipeline.make_request("TOPROL-XL"))
        assert pipeline.knowledge is not None
        ctx = OpContext(profile=profile, settings=settings, knowledge=pipeline.knowledge)
        cw = CodingWorkflow(
            profile=profile,
            cascade=CodingCascade(
                [
                    CodingTier(
                        name="exact",
                        match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    ),
                    CodingTier(
                        name="cleaned",
                        cleaners=(cleaner,),
                        match=build("match", "exact", ctx, {"vocabularies": ["RxNorm"]}),
                    ),
                ]
            ),
            knowledge=pipeline.knowledge,
        )
        code_run = pipeline.new_run(run_id=sau.provenance.run_id)
        code_run.source_adapter = sau.provenance.source_adapter
        code_run.source_adapter_version = sau.provenance.source_adapter_version
        per_sau = cw.code(sau, code_run)
    finally:
        pipeline.close()

    assert [c.concept_code for c in per_sau.codes] == ["T100"]


def test_issue_12_model_service_does_not_accumulate_records_across_batch(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    profile = appendix_a_llm_profile()
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    cache = ResponseCache(cache_dir, mode=settings.mode)
    backend = ScriptedLLMBackend(
        {
            "candidate": 1,
            "abstain_reason": "none",
            "relationship": "EXACT",
            "semantic_loss": [],
            "needs_review": False,
            "rationale": "exact ingredient match",
        }
    )
    svc = ModelService(
        backend=backend,
        model_name="scripted-llm",
        model_revision="1",
    )
    resolver = LeashedLLMResolver(profile, svc)
    pipeline = BioSemAlign(
        profile,
        settings=settings,
        cache=cache,
        resolver=resolver,
        model_service=svc,
    )
    try:
        run_ctx = pipeline.new_run()
        rows = list(
            pipeline.align_batch_records(
                [{"drug": "toyprolol"}, {"drug": "alphazine"}, {"drug": "betazine"}],
                value_column="drug",
                run=run_ctx,
            )
        )
        manifest_obj = build_run_manifest(
            pipeline=pipeline,
            run_id=run_ctx.run_id,
            started_at=utc_now(),
            input_sha256="0" * 64,
            output_paths=[],
            record_counts={"rows": 3},
        )
    finally:
        pipeline.close()

    assert len(rows) == 3
    for row in rows:
        assert row.artifact is not None
        assert row.artifact.proposal.decision == DecisionType.MAPPED
        assert len(row.artifact.decision.provenance.model_calls) == 1
    assert svc.record_count == 0
    assert svc._records == []
    assert manifest_obj.model_calls == 3


def test_issue_13_unique_refinement_preserves_anchor_similarity_against_rejection_floor(
    tmp_path: Path,
) -> None:
    long_label = "24 HR betax hydrochloride 50 MG Extended Release Oral Tablet"
    assert lexical_similarity("BetaBrand 50MG", long_label) < 0.30
    bundle, manifest = _build_custom_bundle(
        tmp_path / "long_sbd",
        vocabularies=[{"vocabulary_id": "RxNorm", "release": "1", "license": "synthetic"}],
        concepts=[
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "IN1",
                "concept_name": "betax",
                "concept_class": "IN",
                "status": "active",
            },
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "BN1",
                "concept_name": "BetaBrand",
                "concept_class": "BN",
                "status": "active",
            },
            {
                "vocabulary_id": "RxNorm",
                "concept_code": "SBD1",
                "concept_name": long_label,
                "concept_class": "SBD",
                "status": "active",
            },
        ],
        relationships=[
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD1",
                "predicate": "has_tradename",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "BN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "SBD1",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "BN1",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "IN1",
            },
        ],
    )
    payload = appendix_a_drug_profile(include_cleaned_tier=True).model_dump(mode="json")
    payload["tiers"] = [payload["tiers"][1]]
    payload["routing"]["rejection_lexical_similarity"] = 0.30
    profile = TaskProfile.model_validate(payload)
    pipeline = appendix_a_pipeline(tmp_path / "p_rules", bundle, manifest, profile=profile)
    try:
        art = pipeline.align_artifact(pipeline.make_request("BetaBrand 50MG"))
    finally:
        pipeline.close()

    assert art.proposal.decision == DecisionType.MAPPED
    assert art.proposal.source_level_concept is not None
    assert art.proposal.source_level_concept.concept_id == "SBD1"
    assert art.proposal.selected_concepts[0].concept_id == "IN1"

    # Also verify LeashedLLMResolver projection preserves the anchor against the floor.
    cache_dir = tmp_path / "cache_llm"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    cache = ResponseCache(cache_dir, mode=settings.mode)
    backend = ScriptedLLMBackend(
        {
            "candidate": 1,
            "abstain_reason": "none",
            "relationship": "TARGET_BROADER_THAN_SOURCE",
            "semantic_loss": ["brand_identity", "strength", "dose_form", "specificity"],
            "needs_review": False,
            "rationale": "refined product",
        }
    )
    svc = ModelService(backend=backend, model_name="scripted-llm", model_revision="1")
    p_llm = BioSemAlign(
        profile,
        settings=settings,
        cache=cache,
        resolver=LeashedLLMResolver(profile, svc),
        model_service=svc,
    )
    try:
        art_llm = p_llm.align_artifact(p_llm.make_request("BetaBrand 50MG"))
    finally:
        p_llm.close()

    assert art_llm.proposal.decision == DecisionType.MAPPED
    assert art_llm.proposal.source_level_concept is not None
    assert art_llm.proposal.source_level_concept.concept_id == "SBD1"


def test_issue_14_core_workflows_and_bulk_coding_decoupled_from_drug_and_rxnorm(
    tmp_path: Path,
) -> None:
    profile = TaskProfile.model_validate(
        {
            "profile_name": "no_drug_test",
            "profile_version": "0.1.0",
            "domain": "drug",
            "workflow": "mapping",
            "task": {"operation": "test", "mapping_intent": "exact", "relation_types": []},
            "target": {"terminology": "RxNorm", "spec": None, "level": None},
            "tiers": None,
        }
    )
    settings = synthetic_settings(tmp_path / "cache")
    ctx = OpContext(profile=profile, settings=settings, knowledge=None)
    with without_domain("drug"):
        assert default_matcher_for_domain("drug") == "exact"
        default_map = _default_tiers(ctx)
        default_code = _default_coding_tiers(ctx)
        assert [op.op for op in default_map[0].match] == ["exact"]
        assert [op.op for op in default_code[0].match] == ["exact"]
        with pytest.raises(Exception, match="exact"):
            workflow_from_profile(ctx)
        with pytest.raises(Exception, match="exact"):
            coding_workflow_from_profile(ctx)

    toy_bundle, _ = _build_custom_bundle(
        tmp_path / "toy_only",
        vocabularies=[{"vocabulary_id": "ToyDisease", "release": "1", "license": "synthetic"}],
        concepts=[
            {
                "vocabulary_id": "ToyDisease",
                "concept_code": "D1",
                "concept_name": "toy fever",
                "concept_class": "Disease",
                "status": "active",
            }
        ],
    )
    csv_path = tmp_path / "toy_terms.csv"
    _write_csv(csv_path, [{"term": "toy fever"}], ("term",))
    report = code_terms(
        toy_bundle,
        csv_path,
        tmp_path / "toy_out",
        text_column="term",
    )
    assert report.target == "ToyDisease"
    assert report.granularity == "Disease"
    assert report.coded_terms == 1


def test_issue_15_attach_memo_applies_settings_output_permissions(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "bundle")
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir,
        lexicon_path=bundle,
        lexicon_sha256=manifest.bundle_sha256,
        output_dir_mode=0o750,
        output_file_mode=0o640,
    )
    pipeline = BioSemAlign(
        appendix_a_drug_profile(),
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
    )
    memo_dir = tmp_path / "memo_out"
    memo_file = memo_dir / "memo.sqlite"
    try:
        pipeline.attach_memo(memo_file)
    finally:
        pipeline.close()

    assert stat.S_IMODE(memo_dir.stat().st_mode) == 0o750
    assert stat.S_IMODE(memo_file.stat().st_mode) == 0o640
