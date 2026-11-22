"""WP0 baseline scenario tests against the invented Appendix A terminology."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import respx
from tests.scenarios_support import (
    appendix_a_coding_profile,
    appendix_a_drug_profile,
    appendix_a_llm_profile,
    appendix_a_phenotyping_profile,
    appendix_a_pipeline,
    appendix_a_toy_disease_hierarchy_profile,
    appendix_a_toy_disease_profile,
    build_appendix_a_bundle,
)
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.enums import (
    Authority,
    DecisionType,
    MentionFidelity,
    RelationshipType,
    ResourceMode,
    RoutingStatus,
    SemanticLossKind,
    Severity,
    SourceType,
    TierOutcome,
    ValidationStatus,
)
from biosemalign.exceptions import CacheMiss
from biosemalign.input.context import context_view
from biosemalign.kernel import Cascade, Tier
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.modules import OpContext, build
from biosemalign.modules.model import load_prompt
from biosemalign.modules.resolve.leashed import LeashedLLMResolver
from biosemalign.plugins import without_domain
from biosemalign.profiles.schema import TaskProfile
from biosemalign.run_manifest import build_run_manifest
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.testing import (
    JudgeMembersReply,
    ScriptedLLMBackend,
    ScriptedModelCheck,
    ScriptedModelCleaner,
    ScriptedModelLinker,
    ScriptedModelSelector,
    scripted_model_service,
)
from biosemalign.util import utc_now


@pytest.fixture(scope="module")
def appendix_a_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    return build_appendix_a_bundle(tmp_path_factory.mktemp("appendix_a"))


def test_appendix_a_bundle_builds_and_verifies(
    appendix_a_bundle: tuple[Path, LexiconManifest],
) -> None:
    bundle, manifest = appendix_a_bundle
    assert manifest.row_counts["concepts:RxNorm"] == 7
    assert manifest.row_counts["concepts:ToyDisease"] == 4
    assert manifest.row_counts["relationships"] == 9
    assert manifest.row_counts["synonyms"] >= 14

    info = CliRunner().invoke(
        app, ["lexicon", "info", str(bundle), "--sha256", manifest.bundle_sha256]
    )
    assert info.exit_code == 0, info.output
    assert "integrity verified" in info.output


def test_s1_branded_product_with_route_exact_tier(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile()
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG", route="oral"))

    decision = artifact.decision
    assert decision.decision is DecisionType.MAPPED
    assert decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "T200"
    assert [c.concept_id for c in decision.selected_concepts] == ["T100"]
    assert decision.analysis_level_concept is not None
    assert decision.analysis_level_concept.concept_id == "T100"
    assert {loss.kind for loss in decision.semantic_loss} == {
        SemanticLossKind.STRENGTH,
        SemanticLossKind.SALT_FORM,
        SemanticLossKind.BRAND_IDENTITY,
        SemanticLossKind.ROUTE,
    }
    assert decision.preserved_attributes == {"route": "oral"}
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert decision.provenance.resolution is not None
    assert decision.provenance.resolution.resolved_by_tier == "exact"
    assert verify_artifact(artifact, profile=profile).valid


def test_s2_combination_product_all_components_required(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile()
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("alphabeta"))

    decision = artifact.decision
    assert decision.decision is DecisionType.MULTI_CONCEPT_MAPPED
    assert decision.relationship_to_source is RelationshipType.MULTIPLE_COMPONENTS_REQUIRED
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "T302"
    assert {c.concept_id for c in decision.selected_concepts} == {"T300", "T301"}
    assert all(
        c.relationship_to_source is RelationshipType.COMPONENT_OF
        for c in decision.selected_concepts
    )
    assert decision.analysis_level_concept is None
    assert {loss.kind for loss in decision.semantic_loss} == {
        SemanticLossKind.COMBINATION_COMPONENT
    }
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert decision.provenance.resolution is not None
    assert decision.provenance.resolution.resolved_by_tier == "exact"
    assert verify_artifact(artifact, profile=profile).valid


def test_s5_slot_refinement_recovers_specific_product(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile()
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50MG"))

    assert artifact.sau.mention.text == "TOYBRAND XL 50MG"
    assert artifact.sau.mention.retrieval_text == "TOYBRAND XL"
    decision = artifact.decision
    assert decision.preserved_attributes == {"strength": "50MG"}
    assert decision.provenance.resolution is not None
    attempts = decision.provenance.resolution.attempts
    assert [a.tier for a in attempts] == ["exact", "cleaned"]
    assert decision.provenance.resolution.resolved_by_tier == "cleaned"
    assert decision.decision is DecisionType.MAPPED
    assert decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "T200"
    assert [c.concept_id for c in decision.selected_concepts] == ["T100"]
    assert {loss.kind for loss in decision.semantic_loss} == {
        SemanticLossKind.STRENGTH,
        SemanticLossKind.SALT_FORM,
        SemanticLossKind.BRAND_IDENTITY,
    }
    assert decision.refinement is not None
    assert decision.refinement.status == "unique"
    assert decision.refinement.selected is not None
    assert decision.refinement.selected.concept_code == "T200"
    assert decision.refinement.slots_checked == {"strength": "50MG"}
    assert artifact.knowledge_package.links is not None
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert verify_artifact(artifact, profile=profile).valid


def test_s6_nonexistent_strength_falls_back_to_brand_keeping_strength_slot(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile()
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 75 MG"))

    assert artifact.sau.mention.text == "TOYBRAND XL 75 MG"
    assert artifact.sau.mention.retrieval_text == "TOYBRAND XL"
    decision = artifact.decision
    assert decision.preserved_attributes == {"strength": "75 MG"}
    assert decision.provenance.resolution is not None
    attempts = decision.provenance.resolution.attempts
    assert attempts[0].tier == "exact"
    assert attempts[0].outcome is TierOutcome.PASSED
    assert attempts[0].reason == "no_exact_match"
    assert decision.provenance.resolution.resolved_by_tier == "cleaned"
    assert decision.decision is DecisionType.MAPPED
    assert decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "T201"
    assert [c.concept_id for c in decision.selected_concepts] == ["T100"]
    assert {loss.kind for loss in decision.semantic_loss} == {SemanticLossKind.BRAND_IDENTITY}
    assert decision.refinement is not None
    assert decision.refinement.status == "none"
    assert decision.refinement.slots_checked == {"strength": "75 MG"}
    assert decision.rationale is not None
    assert "no specific concept" in decision.rationale
    assert "75 MG" in decision.rationale
    assert artifact.knowledge_package.links is not None
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert verify_artifact(artifact, profile=profile).valid


def test_abstention_unmappable_term(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile()
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("zzzqqq"))

    decision = artifact.decision
    assert decision.decision is DecisionType.NO_VALID_MAPPING
    assert decision.relationship_to_source is RelationshipType.UNSUPPORTED_MAPPING
    assert decision.selected_concepts == []
    assert decision.routing_status is RoutingStatus.ABSTAIN
    assert verify_artifact(artifact, profile=profile).valid


def test_abstention_exhausted_cascade(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile(include_cleaned_tier=False)
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("zzzqqq"))

    decision = artifact.decision
    assert decision.decision is DecisionType.ABSTAINED
    assert decision.relationship_to_source is RelationshipType.UNSUPPORTED_MAPPING
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert decision.provenance.resolution is not None
    assert decision.provenance.resolution.resolved_by_tier is None
    assert decision.provenance.resolution.attempts[0].reason == "no_exact_match"
    assert verify_artifact(artifact, profile=profile).valid


def test_abstention_insufficient_context(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile(required_context=["route"])
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("toyprolol"))

    decision = artifact.decision
    assert decision.decision is DecisionType.INSUFFICIENT_CONTEXT
    assert decision.relationship_to_source is RelationshipType.INSUFFICIENT_CONTEXT
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert "route" in decision.missing_information
    assert verify_artifact(artifact, profile=profile).valid


def test_abstention_unknown_granularity(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_drug_profile(desired_granularity="unknown_level")
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("toyprolol"))

    decision = artifact.decision
    assert decision.decision is DecisionType.ABSTAINED
    assert decision.relationship_to_source is RelationshipType.UNSUPPORTED_MAPPING
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert decision.validation.status is ValidationStatus.REJECTED
    assert any(f.code == "UNSUPPORTED_GRANULARITY" for f in decision.validation.findings)
    assert artifact.proposal.rationale is not None
    assert "no policy for requested granularity" in artifact.proposal.rationale
    assert verify_artifact(artifact, profile=profile).valid


def test_s3_identity_mapping_no_level_drug_pack_unloaded(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_toy_disease_profile(desired_granularity=None)
    with without_domain("drug"):
        assert "biosemalign.domains.drug" not in sys.modules
        with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
            artifact = pipeline.align_artifact(pipeline.make_request("severe toy fever"))
        assert "biosemalign.domains.drug" not in sys.modules

    decision = artifact.decision
    assert decision.decision is DecisionType.MAPPED
    assert decision.relationship_to_source is RelationshipType.EXACT
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "D2"
    assert [c.concept_id for c in decision.selected_concepts] == ["D2"]
    assert decision.semantic_loss == []
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert verify_artifact(artifact, profile=profile).valid


def test_s4_hierarchy_two_hops_drug_pack_unloaded(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    with without_domain("drug"):
        assert "biosemalign.domains.drug" not in sys.modules
        profile = appendix_a_toy_disease_hierarchy_profile(desired_granularity="Group")
        with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
            artifact = pipeline.align_artifact(pipeline.make_request("severe toy fever"))
        assert "biosemalign.domains.drug" not in sys.modules

    decision = artifact.decision
    assert decision.decision is DecisionType.MAPPED
    assert decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "D2"
    assert [c.concept_id for c in decision.selected_concepts] == ["G1"]
    assert decision.analysis_level_concept is not None
    assert decision.analysis_level_concept.concept_id == "G1"
    assert {loss.kind for loss in decision.semantic_loss} == {SemanticLossKind.SPECIFICITY}
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert verify_artifact(artifact, profile=profile).valid


def test_core_and_hierarchy_do_not_import_drug_pack_in_fresh_process() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = (
        "import sys; "
        "import biosemalign.kernel, biosemalign.modules, biosemalign.workflows.mapping, "
        "biosemalign.domains.hierarchy; "
        "biosemalign.domains.hierarchy.register(); "
        "assert 'biosemalign.domains.drug' not in sys.modules, "
        "[m for m in sys.modules if m.startswith('biosemalign.domains.drug')]"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_s7_declared_extension_context_and_blinding(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    payload = appendix_a_drug_profile(desired_granularity=None).model_dump(mode="json")
    payload["domain"] = None
    payload["context"] = {
        "fields": [
            {"name": "matrix", "kind": "context", "use": ["resolve"]},
            {"name": "indication", "kind": "context", "use": [], "blinded": True},
        ]
    }
    profile = TaskProfile.model_validate(payload)

    seen_saus: list[SemanticAlignmentUnit] = []

    class _SpyResolver:
        op_id = "rules"
        version = "0.3.0"
        uses_model = False
        requires_network = False

        def __init__(self, inner: object) -> None:
            self._inner = inner
            self.options: dict[str, object] = {}

        def resolve(self, sau: SemanticAlignmentUnit, package: object) -> object:
            seen_saus.append(sau)
            return self._inner.resolve(sau, package)  # type: ignore[attr-defined]

        def provenance(self) -> object:
            return self._inner.provenance()  # type: ignore[attr-defined]

    def build_workflow(context: OpContext) -> Cascade:
        assert context.knowledge is not None
        tier_cfg = context.profile.tiers[0]
        return Cascade(
            [
                Tier(
                    name=tier_cfg.name,
                    match=build("match", tier_cfg.match[0].op, context, tier_cfg.match[0].options),
                    linker=build("link", tier_cfg.link.op, context, tier_cfg.link.options),
                    resolver=_SpyResolver(build("resolve", "rules", context, {})),
                )
            ]
        )

    cache_dir = tmp_path / "cache"
    from tests.support import synthetic_settings

    from biosemalign.storage.cache import ResponseCache

    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=build_workflow,
    ) as pipeline:
        req1 = pipeline.make_request(
            "toyprolol",
            matrix="serum",
            indication="studied outcome",
            analyte="plasma",
        )
        req2 = pipeline.make_request(
            "toyprolol",
            matrix="urine",
            indication="studied outcome",
            analyte="plasma",
        )
        req3 = pipeline.make_request(
            "toyprolol",
            matrix="serum",
            indication="other outcome",
            analyte="blood",
        )
        art1 = pipeline.align_artifact(req1)
        art2 = pipeline.align_artifact(req2)
        art3 = pipeline.align_artifact(req3)

    assert art1.sau.context.neighboring_fields == {"analyte": "plasma"}
    assert art1.sau.context.extensions == {"matrix": "serum"}
    assert art1.sau.context.indication == "studied outcome"
    assert context_view(art1.sau, "resolve", profile) == {"matrix": "serum"}
    assert context_view(art1.sau, "key", profile) == {"matrix": "serum"}
    assert art1.mapping_key != art2.mapping_key
    assert art1.mapping_key == art3.mapping_key

    assert seen_saus[0].context.extensions == {"matrix": "serum"}
    assert seen_saus[0].context.neighboring_fields == {}
    assert seen_saus[0].context.indication is None
    assert art1.decision.provenance.resolution is not None
    assert art1.decision.provenance.resolution.withheld_context == ("indication",)
    assert verify_artifact(art1, profile=profile).valid


def test_s8_extracted_terms_fidelity_and_mention_not_in_source(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    base_payload = appendix_a_drug_profile(desired_granularity=None).model_dump(mode="json")
    base_payload["input"]["source_type"] = "extracted_terms"
    profile = TaskProfile.model_validate(base_payload)
    source_payload = {
        "text": "Toyprolol was given to neonates.",
        "terms": [
            {"text": "Toyprolol", "extractor": "ner:spacy", "type": "drug"},
            {"text": "toyprolol", "extractor": "ner:spacy", "type": "drug"},
            {"text": "toyprolol tartrate", "extractor": "ner:spacy", "type": "drug"},
        ],
    }

    with appendix_a_pipeline(tmp_path / "warn", bundle, manifest, profile=profile) as pipeline:
        req = AlignmentRequest(
            source_type=SourceType.EXTRACTED_TERMS,
            value=source_payload,
            task=profile.task.operation,
            mapping_intent=profile.task.mapping_intent,
            target_terminology=profile.target.terminology,
            desired_granularity=profile.target.level,
        )
        run = pipeline.new_run()
        saus = pipeline.build_sau(req, run=run)
        artifacts = [pipeline.kernel.align(sau, run) for sau in saus]

    assert [a.sau.mention.fidelity for a in artifacts] == [
        MentionFidelity.VERBATIM,
        MentionFidelity.NORMALIZED_MATCH,
        MentionFidelity.NOT_FOUND,
    ]
    assert all(a.sau.mention.extractor == "ner:spacy" for a in artifacts)
    assert artifacts[0].decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert artifacts[1].decision.routing_status is RoutingStatus.AUTO_ACCEPT
    not_found_findings = [
        f for f in artifacts[2].validation.findings if f.code == "MENTION_NOT_IN_SOURCE"
    ]
    assert len(not_found_findings) == 1
    assert not_found_findings[0].severity is Severity.WARNING
    assert artifacts[2].decision.relationship_to_source is RelationshipType.EXACT
    assert artifacts[2].decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert all(verify_artifact(a, profile=profile).valid for a in artifacts)

    strict_payload = profile.model_dump(mode="json")
    strict_payload["input"]["mention_not_in_source_severity"] = "error"
    strict_profile = TaskProfile.model_validate(strict_payload)
    with appendix_a_pipeline(
        tmp_path / "err", bundle, manifest, profile=strict_profile
    ) as strict_pl:
        strict_run = strict_pl.new_run()
        saus_strict = strict_pl.build_sau(req, run=strict_run)
        strict_third = strict_pl.kernel.align(saus_strict[2], strict_run)
    assert strict_third.validation.status is ValidationStatus.REJECTED
    assert strict_third.decision.decision is DecisionType.ABSTAINED
    assert any(
        f.code == "MENTION_NOT_IN_SOURCE" and f.severity is Severity.ERROR
        for f in strict_third.validation.findings
    )
    assert verify_artifact(strict_third, profile=strict_profile).valid


def test_s11_leashed_llm_resolver_four_replies_and_declared_context_prompt(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    drug_prompt = load_prompt("choose_candidate", domain="drug")
    core_prompt = load_prompt("choose_candidate", domain=None)
    assert "strength, dose form, salt, brand, route" in drug_prompt.text
    assert "strength" not in core_prompt.text
    assert "dose form" not in core_prompt.text

    payload = appendix_a_llm_profile().model_dump(mode="json")
    payload["context"] = {
        "fields": [
            {"name": "route", "kind": "attribute", "use": ["key", "resolve", "check"]},
            {"name": "matrix", "kind": "context", "use": ["resolve"]},
            {"name": "cohort_note", "kind": "context", "use": ["check"]},
            {"name": "indication", "kind": "context", "use": [], "blinded": True},
        ]
    }
    profile = TaskProfile.model_validate(payload)

    agree = {
        "candidate": 1,
        "abstain_reason": "none",
        "relationship": "TARGET_BROADER_THAN_SOURCE",
        "semantic_loss": ["brand_identity", "route", "salt_form", "strength"],
        "needs_review": False,
        "rationale": "branded tablet for toyprolol",
    }
    disagree = {
        **agree,
        "relationship": "EXACT",
        "semantic_loss": [],
        "rationale": "wrongly claimed exact",
    }
    decline = {
        "candidate": 0,
        "abstain_reason": "no_valid_candidate",
        "relationship": "UNSUPPORTED_MAPPING",
        "semantic_loss": [],
        "needs_review": False,
        "rationale": "none fit",
    }
    unusable = {"candidate": 999, "bad": True}

    backend = ScriptedLLMBackend([agree, disagree, decline, unusable])
    service = scripted_model_service([], model_name="scripted-s11")
    service.backend = backend
    resolver = LeashedLLMResolver(profile, service)

    from tests.support import synthetic_settings

    from biosemalign.storage.cache import ResponseCache

    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        resolver=resolver,
        model_service=service,
    ) as pipeline:
        req_a = pipeline.make_request(
            "TOYBRAND XL 50 MG",
            route="oral",
            matrix="serum",
            cohort_note="check only note",
            indication="blinded outcome",
            undeclared_col="undeclared noise",
        )
        art_a = pipeline.align_artifact(req_a)
        art_b = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG", route="oral"))
        art_c = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG", route="oral"))
        art_d = pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG", route="oral"))

    # Declared context only in the prompt (M5).
    prompt_user = backend.messages[0][1]["content"]
    assert "route: oral" in prompt_user
    assert "matrix: serum" in prompt_user
    assert "check only note" not in prompt_user
    assert "blinded outcome" not in prompt_user
    assert "undeclared noise" not in prompt_user

    # (a) agrees -> mapped, auto-accepted
    assert art_a.decision.decision is DecisionType.MAPPED
    assert [c.concept_id for c in art_a.decision.selected_concepts] == ["T100"]
    assert art_a.decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert "llm" not in art_a.proposal.extensions
    assert art_a.decision.provenance.model_calls[0].agrees_with_terminology is True

    # (b) disagrees (claimed EXACT) -> mapped to T100, sent to review
    assert art_b.decision.decision is DecisionType.MAPPED
    assert [c.concept_id for c in art_b.decision.selected_concepts] == ["T100"]
    assert art_b.decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert art_b.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert art_b.decision.provenance.model_calls[0].agrees_with_terminology is False

    # (c) model declined (0) -> NO_VALID_MAPPING, sent to review under M3 default
    assert art_c.decision.decision is DecisionType.NO_VALID_MAPPING
    assert art_c.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert art_c.decision.provenance.model_calls[0].choice == 0

    # (d) unusable reply -> ABSTAINED, sent to review under M6
    assert art_d.decision.decision is DecisionType.ABSTAINED
    assert art_d.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert art_d.decision.provenance.model_calls[0].error is not None

    for art in (art_a, art_b, art_c, art_d):
        assert verify_artifact(art, profile=profile).valid


def test_s12_model_cleaner_readiness_marks_and_review_routing(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    payload = appendix_a_drug_profile(
        desired_granularity="ingredient", include_cleaned_tier=False
    ).model_dump(mode="json")
    payload["task"]["relation_types"] = [
        *payload["task"]["relation_types"],
        "RELATED_NOT_EQUIVALENT",
    ]
    payload["context"] = {
        "fields": [
            {"name": "sentence", "kind": "context", "use": ["clean"]},
            {"name": "indication", "kind": "context", "use": [], "blinded": True},
        ]
    }
    profile = TaskProfile.model_validate(payload)
    service = scripted_model_service({"queries": ["toyprolol"]}, model_name="scripted-clean")
    backend = service.backend
    assert isinstance(backend, ScriptedLLMBackend)

    def build_workflow(context: OpContext) -> Cascade:
        cfg = context.profile.tiers[0]
        return Cascade(
            [
                Tier(
                    name="model_clean",
                    cleaners=(ScriptedModelCleaner(context.profile, service),),
                    match=build("match", cfg.match[0].op, context, cfg.match[0].options),
                    linker=build("link", cfg.link.op, context, cfg.link.options),
                    resolver=build("resolve", "rules", context, {}),
                )
            ]
        )

    from tests.support import synthetic_settings

    from biosemalign.storage.cache import ResponseCache

    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=build_workflow,
        model_service=service,
    ) as pipeline:
        artifact = pipeline.align_artifact(
            pipeline.make_request(
                "TPL",
                sentence="Patient started TPL yesterday.",
                indication="blinded outcome",
                extra_col="undeclared noise",
            )
        )

    cleaner_prompt = backend.messages[0][1]["content"]
    assert "Patient started TPL yesterday." in cleaner_prompt
    assert "blinded outcome" not in cleaner_prompt
    assert "undeclared noise" not in cleaner_prompt

    assert artifact.sau.mention.text == "TPL"
    assert artifact.sau.mention.retrieval_text == "toyprolol"
    assert artifact.sau.mention.retrieval_text_from_model is True
    t100 = artifact.knowledge_package.candidate_by_qualified_id("RxNorm", "T100")
    assert t100 is not None
    assert t100.retrieval_evidence
    assert all(ev.from_model_suggestion is True for ev in t100.retrieval_evidence)
    assert [c.concept_id for c in artifact.decision.selected_concepts] == ["T100"]
    assert artifact.decision.relationship_to_source is RelationshipType.RELATED_NOT_EQUIVALENT
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert artifact.decision.provenance.model_calls[0].operator_kind == "clean"
    assert verify_artifact(artifact, profile=profile).valid


@respx.mock
def test_s14_http_cache_replay_frozen_and_changed_prompt_refused(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    endpoint = "http://localhost:8000/v1"
    agree = {
        "candidate": 1,
        "abstain_reason": "none",
        "relationship": "TARGET_BROADER_THAN_SOURCE",
        "semantic_loss": ["brand_identity", "route", "salt_form", "strength"],
        "needs_review": False,
        "rationale": "branded tablet for toyprolol",
    }
    http_route = respx.post(f"{endpoint}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(agree)}, "finish_reason": "stop"}]
            },
        )
    )

    payload = appendix_a_llm_profile().model_dump(mode="json")
    payload["tiers"][0]["resolve"] = {"op": "llm"}
    profile_v1 = TaskProfile.model_validate(payload)
    cache_dir = tmp_path / "cache"

    from tests.support import synthetic_settings

    from biosemalign.storage.cache import ResponseCache

    settings_live = synthetic_settings(
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
        profile_v1,
        settings=settings_live,
        cache=ResponseCache(cache_dir, mode=ResourceMode.CACHE_PREFER),
    ) as live_pipeline:
        art1 = live_pipeline.align_artifact(
            live_pipeline.make_request("TOYBRAND XL 50 MG", route="oral")
        )
        manifest1 = build_run_manifest(
            pipeline=live_pipeline,
            run_id="r1",
            started_at=utc_now(),
            input_sha256="in1",
            output_paths=[],
            record_counts={"artifacts": 1},
        )
    assert http_route.call_count == 1
    assert manifest1.model_calls == 1
    assert manifest1.model_cache_hits == 0

    settings_frozen = synthetic_settings(
        cache_dir,
        mode=ResourceMode.FROZEN,
        deployment_environment="local",
        lexicon_path=bundle,
        lexicon_sha256=manifest.bundle_sha256,
        llm_base_url=endpoint,
        llm_model="local-model",
        llm_model_revision="rev-1",
    )
    with BioSemAlign(
        profile_v1,
        settings=settings_frozen,
        cache=ResponseCache(cache_dir, mode=ResourceMode.FROZEN),
    ) as frozen_pipeline:
        art2 = frozen_pipeline.align_artifact(
            frozen_pipeline.make_request("TOYBRAND XL 50 MG", route="oral")
        )
        manifest2 = build_run_manifest(
            pipeline=frozen_pipeline,
            run_id="r2",
            started_at=utc_now(),
            input_sha256="in2",
            output_paths=[],
            record_counts={"artifacts": 1},
        )
    assert http_route.call_count == 1
    assert art1.decision.selected_concepts == art2.decision.selected_concepts
    assert art1.decision.relationship_to_source == art2.decision.relationship_to_source
    assert art1.decision.routing_status == art2.decision.routing_status
    assert art2.decision.provenance.model_calls[0].from_cache is True
    assert manifest2.model_calls == 1
    assert manifest2.model_cache_hits == 1

    # Changed prompt in FROZEN mode is refused on cache miss and issues zero HTTP calls.
    changed_payload = profile_v1.model_dump(mode="json")
    changed_payload["tiers"][0]["resolve"] = {
        "op": "llm",
        "options": {"prompt_version": "leashed-2"},
    }
    changed_profile = TaskProfile.model_validate(changed_payload)
    with (
        BioSemAlign(
            changed_profile,
            settings=settings_frozen,
            cache=ResponseCache(cache_dir, mode=ResourceMode.FROZEN),
        ) as changed_pipeline,
        pytest.raises(CacheMiss),
    ):
        changed_pipeline.align_artifact(
            changed_pipeline.make_request("TOYBRAND XL 50 MG", route="oral")
        )
    assert http_route.call_count == 1


def test_s15_model_linker_internal_knowledge_and_model_check_warning(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = appendix_a_bundle
    payload = appendix_a_drug_profile(
        desired_granularity="ingredient", include_cleaned_tier=False
    ).model_dump(mode="json")
    payload["input"]["source_type"] = "extracted_terms"
    payload["context"] = {
        "fields": [
            {"name": "route", "kind": "attribute", "use": ["key", "check"]},
            {"name": "indication", "kind": "context", "use": [], "blinded": True},
        ]
    }
    profile = TaskProfile.model_validate(payload)

    linker_service = scripted_model_service(
        {
            "assertions": [
                {
                    "subject_code": "T201",
                    "predicate": "has_ingredient",
                    "object_code": "T300",
                    "object_vocabulary": "RxNorm",
                    "object_label": "alphazine",
                }
            ]
        },
        model_name="scripted-link",
    )
    check_service = scripted_model_service(
        {
            "findings": [
                {
                    "code": "MODEL_CHECK_NOTE",
                    "severity": "warning",
                    "message": "check flagged ambiguous brand",
                    "concept_id": "T201",
                }
            ]
        },
        model_name="scripted-check",
    )

    def build_workflow(context: OpContext) -> Cascade:
        cfg = context.profile.tiers[0]
        base_linker = build("link", cfg.link.op, context, cfg.link.options)
        return Cascade(
            [
                Tier(
                    name="with_model_link",
                    match=build("match", cfg.match[0].op, context, cfg.match[0].options),
                    linker=ScriptedModelLinker(base_linker, context.profile, linker_service),
                    resolver=build("resolve", "rules", context, {}),
                )
            ]
        )

    from tests.support import synthetic_settings

    from biosemalign.storage.cache import ResponseCache

    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=build_workflow,
    ) as pipeline:
        pipeline.validator = type(pipeline.validator)(
            profile,
            provider=pipeline.knowledge.primary_provider(),
            checks=[*pipeline.validator.checks, ScriptedModelCheck(profile, check_service)],
            model_service=check_service,
        )
        pipeline.kernel.validator = pipeline.validator
        # Use extracted_terms with term absent from source text to verify prior findings
        # (MENTION_NOT_IN_SOURCE) cannot be removed when ScriptedModelCheck adds its warning.
        req = AlignmentRequest(
            source_type=SourceType.EXTRACTED_TERMS,
            value={
                "text": "Unrelated source sentence.",
                "terms": [{"text": "TOYBRAND XL", "extractor": "ner:test", "type": "drug"}],
                "indication": "blinded outcome",
            },
            task=profile.task.operation,
            mapping_intent=profile.task.mapping_intent,
            target_terminology=profile.target.terminology,
            desired_granularity=profile.target.level,
        )
        run = pipeline.new_run()
        sau = pipeline.build_sau(req, run=run)[0]
        artifact = pipeline.kernel.align(sau, run)

    t201 = artifact.knowledge_package.candidate_by_qualified_id("RxNorm", "T201")
    assert t201 is not None
    llm_edges = [s for s in t201.relationships if s.object_id == "T300"]
    assert len(llm_edges) == 1
    assert llm_edges[0].authority is Authority.LLM_INTERNAL_KNOWLEDGE

    assert artifact.knowledge_package.links is not None
    targets_for_t201 = artifact.knowledge_package.links.targets_of("RxNorm", "T201")
    assert [t.concept_code for t in targets_for_t201] == ["T100"]

    assert [c.concept_id for c in artifact.decision.selected_concepts] == ["T100"]
    codes = {f.code for f in artifact.validation.findings}
    assert {"MENTION_NOT_IN_SOURCE", "MODEL_CHECK_NOTE"} <= codes
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    kinds = [call.operator_kind for call in artifact.decision.provenance.model_calls]
    assert kinds == ["link", "check"]
    assert verify_artifact(artifact, profile=profile).valid


def test_s10_coding_per_sau_and_bulk_agree_with_cleaning_pass(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    import csv

    import duckdb
    import yaml

    from biosemalign.schemas.coding import CodingArtifact
    from biosemalign.workflows.coding import code_terms

    bundle, manifest = appendix_a_bundle
    profile = appendix_a_coding_profile()
    terms = ["TOYBRAND XL 50 MG", "TOYBRAND XL 50MG", "zzzqqq"]

    per_sau: dict[str, CodingArtifact] = {}
    with appendix_a_pipeline(tmp_path / "sau", bundle, manifest, profile=profile) as pipeline:
        for term in terms:
            art = pipeline.run(pipeline.make_request(term))
            assert isinstance(art, CodingArtifact)
            assert verify_artifact(art, profile=profile).valid
            per_sau[term] = art
        assert pipeline.code_text("TOYBRAND XL 50MG").codes == per_sau["TOYBRAND XL 50MG"].codes

    s1 = per_sau["TOYBRAND XL 50 MG"]
    assert s1.uncoded is False
    assert [(c.vocabulary, c.concept_code) for c in s1.codes] == [("RxNorm", "T200")]
    assert s1.coded_by_tier == "exact"
    assert s1.cleaned_by == []

    s5 = per_sau["TOYBRAND XL 50MG"]
    assert s5.uncoded is False
    assert [(c.vocabulary, c.concept_code) for c in s5.codes] == [("RxNorm", "T201")]
    assert s5.coded_by_tier == "cleaned"
    assert s5.cleaned_by == ["drug.extract_strength"]
    assert s5.sau.mention.text == "TOYBRAND XL 50MG"
    assert s5.sau.mention.retrieval_text == "TOYBRAND XL"

    un = per_sau["zzzqqq"]
    assert un.uncoded is True
    assert un.codes == []
    assert un.coded_by_tier is None
    assert un.cleaned_by == []

    # Tampered coding artifact fails verification.
    tampered = s5.model_dump(mode="json")
    tampered["cleaned_by"] = []
    assert not verify_artifact(tampered, profile=profile).valid

    # Bulk Python coding + CLI --profile produce identical codes and pass metadata.
    input_csv = tmp_path / "terms.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "drugname"])
        writer.writeheader()
        for idx, term in enumerate(terms, start=1):
            writer.writerow({"id": f"r{idx}", "drugname": term})

    out_bulk = tmp_path / "bulk"
    report = code_terms(
        bundle,
        input_csv,
        out_bulk,
        text_column="drugname",
        id_columns=["id"],
        profile=profile,
    )
    assert report.coded_terms == 2
    assert report.coded_terms_exact == 1
    assert report.coded_terms_cleaned == 1

    profile_yaml = tmp_path / "coding_profile.yaml"
    profile_yaml.write_text(
        yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    out_cli = tmp_path / "cli_bulk"
    cli_result = CliRunner().invoke(
        app,
        [
            "code",
            "--bundle",
            str(bundle),
            "--input",
            str(input_csv),
            "--text-column",
            "drugname",
            "--id-column",
            "id",
            "--profile",
            str(profile_yaml),
            "--output",
            str(out_cli),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "1 after cleaning" in cli_result.output

    for out_dir in (out_bulk, out_cli):
        codes_rows = duckdb.sql(
            f"SELECT normalized, vocabulary_id, concept_code, coding_pass, cleaned_by "
            f"FROM '{out_dir / 'codes.parquet'}'"
        ).fetchall()
        by_term: dict[str, set[tuple[str, str]]] = {t.lower(): set() for t in terms}
        pass_info: dict[str, tuple[str, list[str]]] = {}
        for norm, vocab, code, coding_pass, cleaned_by in codes_rows:
            by_term[norm].add((vocab, code))
            pass_info[norm] = (coding_pass, list(cleaned_by))

        for term in terms:
            expected_codes = {(c.vocabulary, c.concept_code) for c in per_sau[term].codes}
            assert by_term[term.lower()] == expected_codes

        assert pass_info["toybrand xl 50 mg"] == ("exact", [])
        assert pass_info["toybrand xl 50mg"] == ("cleaned", ["drug.extract_strength"])
        uncoded_rows = duckdb.sql(
            f"SELECT normalized FROM '{out_dir / 'uncoded.parquet'}'"
        ).fetchall()
        assert [row[0] for row in uncoded_rows] == ["zzzqqq"]


def test_s9_phenotyping_toy_fever_propose_review_export_and_tamper(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    import yaml
    from pydantic import ValidationError

    from biosemalign.exceptions import WorkflowViolation
    from biosemalign.schemas.phenotyping import (
        ConceptSet,
        ConceptSetArtifact,
        ConceptSetReview,
        MemberReviewAction,
    )
    from biosemalign.workflows.phenotyping import apply_concept_set_review, export_concept_set

    bundle, manifest = appendix_a_bundle
    profile = appendix_a_phenotyping_profile()

    with appendix_a_pipeline(tmp_path / "run", bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.run(pipeline.make_request("toy fever"))
        assert isinstance(artifact, ConceptSetArtifact)

        # Anchors, expansion with paths, proposed members, and human-review routing.
        assert [(a.vocabulary, a.concept_code) for a in artifact.expansion.anchors] == [
            ("ToyDisease", "D1")
        ]
        exp_by_code = {m.concept.candidate_id: m for m in artifact.expansion.members}
        assert set(exp_by_code) == {"D2", "D3"}
        assert exp_by_code["D2"].path[0].predicate == "is_a"
        assert exp_by_code["D3"].path[0].predicate == "is_a"

        prop_by_code = {m.concept_code: m for m in artifact.proposal.members}
        assert prop_by_code["D1"].relation is RelationshipType.EXACT
        assert prop_by_code["D1"].scope == "narrow"
        assert prop_by_code["D2"].relation is RelationshipType.TARGET_NARROWER_THAN_SOURCE
        assert prop_by_code["D2"].scope == "narrow"
        assert prop_by_code["D3"].relation is RelationshipType.TARGET_NARROWER_THAN_SOURCE
        assert prop_by_code["D3"].scope == "narrow"

        assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
        assert verify_artifact(artifact, profile=profile).valid

        # Interactive search expansion from code + level.
        expanded = pipeline.expand_code("D1", level="Disease")
        assert {m.concept.candidate_id for m in expanded.members} == {"D2", "D3"}

        # G8: export refused before recorded approval.
        with pytest.raises(WorkflowViolation, match="approval"):
            export_concept_set(artifact, name="toy_fever_set")

        # Tampering to include a code outside expansion (unsigned or re-signed) fails verification.
        tampered = artifact.model_dump(mode="json")
        extra_member = dict(tampered["proposal"]["members"][0])
        extra_member["concept_code"] = "G1"
        tampered["proposal"]["members"].append(extra_member)
        tampered["decision"]["members"].append(extra_member)
        assert not verify_artifact(tampered, profile=profile).valid
        tampered["artifact_id"] = ConceptSetArtifact.compute_artifact_id(
            run_id=tampered["run_id"],
            sau=SemanticAlignmentUnit.model_validate(tampered["sau"]),
            coding=artifact.coding,
            expansion=artifact.expansion,
            proposal=type(artifact.proposal).model_validate(tampered["proposal"]),
            validation=artifact.validation,
            decision=type(artifact.decision).model_validate(tampered["decision"]),
        )
        with pytest.raises(ValidationError, match="outside the expansion"):
            ConceptSetArtifact.model_validate(tampered)
        assert not verify_artifact(tampered, profile=profile).valid

        # Reviewer-added code must exist in terminology.
        bad_review = ConceptSetReview(
            review_id="rev-bad",
            artifact_id=artifact.artifact_id,
            reviewer="dr-review",
            approved=True,
            actions=[
                MemberReviewAction(
                    action="add",
                    vocabulary="ToyDisease",
                    concept_code="D999",
                    relation=RelationshipType.RELATED_NOT_EQUIVALENT,
                    scope="broad",
                )
            ],
        )
        with pytest.raises(ValueError, match="does not exist"):
            apply_concept_set_review(artifact, bad_review, knowledge=pipeline.knowledge)

        # Valid review: approve, add existing terminology code G1 (marked added_by_reviewer), export.
        good_review = ConceptSetReview(
            review_id="rev-1",
            artifact_id=artifact.artifact_id,
            reviewer="dr-review",
            approved=True,
            actions=[
                MemberReviewAction(
                    action="add",
                    vocabulary="ToyDisease",
                    concept_code="G1",
                    relation=RelationshipType.TARGET_BROADER_THAN_SOURCE,
                    scope="broad",
                    reason="Add umbrella group for sensitivity analysis.",
                )
            ],
        )
        reviewed = apply_concept_set_review(artifact, good_review, knowledge=pipeline.knowledge)
        assert verify_artifact(reviewed, profile=profile).valid
        by_approved = {m.concept_code: m for m in reviewed.approved_members}
        assert set(by_approved) == {"D1", "D2", "D3", "G1"}
        assert by_approved["G1"].added_by_reviewer is True
        assert by_approved["D1"].added_by_reviewer is False

        broad_set = export_concept_set(reviewed, name="toy_fever", version="1.0.0", scope="broad")
        narrow_set = export_concept_set(reviewed, name="toy_fever", version="1.0.0", scope="narrow")
        assert isinstance(broad_set, ConceptSet)
        assert broad_set.artifact_id == artifact.artifact_id
        assert broad_set.artifact_digest == reviewed.compute_digest()
        assert broad_set.artifact_digest != artifact.artifact_id
        assert broad_set.terminology == "ToyDisease"
        assert broad_set.terminology_release == "toy-disease-1"
        assert broad_set.level == "Disease"
        assert {m.concept_code for m in broad_set.members} == {"D1", "D2", "D3", "G1"}
        assert {m.concept_code for m in narrow_set.members} == {"D1", "D2", "D3"}
        assert verify_artifact(broad_set, profile=profile).valid

    # Exercise CLI `phenotype {propose, review, export, verify}`.
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    profile_yaml = cli_dir / "phenotype.yaml"
    profile_yaml.write_text(
        yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    prop_json = cli_dir / "proposed.json"
    rev_json = cli_dir / "review.json"
    reviewed_json = cli_dir / "reviewed.json"
    export_json = cli_dir / "exported.json"

    runner = CliRunner()
    env = {
        "BIOSEMALIGN_MODE": "FROZEN",
        "BIOSEMALIGN_CACHE_DIR": str(tmp_path / "cli_cache"),
        "BIOSEMALIGN_LEXICON_PATH": str(bundle),
        "BIOSEMALIGN_LEXICON_SHA256": manifest.bundle_sha256,
    }
    res_prop = runner.invoke(
        app,
        ["phenotype", "propose", "-p", str(profile_yaml), "-t", "toy fever", "-o", str(prop_json)],
        env=env,
    )
    assert res_prop.exit_code == 0, res_prop.output
    cli_prop_art = ConceptSetArtifact.model_validate_json(prop_json.read_text(encoding="utf-8"))
    rev_json.write_text(
        ConceptSetReview(
            review_id="cli-rev-1",
            artifact_id=cli_prop_art.artifact_id,
            reviewer="cli-user",
            approved=True,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    res_rev = runner.invoke(
        app,
        [
            "phenotype",
            "review",
            "-p",
            str(profile_yaml),
            "-a",
            str(prop_json),
            "-r",
            str(rev_json),
            "-o",
            str(reviewed_json),
        ],
        env=env,
    )
    assert res_rev.exit_code == 0, res_rev.output
    res_exp = runner.invoke(
        app,
        [
            "phenotype",
            "export",
            "-a",
            str(reviewed_json),
            "-n",
            "cli_toy_fever",
            "-o",
            str(export_json),
        ],
        env=env,
    )
    assert res_exp.exit_code == 0, res_exp.output
    res_ver = runner.invoke(
        app,
        ["phenotype", "verify", "-i", str(export_json), "-p", str(profile_yaml)],
        env=env,
    )
    assert res_ver.exit_code == 0, res_ver.output
    assert "verified: 1" in res_ver.output


def test_s13_scripted_model_selector_flags_disagreement_and_forbids_codes(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    from pydantic import ValidationError

    from biosemalign.workflows.phenotyping import PhenotypingWorkflow

    # M1: selector reply schema rejects any terminology code fields.
    with pytest.raises(ValidationError):
        JudgeMembersReply.model_validate(
            {
                "judgments": [
                    {
                        "member": 2,
                        "included": True,
                        "relationship": "TARGET_BROADER_THAN_SOURCE",
                        "scope": "broad",
                        "rationale": "bad code field",
                        "concept_code": "D2",
                    }
                ]
            }
        )

    bundle, manifest = appendix_a_bundle
    profile = appendix_a_phenotyping_profile()

    # Model calls D2 a vaguer term (broader/broad), disagreeing with hierarchy (narrower/narrow).
    model_service = scripted_model_service(
        {
            "judgments": [
                {
                    "member": 1,
                    "included": True,
                    "relationship": "EXACT",
                    "scope": "narrow",
                    "rationale": "D1 matches the idea directly.",
                },
                {
                    "member": 2,
                    "included": True,
                    "relationship": "TARGET_BROADER_THAN_SOURCE",
                    "scope": "broad",
                    "rationale": "Severe toy fever appears broader in clinical notes.",
                },
                {
                    "member": 3,
                    "included": True,
                    "relationship": "TARGET_NARROWER_THAN_SOURCE",
                    "scope": "narrow",
                    "rationale": "Recurrent toy fever is a subtype.",
                },
            ]
        },
        model_name="scripted-selector",
    )

    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        base_runner = pipeline.phenotyping_runner
        assert base_runner is not None
        selector = ScriptedModelSelector(pipeline.knowledge, profile, model_service)
        runner = PhenotypingWorkflow(
            profile=profile,
            knowledge=pipeline.knowledge,
            coding_cascade=base_runner.coding_cascade,
            expanders=base_runner.expanders,
            selector=selector,
            model_services=(model_service,),
        )
        run = pipeline.new_run()
        sau = pipeline.build_sau(
            pipeline.make_request("toy fever", indication="secret outcome"), run
        )[0]
        artifact = runner.propose(sau, run)

    by_code = {m.concept_code: m for m in artifact.proposal.members}
    d2 = by_code["D2"]
    assert d2.relation is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert d2.hierarchy_relation is RelationshipType.TARGET_NARROWER_THAN_SOURCE
    assert d2.model_disagreement is True
    assert d2.model_rationale == "Severe toy fever appears broader in clinical notes."
    assert by_code["D3"].model_disagreement is False
    assert any(f.code == "MODEL_HIERARCHY_DISAGREEMENT" for f in artifact.validation.findings)
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert artifact.decision.provenance.model_calls[0].operator_kind == "select"
    # Blinded field must not appear in selector prompt (M5).
    assert "secret outcome" not in model_service.backend.messages[0][1]["content"]
    assert verify_artifact(artifact, profile=profile).valid
