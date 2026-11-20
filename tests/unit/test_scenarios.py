"""WP0 baseline scenario tests against the invented Appendix A terminology."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.scenarios_support import (
    appendix_a_drug_profile,
    appendix_a_pipeline,
    appendix_a_toy_disease_profile,
    build_appendix_a_bundle,
)
from typer.testing import CliRunner

from biosemalign import verify_artifact
from biosemalign.cli import app
from biosemalign.enums import (
    DecisionType,
    RelationshipType,
    RoutingStatus,
    SemanticLossKind,
    TierOutcome,
    ValidationStatus,
)
from biosemalign.knowledge.lexicon.contract import LexiconManifest


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
    assert SemanticLossKind.COMBINATION_COMPONENT in {loss.kind for loss in decision.semantic_loss}
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert decision.provenance.resolution is not None
    assert decision.provenance.resolution.resolved_by_tier == "exact"
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
    assert SemanticLossKind.BRAND_IDENTITY in {loss.kind for loss in decision.semantic_loss}
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


def test_abstention_s3_today_no_granularity_on_toy_disease(
    appendix_a_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    # Records today's S3 baseline (abstains when desired_granularity is None);
    # WP2 updates this expectation when identity mapping lands.
    bundle, manifest = appendix_a_bundle
    profile = appendix_a_toy_disease_profile(desired_granularity=None)
    with appendix_a_pipeline(tmp_path, bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("severe toy fever"))

    decision = artifact.decision
    assert decision.decision is DecisionType.ABSTAINED
    assert decision.relationship_to_source is RelationshipType.UNSUPPORTED_MAPPING
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert "no policy for requested granularity None" in decision.rationale
    assert verify_artifact(artifact, profile=profile).valid
