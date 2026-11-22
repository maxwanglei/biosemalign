"""Tests for the external ``biosemalign-pk-pack`` plugin distribution."""

from __future__ import annotations

import csv
import importlib.metadata
from pathlib import Path

import yaml
from biosemalign_pk_pack.checks import PKSpecimenCheck
from biosemalign_pk_pack.cleaners import NormalizePKParameterCleaner
from tests.support import synthetic_settings

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.enums import DecisionType, RelationshipType, RoutingStatus
from biosemalign.knowledge.lexicon.build import build_lexicon, load_declaration
from biosemalign.modules import OpContext, available, build
from biosemalign.plugins import PLUGIN_ENTRY_POINT_GROUP
from biosemalign.profiles import available_profiles, load_profile
from biosemalign.storage.cache import ResponseCache
from biosemalign.targets import builtin_specs
from biosemalign.testing import (
    check_check_contract,
    check_cleaner_contract,
    check_domain_pack_contract,
    check_target_spec_contract,
    make_contract_candidate,
    make_contract_package,
    make_contract_sau,
)


def _build_toypk_bundle(root: Path) -> tuple[Path, str]:
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "concepts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["vocabulary_id", "concept_code", "concept_name", "concept_class", "status"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "vocabulary_id": "ToyPK",
                    "concept_code": "PK1",
                    "concept_name": "Cmax",
                    "concept_class": "Parameter",
                    "status": "active",
                },
                {
                    "vocabulary_id": "ToyPK",
                    "concept_code": "PK2",
                    "concept_name": "AUC0-inf",
                    "concept_class": "Parameter",
                    "status": "active",
                },
                {
                    "vocabulary_id": "ToyPK",
                    "concept_code": "PK3",
                    "concept_name": "Clearance",
                    "concept_class": "Parameter",
                    "status": "active",
                },
            ]
        )
    with (tables / "synonyms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vocabulary_id", "concept_code", "synonym"])
        writer.writeheader()
        writer.writerows(
            [
                {"vocabulary_id": "ToyPK", "concept_code": "PK1", "synonym": "Cmax"},
                {"vocabulary_id": "ToyPK", "concept_code": "PK2", "synonym": "AUC0-inf"},
                {"vocabulary_id": "ToyPK", "concept_code": "PK3", "synonym": "Clearance"},
            ]
        )
    decl_path = root / "vocabularies.yaml"
    decl_path.write_text(
        yaml.safe_dump(
            {
                "vocabularies": [
                    {"vocabulary_id": "ToyPK", "release": "toypk-1.0", "license": "synthetic"}
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    bundle = root / "bundle"
    manifest = build_lexicon([tables], bundle, declaration=load_declaration(decl_path))
    return bundle, manifest.bundle_sha256


def test_pk_pack_discovers_via_entry_points_and_passes_contracts() -> None:
    eps = [
        ep.name for ep in importlib.metadata.entry_points().select(group=PLUGIN_ENTRY_POINT_GROUP)
    ]
    assert "pk" in eps
    check_domain_pack_contract("pk")
    assert "toypk_parameter_coding" in available_profiles()
    assert "pk.normalize_parameter" in available("clean")
    assert "pk.specimen_check" in available("check")

    profile = load_profile("toypk_parameter_coding")
    check_target_spec_contract(builtin_specs()["toypk"])

    ctx = OpContext(profile=profile)
    cleaner = build("clean", "pk.normalize_parameter", ctx, {})
    assert isinstance(cleaner, NormalizePKParameterCleaner)
    cleaned = check_cleaner_contract(
        cleaner,
        make_contract_sau(
            "Cmax (plasma)", target_terminology="ToyPK", desired_granularity="Parameter"
        ),
    )
    assert cleaned.mention.text == "Cmax (plasma)"
    assert cleaned.mention.retrieval_text == "Cmax"
    assert cleaned.attributes.specimen == "plasma"

    checker = build("check", "pk.specimen_check", ctx, {})
    assert isinstance(checker, PKSpecimenCheck)
    sau = make_contract_sau("Cmax", target_terminology="ToyPK", desired_granularity="Parameter")
    cand = make_contract_candidate(
        code="PK1", label="Cmax", vocabulary="ToyPK", concept_class="Parameter"
    )
    pkg = make_contract_package(sau, [cand])
    from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept

    sel = SelectedConcept(
        concept_id="PK1",
        label="Cmax",
        vocabulary="ToyPK",
        concept_class="Parameter",
        relationship_to_source=RelationshipType.EXACT,
    )
    prop = AlignmentProposal(
        sau_id=sau.sau_id,
        decision=DecisionType.MAPPED,
        selected_concepts=[sel],
        source_level_concept=sel,
        analysis_level_concept=sel,
        relationship_to_source=RelationshipType.EXACT,
        rationale="exact PK match",
    )
    findings = check_check_contract(checker, prop, pkg, sau)
    assert [f.code for f in findings] == ["MISSING_PK_SPECIMEN"]


def test_pk_pack_end_to_end_pipeline(tmp_path: Path) -> None:
    bundle, sha256 = _build_toypk_bundle(tmp_path / "toypk")
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(cache_dir, lexicon_path=bundle, lexicon_sha256=sha256)
    profile = load_profile("toypk_parameter_coding")

    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        cleaned_art = pipeline.run(pipeline.make_request("Cmax (plasma)"))
        assert cleaned_art.decision.decision is DecisionType.MAPPED
        assert cleaned_art.decision.relationship_to_source is RelationshipType.EXACT
        assert cleaned_art.decision.selected_concepts[0].concept_id == "PK1"
        assert cleaned_art.decision.preserved_attributes == {"specimen": "plasma"}
        assert cleaned_art.decision.routing_status is RoutingStatus.AUTO_ACCEPT
        assert not any(f.code == "MISSING_PK_SPECIMEN" for f in cleaned_art.validation.findings)
        assert verify_artifact(cleaned_art, profile=profile).valid

        no_specimen_art = pipeline.run(pipeline.make_request("Cmax"))
        assert no_specimen_art.decision.decision is DecisionType.MAPPED
        assert no_specimen_art.decision.selected_concepts[0].concept_id == "PK1"
        assert any(f.code == "MISSING_PK_SPECIMEN" for f in no_specimen_art.validation.findings)
        assert no_specimen_art.decision.routing_status is RoutingStatus.HUMAN_REVIEW
        assert verify_artifact(no_specimen_art, profile=profile).valid
