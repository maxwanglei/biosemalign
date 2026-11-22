"""Unit tests for the WP5 coding workflow, per-SAU and bulk cleaning."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from tests.scenarios_support import appendix_a_coding_profile, build_appendix_a_bundle
from tests.support import ALPHA, synthetic_settings
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.enums import DecisionType
from biosemalign.exceptions import ConfigurationError, WorkflowViolation
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.run_manifest import build_run_manifest
from biosemalign.schemas.coding import CodedConcept, CodingArtifact
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.storage.cache import ResponseCache
from biosemalign.storage.jsonl import write_models
from biosemalign.testing import (
    ScriptedModelCleaner,
    ScriptedModelMatcher,
    scripted_model_service,
)
from biosemalign.util import utc_now
from biosemalign.workflows.coding import CodingCascade, CodingTier, CodingWorkflow, code_terms


def _resign_coding(payload: dict[str, Any]) -> dict[str, Any]:
    validated_codes = [CodedConcept.model_validate(c) for c in payload["codes"]]
    from biosemalign.schemas.coding import CodingTierAttempt
    from biosemalign.schemas.links import LinkSet
    from biosemalign.schemas.provenance import Provenance
    from biosemalign.schemas.sau import SemanticAlignmentUnit

    payload["artifact_id"] = CodingArtifact.compute_artifact_id(
        run_id=payload["run_id"],
        sau=SemanticAlignmentUnit.model_validate(payload["sau"]),
        codes=validated_codes,
        uncoded=payload["uncoded"],
        coded_by_tier=payload["coded_by_tier"],
        cleaned_by=payload["cleaned_by"],
        attempts=[CodingTierAttempt.model_validate(a) for a in payload["attempts"]],
        links=LinkSet.model_validate(payload["links"]) if payload["links"] is not None else None,
        provenance=Provenance.model_validate(payload["provenance"]),
    )
    return payload


@pytest.fixture(scope="module")
def coding_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    return build_appendix_a_bundle(tmp_path_factory.mktemp("coding_wf"))


def test_coding_artifact_self_verification_and_tampering_checks(
    coding_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = coding_bundle
    profile = appendix_a_coding_profile()
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        s5 = pipeline.code_text("TOYBRAND XL 50MG")
        assert verify_artifact(s5, profile=profile).valid

    # Wrong profile fails verification.
    wrong = profile.model_copy(update={"profile_version": "9.9.9"})
    assert any("9.9.9" in err for err in verify_artifact(s5, profile=wrong).errors)

    # Prohibited predicate in links fails profile verification.
    assert s5.links is not None
    bad_pred_profile = profile.model_copy(
        update={
            "knowledge": profile.knowledge.model_copy(
                update={
                    "allowed_relationships": [],
                    "prohibited_relationships": ["has_ingredient"],
                }
            )
        }
    )
    assert any("prohibits" in err for err in verify_artifact(s5, profile=bad_pred_profile).errors)

    # Tampered artifact_id / uncoded / attempts / links are rejected.
    raw = s5.model_dump(mode="json")
    raw["uncoded"] = True
    assert not verify_artifact(raw).valid

    re_signed = s5.model_dump(mode="json")
    re_signed["attempts"][-1]["cleaners_applied"] = []
    assert any("cleaned_by" in err for err in verify_artifact(_resign_coding(re_signed)).errors)

    missing_model_mark = s5.model_dump(mode="json")
    missing_model_mark["sau"]["mention"]["retrieval_text_from_model"] = True
    assert any(
        "from_model_suggestion" in err
        for err in verify_artifact(_resign_coding(missing_model_mark)).errors
    )

    bogus_link = s5.model_dump(mode="json")
    bogus_link["codes"][0]["target_links"][0]["target"]["concept_code"] = "T999"
    assert any("absent" in err for err in verify_artifact(_resign_coding(bogus_link)).errors)


def test_biosemalign_run_dispatches_coding_and_blocks_mapping_only_methods(
    coding_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = coding_bundle
    coding_profile = appendix_a_coding_profile()
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    with BioSemAlign(
        coding_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        result = pipeline.run(pipeline.make_request("TOYBRAND XL 50 MG"))
        assert isinstance(result, CodingArtifact)
        all_results = pipeline.code_all_artifacts(pipeline.make_request("TOYBRAND XL 50 MG"))
        assert len(all_results) == 1
        assert all_results[0].codes == result.codes

        with pytest.raises(ConfigurationError, match="not 'mapping'"):
            pipeline.align_artifact(pipeline.make_request("TOYBRAND XL 50 MG"))
        with pytest.raises(ConfigurationError, match="not 'mapping'"):
            list(pipeline.align_batch_records([{"drug": "TOYBRAND XL 50 MG"}], value_column="drug"))
        with pytest.raises(ConfigurationError, match="not 'mapping'"):
            pipeline.attach_memo(tmp_path / "memo.sqlite")

        manifest_obj = build_run_manifest(
            pipeline=pipeline,
            run_id="coding-run",
            started_at=utc_now(),
            input_sha256="a" * 64,
            output_paths=[],
            record_counts={"coded": 1},
        )
        assert manifest_obj.workflow is not None
        assert manifest_obj.adjudicator_version is None

    # Mapping profile still works with BioSemAlign.run() AND BioSemAlign.code_artifact().
    mapping_payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    mapping_payload["tiers"] = [
        {"name": "exact", "match": [{"op": "exact", "options": {"vocabularies": ["RxNorm"]}}]}
    ]
    mapping_profile = TaskProfile.model_validate(mapping_payload)
    with BioSemAlign(
        mapping_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as map_pipeline:
        mapped = map_pipeline.run(map_pipeline.make_request("TOYBRAND XL 50 MG"))
        assert mapped.decision.decision is DecisionType.MAPPED
        coded = map_pipeline.code_artifact(map_pipeline.make_request("TOYBRAND XL 50 MG"))
        assert isinstance(coded, CodingArtifact)
        assert [(c.vocabulary, c.concept_code) for c in coded.codes] == [("RxNorm", "T200")]


def test_code_text_refuses_network_and_model_operators(
    coding_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = coding_bundle
    cache_dir = tmp_path / "cache"
    online_settings = synthetic_settings(cache_dir)
    online_profile = load_profile("faers_drug_normalization")
    with (
        BioSemAlign(
            online_profile,
            settings=online_settings.model_copy(update={"mode": online_settings.mode.ONLINE}),
            environment="local",
        ) as net_pipeline,
        pytest.raises(ConfigurationError, match="requires network"),
    ):
        net_pipeline.code_text(ALPHA)

    profile = appendix_a_coding_profile(include_cleaned_tier=False)
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    model_service = scripted_model_service(
        {"queries": ["toyprolol"]}, model_name="scripted-cleaner"
    )
    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        pipeline.coding_cascade = CodingCascade(
            [
                CodingTier(
                    name="with_model_clean",
                    cleaners=(ScriptedModelCleaner(profile, model_service),),
                    match=pipeline.coding_cascade.tiers[0].match,
                )
            ]
        )
        pipeline.coding_runner = CodingWorkflow(
            profile=profile,
            cascade=pipeline.coding_cascade,
            knowledge=pipeline.knowledge,
            model_services=(model_service,),
        )
        with pytest.raises(ConfigurationError, match="uses model"):
            pipeline.code_text("TPL")


class _DestructiveCleaner:
    op_id = "destructive"
    version = "0.1.0"
    uses_model = False
    options: ClassVar[dict[str, Any]] = {}

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        return sau.model_copy(
            update={"mention": sau.mention.model_copy(update={"text": "rewritten"})}
        )


def test_coding_workflow_enforces_g1_model_marks_and_optional_linking(
    coding_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = coding_bundle
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    # link: null skips annotation linking while still returning matched codes.
    payload = appendix_a_coding_profile(include_cleaned_tier=False).model_dump(mode="json")
    payload["tiers"][0]["link"] = None
    payload["limits"]["candidates"] = 1
    no_link_profile = TaskProfile.model_validate(payload)
    with BioSemAlign(
        no_link_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        art = pipeline.code_artifact(pipeline.make_request("TOYBRAND XL 50 MG"))
        assert art.links is None
        assert art.codes[0].target_links == []

    # Destructive cleaner raises WorkflowViolation (G1).
    with BioSemAlign(
        no_link_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        matcher = pipeline.coding_cascade.tiers[0].match
        bad_cascade = CodingCascade(
            [CodingTier(name="bad", cleaners=(_DestructiveCleaner(),), match=matcher)]
        )
        runner = CodingWorkflow(
            profile=no_link_profile, cascade=bad_cascade, knowledge=pipeline.knowledge
        )
        run = pipeline.new_run()
        sau = pipeline.build_sau(pipeline.make_request("TOYBRAND XL 50 MG"), run)[0]
        with pytest.raises(WorkflowViolation, match="changed the source mention"):
            runner.code(sau, run)

    # Model-backed cleaner + matcher propagate M4/G10 marks and provenance.
    profile = appendix_a_coding_profile(include_cleaned_tier=False)
    cleaner_svc = scripted_model_service({"queries": ["toyprolol"]}, model_name="scripted-clean")
    matcher_svc = scripted_model_service({"queries": ["toyprolol"]}, model_name="scripted-m")
    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        model_matcher = ScriptedModelMatcher(pipeline.knowledge, profile, matcher_svc)
        cascade = CodingCascade(
            [
                CodingTier(
                    name="model_code",
                    cleaners=(ScriptedModelCleaner(profile, cleaner_svc),),
                    match=model_matcher,
                )
            ]
        )
        runner = CodingWorkflow(
            profile=profile,
            cascade=cascade,
            knowledge=pipeline.knowledge,
            model_services=(cleaner_svc, matcher_svc),
        )
        run = pipeline.new_run()
        sau = pipeline.build_sau(pipeline.make_request("TPL"), run)[0]
        model_art = runner.code(sau, run)
        assert not model_art.uncoded
        assert model_art.codes[0].from_model_suggestion is True
        assert model_art.codes[0].concept_code == "T100"
        assert [c.operator_kind for c in model_art.provenance.model_calls] == ["clean", "match"]
        assert verify_artifact(model_art, profile=profile).valid


def test_cli_verify_and_bulk_guards_for_coding(
    coding_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = coding_bundle
    profile = appendix_a_coding_profile()
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    with BioSemAlign(
        profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipeline:
        art = pipeline.code_artifact(pipeline.make_request("TOYBRAND XL 50MG"))

    jsonl_path = tmp_path / "coding_artifacts.jsonl"
    write_models(jsonl_path, [art])
    profile_yaml = tmp_path / "profile.yaml"
    profile_yaml.write_text(json.dumps(profile.model_dump(mode="json")), encoding="utf-8")
    runner = CliRunner()
    ok = runner.invoke(app, ["verify", "-i", str(jsonl_path), "-p", str(profile_yaml)])
    assert ok.exit_code == 0, ok.output
    assert "verified: 1" in ok.output

    tampered = art.model_dump(mode="json")
    tampered["codes"] = []
    jsonl_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    bad = runner.invoke(app, ["verify", "-i", str(jsonl_path)])
    assert bad.exit_code == 1
    assert "invalid: 1" in bad.output

    # Bulk coding refuses model-backed cleaners and supports ToyDisease level=None.
    csv_path = tmp_path / "disease.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["term"])
        writer.writeheader()
        writer.writerow({"term": "toy pyrexia"})

    model_cleaner = ScriptedModelCleaner(profile, scripted_model_service({"queries": ["x"]}))
    with pytest.raises(ConfigurationError, match="forbids model-backed cleaners"):
        code_terms(
            bundle,
            csv_path,
            tmp_path / "out_bad",
            text_column="term",
            cleaners=[model_cleaner],
        )

    td_profile = TaskProfile.model_validate(
        {
            "profile_schema": 2,
            "profile_name": "td_coding",
            "profile_version": "0.1.0",
            "workflow": "coding",
            "task": {"operation": "disease_coding", "mapping_intent": "coding_only"},
            "target": {"terminology": "ToyDisease", "level": None},
            "knowledge": {"resources": ["ToyDisease"]},
            "tiers": [
                {
                    "name": "exact",
                    "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
                }
            ],
        }
    )
    td_report = code_terms(
        bundle, csv_path, tmp_path / "out_td", text_column="term", profile=td_profile
    )
    assert td_report.coded_terms == 1
    assert td_report.coded_terms_exact == 1
