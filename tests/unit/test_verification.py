"""Consumer-side artifact verification, from Python and the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.support import ALPHA, NO_MATCH, make_pipeline
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.enums import Authority
from biosemalign.profiles.loader import load_profile
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.knowledge import KnowledgeStatement
from biosemalign.storage.jsonl import write_models
from biosemalign.util import stable_id


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    payload["artifact_id"] = stable_id(
        "ART",
        {
            name: payload[name]
            for name in ("run_id", "sau", "knowledge_package", "proposal", "validation", "decision")
        },
        length=32,
    )
    return payload


def test_kernel_artifacts_verify_with_and_without_their_profile(
    artifact: AlignmentArtifact,
) -> None:
    assert verify_artifact(artifact).valid
    result = verify_artifact(artifact, profile=load_profile("faers_drug_normalization"))
    assert result.valid
    assert result.artifact_id == artifact.artifact_id


def test_tampered_decision_fails_verification(artifact: AlignmentArtifact) -> None:
    payload = artifact.model_dump(mode="json")
    payload["decision"]["routing_status"] = "AUTO_ACCEPT"
    payload["decision"]["selected_concepts"][0]["label"] = "Something else"
    result = verify_artifact(_resign(payload))
    assert not result.valid
    assert any("canonical" in error for error in result.errors)


def test_unsigned_edit_fails_the_content_hash(artifact: AlignmentArtifact) -> None:
    payload = artifact.model_dump(mode="json")
    payload["sau"]["source"]["raw_value"] = "edited"
    payload["decision"]["source_value"] = "edited"
    result = verify_artifact(payload)
    assert not result.valid


def test_profile_checks_catch_wrong_profile_and_leaked_predicates(
    artifact: AlignmentArtifact,
) -> None:
    other = load_profile("faers_drug_normalization")
    other.profile_version = "9.9.9"
    assert any(
        "not faers_drug_normalization@9.9.9" in e
        for e in verify_artifact(artifact, profile=other).errors
    )

    payload = artifact.model_dump(mode="json")
    leak = KnowledgeStatement(
        subject_id="1001",
        predicate="has_adverse_effect",
        object_id="outcome",
        source_resource="RxNorm",
        authority=Authority.TARGET_TERMINOLOGY,
    )
    payload["knowledge_package"]["kg_guidance"].append(leak.model_dump(mode="json"))
    result = verify_artifact(_resign(payload), profile=load_profile("faers_drug_normalization"))
    assert result.errors == [
        "knowledge package contains predicates the profile prohibits: ['has_adverse_effect']"
    ]


def test_cli_verifies_batch_records_and_reports_tampering(
    synthetic_pipeline: BioSemAlign, tmp_path: Path
) -> None:
    records = list(
        synthetic_pipeline.align_batch_records(
            [{"drug": ALPHA}, {"drug": "  "}, {"drug": NO_MATCH}], value_column="drug"
        )
    )
    path = tmp_path / "records.jsonl"
    write_models(path, records)
    runner = CliRunner()

    result = runner.invoke(app, ["verify", "-i", str(path), "-p", "faers_drug_normalization"])
    assert result.exit_code == 0, result.output
    assert "verified: 2" in result.output
    assert "records without an artifact: 1" in result.output

    lines = path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["artifact"]["decision"]["routing_status"] = "ABSTAIN"
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n")
    result = runner.invoke(app, ["verify", "-i", str(path)])
    assert result.exit_code == 1
    assert "invalid: 1" in result.output
    assert "line 1 (row 1)" in result.output


def test_cli_accepts_bare_artifact_lines(artifact: AlignmentArtifact, tmp_path: Path) -> None:
    path = tmp_path / "artifacts.jsonl"
    write_models(path, [artifact])
    result = CliRunner().invoke(app, ["verify", "--input", str(path)])
    assert result.exit_code == 0, result.output
    assert "verified: 1" in result.output


def test_two_tier_artifacts_verify(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    payload = profile.model_dump(mode="json")
    payload["workflow"] = {"tiers": [{"name": "first"}, {"name": "second"}]}
    profile = type(profile).model_validate(payload)
    with make_pipeline(tmp_path, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(ALPHA))
    assert verify_artifact(artifact, profile=profile).valid


LEGACY_RECORDS = Path(__file__).parents[1] / "golden" / "schema_0_2_0_records.jsonl"
SCHEMA_0_3_0_RECORDS = Path(__file__).parents[1] / "golden" / "schema_0_3_0_records.jsonl"


def test_artifacts_written_under_schema_0_2_0_still_verify() -> None:
    # Produced by the v0.1 pipeline (core schema 0.2.0) from the synthetic
    # fixtures. Fields added in 0.3.0 must not invalidate their content hashes.
    from biosemalign.verification import verify_jsonl

    results = list(verify_jsonl(LEGACY_RECORDS, profile=load_profile("faers_drug_normalization")))
    assert len(results) == 3
    assert all(result is not None and result.valid for result in results)
    first = json.loads(LEGACY_RECORDS.read_text().splitlines()[0])["artifact"]
    assert first["schema_version"] == "0.2.0"
    assert "workflow" not in first["decision"]["provenance"]


def test_legacy_stamp_does_not_excuse_new_fields() -> None:
    payload = json.loads(LEGACY_RECORDS.read_text().splitlines()[0])["artifact"]
    payload["decision"]["provenance"]["resolution"] = {
        "attempts": [{"tier": "default", "outcome": "RESOLVED"}],
        "resolved_by_tier": "default",
    }
    result = verify_artifact(payload)
    assert any("artifact_id does not match" in error for error in result.errors)


def test_artifacts_written_under_schema_0_3_0_verify() -> None:
    from biosemalign.verification import verify_jsonl

    results = list(verify_jsonl(SCHEMA_0_3_0_RECORDS))
    assert len(results) == 5
    assert all(result is not None and result.valid for result in results)

    artifacts = [
        json.loads(line)["artifact"]
        for line in SCHEMA_0_3_0_RECORDS.read_text(encoding="utf-8").splitlines()
    ]
    assert all(art["schema_version"] == "0.3.0" for art in artifacts)

    workflow = artifacts[0]["decision"]["provenance"]["workflow"]
    assert workflow["target_spec"] == "rxnorm@1.0"
    tier_kinds = {op["kind"] for tier in workflow["tiers"] for op in tier["operators"]}
    validator_kinds = {op["kind"] for op in workflow["validators"]}
    assert {"transform", "retrieve", "fuse", "propose"} <= tier_kinds
    assert "validate" in validator_kinds

    for idx in (0, 1, 2):
        pkg = artifacts[idx]["knowledge_package"]
        assert pkg["concordance"] is not None
        assert pkg["links"] is not None
        assert pkg["links"]["target_spec"] == "rxnorm@1.0"

    cleaned_row = artifacts[3]
    assert cleaned_row["sau"]["mention"]["retrieval_text"] == "TOYBRAND XL"
    assert cleaned_row["sau"]["attributes"]["strength"] == "75 MG"
    attempts = cleaned_row["decision"]["provenance"]["resolution"]["attempts"]
    assert [attempt["tier"] for attempt in attempts] == ["exact", "cleaned"]


def test_schema_0_3_0_tampering_fails_verification() -> None:
    payload = json.loads(SCHEMA_0_3_0_RECORDS.read_text(encoding="utf-8").splitlines()[0])[
        "artifact"
    ]
    payload["decision"]["selected_concepts"][0]["concept_id"] = "T999"
    assert not verify_artifact(payload).valid

    forged = json.loads(SCHEMA_0_3_0_RECORDS.read_text(encoding="utf-8").splitlines()[0])[
        "artifact"
    ]
    forged["knowledge_package"]["links"]["links"][0]["target"]["concept_code"] = "T300"
    result = verify_artifact(_resign(forged))
    assert not result.valid
    assert any("links" in error for error in result.errors)
