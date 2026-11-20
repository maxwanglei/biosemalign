"""Regression snapshots preserve current semantics, including limitations."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from tests.support import (
    ALPHA,
    COMBINATION,
    HISTORICAL,
    PRODUCT,
    SYNTHETIC_RELEASE,
    baseline_rows,
    canonicalize,
    make_pipeline,
)

from biosemalign import BioSemAlign
from biosemalign.input.adapters.existing import ExistingCleanerAdapter
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.util import stable_digest

GOLDEN = Path(__file__).parent


def test_existing_cleaner_matches_direct_pipeline_for_every_row(tmp_path: Path) -> None:
    with (
        make_pipeline(tmp_path / "direct") as direct,
        make_pipeline(tmp_path / "wrapped") as wrapped,
    ):
        options = {
            "value_column": "drugname",
            "id_columns": ["record_id"],
            "context_columns": ["route"],
        }
        direct_records = list(direct.align_batch_records(baseline_rows(), **options))
        wrapped_records = list(
            ExistingCleanerAdapter(wrapped).clean_records(baseline_rows(), **options)
        )
    actual = [canonicalize(record, temp_root=tmp_path / "direct") for record in direct_records]
    adapted = [canonicalize(record, temp_root=tmp_path / "wrapped") for record in wrapped_records]
    assert adapted == actual
    assert actual == json.loads((GOLDEN / "expected_records.json").read_text())
    assert len(actual) == len(baseline_rows()) == 20
    assert [record["status"] for record in actual].count("success") == 13
    assert [record["status"] for record in actual].count("skipped") == 1
    assert [record["status"] for record in actual].count("error") == 6
    assert actual[13]["error_code"] == "CacheMiss"
    assert actual[14]["artifact"]["decision"]["selected_concepts"][0]["concept_id"] == "1002"


def test_baseline_manifest_records_input_profile_and_resource_fingerprints(
    synthetic_pipeline: BioSemAlign,
) -> None:
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    inputs = [
        dataclasses.asdict(row) if dataclasses.is_dataclass(row) else row for row in baseline_rows()
    ]
    assert manifest["input_sha256"] == stable_digest(inputs)
    assert manifest["profile_sha256"] == stable_digest(
        synthetic_pipeline.profile.model_dump(mode="json")
    )
    assert manifest["terminology_release"] == SYNTHETIC_RELEASE
    assert manifest["expected_records_sha256"] == stable_digest(
        json.loads((GOLDEN / "expected_records.json").read_text())
    )
    assert manifest["source_release_validation"] == "not implemented"
    assert (
        manifest["fixture_provenance"]
        == "invented offline responses; no terminology or patient data"
    )
    responses = []
    for path in sorted(synthetic_pipeline.knowledge.cache.root.rglob("*.json")):
        response = json.loads(path.read_text())
        response.pop("stored_at")
        response.pop("checksum")
        responses.append(response)
    assert manifest["resource_fixture_sha256"] == stable_digest(responses)


def test_adapter_forwards_start_row_run_and_retains_caller_lifecycle(
    synthetic_pipeline: BioSemAlign,
) -> None:
    options = {"value_column": "drug", "start_row": 42}
    rows = [{"drug": ALPHA}, {"drug": PRODUCT}]
    direct = list(
        synthetic_pipeline.align_batch_records(
            rows, run=synthetic_pipeline.new_run(run_id="supplied-run"), **options
        )
    )
    adapted = list(
        ExistingCleanerAdapter(synthetic_pipeline).clean_records(
            rows, run=synthetic_pipeline.new_run(run_id="supplied-run"), **options
        )
    )
    assert [record.row_number for record in adapted] == [42, 43]
    assert [record.artifact.run_id for record in adapted] == ["supplied-run", "supplied-run"]
    assert canonicalize(adapted) == canonicalize(direct)
    assert synthetic_pipeline.align_value(ALPHA).selected_concepts[0].concept_id == "1001"


@pytest.mark.parametrize("value", [ALPHA, PRODUCT, COMBINATION, HISTORICAL])
def test_repeated_frozen_input_preserves_semantic_output(
    value: str, synthetic_pipeline: BioSemAlign
) -> None:
    request = synthetic_pipeline.make_request(value, route="oral")
    first = synthetic_pipeline.align_artifact(request)
    second = synthetic_pipeline.align_artifact(request)
    assert first.artifact_id != second.artifact_id
    assert canonicalize(first) == canonicalize(second)
    assert first.sau.sau_id == second.sau.sau_id
    assert first.mapping_key == second.mapping_key


def test_snapshot_normalization_preserves_source_fields_and_stable_identifiers(
    artifact: AlignmentArtifact,
) -> None:
    payload = artifact.model_dump(mode="json")
    payload["sau"]["source"]["row"].update(
        {"run_id": "reported-run", "created_at": "reported-date", "artifact_id": "reported-id"}
    )
    snapshot = canonicalize(payload)
    assert snapshot["artifact_id"] == "<artifact_id>"
    assert snapshot["sau"]["sau_id"] == artifact.sau.sau_id
    assert snapshot["mapping_key"] == artifact.mapping_key
    assert snapshot["sau"]["source"]["row"]["run_id"] == "reported-run"
    assert snapshot["sau"]["source"]["row"]["created_at"] == "reported-date"
    assert snapshot["sau"]["source"]["row"]["artifact_id"] == "reported-id"
