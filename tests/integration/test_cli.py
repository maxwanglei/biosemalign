from __future__ import annotations

import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer
from tests.support import ALPHA, NO_MATCH, PRODUCT, SYNTHETIC_RELEASE, make_pipeline
from typer.testing import CliRunner

from biosemalign import cli
from biosemalign.exceptions import ConfigurationError
from biosemalign.run_manifest import file_sha256
from biosemalign.schemas.artifact import BatchAlignmentRecord
from biosemalign.storage.jsonl import read_jsonl, read_models, write_jsonl, write_models

runner = CliRunner()


@pytest.fixture
def local_cli(tmp_path, monkeypatch):
    control = {"interrupt": False, "calls": 0}

    def factory(*args, **kwargs):
        pipeline = make_pipeline(tmp_path / "cache")
        original = pipeline.align_batch_records

        def records(*arguments, **options):
            for record in original(*arguments, **options):
                control["calls"] += 1
                yield record
                if control["interrupt"]:
                    control["interrupt"] = False
                    raise KeyboardInterrupt("synthetic interrupted batch")

        pipeline.align_batch_records = records
        return pipeline

    monkeypatch.setattr(cli, "_pipeline", factory)
    return control


def csv_input(path, rows, fields=("value", "id", "route")):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return path


def batch_args(input_path, output, *extra):
    return [
        "align-batch",
        "-p",
        "faers_drug_normalization",
        "-i",
        str(input_path),
        "-o",
        str(output),
        "--value-column",
        "value",
        "--id-columns",
        "id",
        "--context-columns",
        "route",
        *extra,
    ]


def output_paths(output):
    crosswalk = output if output.suffix == ".parquet" else output.with_suffix(".csv")
    artifacts = crosswalk.with_suffix(".jsonl")
    return (
        crosswalk,
        artifacts,
        crosswalk.with_suffix(".manifest.json"),
        artifacts.with_name(artifacts.name + ".partial"),
        artifacts.with_name(artifacts.name + ".checkpoint.json"),
    )


def test_version_profiles_and_empty_profile_registry(monkeypatch):
    assert runner.invoke(cli.app, ["version"]).exit_code == 0
    result = runner.invoke(cli.app, ["profiles"])
    assert result.exit_code == 0 and "faers_drug_normalization" in result.stdout
    monkeypatch.setattr(cli, "available_profiles", lambda: [])
    result = runner.invoke(cli.app, ["profiles"])
    assert result.exit_code == 1 and "no profiles" in result.stdout


def test_pipeline_configuration_error_becomes_actionable_cli_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise ConfigurationError("synthetic invalid configuration")

    monkeypatch.setattr(cli.BioSemAlign, "from_profile", fail)
    result = runner.invoke(cli.app, ["align-text", "-p", "invalid", "-t", ALPHA])
    assert result.exit_code == 2
    assert "synthetic invalid configuration" in result.output


def test_align_text_prints_or_writes_decision_and_validates_context(tmp_path, local_cli):
    result = runner.invoke(
        cli.app, ["align-text", "-p", "faers_drug_normalization", "-t", ALPHA, "-c", "route=oral"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["selected_concepts"][0]["concept_id"] == "1001"
    output = tmp_path / "nested" / "decision.json"
    result = runner.invoke(
        cli.app, ["align-text", "-p", "faers_drug_normalization", "-t", ALPHA, "-o", str(output)]
    )
    assert result.exit_code == 0 and "wrote" in result.stdout
    assert json.loads(output.read_text())["source_value"] == ALPHA
    result = runner.invoke(
        cli.app, ["align-text", "-p", "faers_drug_normalization", "-t", ALPHA, "-c", "invalid"]
    )
    assert result.exit_code == 2 and "key=value" in result.output


def test_inspect_shows_candidates_evidence_validation_and_provenance(local_cli):
    result = runner.invoke(
        cli.app,
        [
            "inspect",
            "-p",
            "faers_drug_normalization",
            "-t",
            PRODUCT,
            "--show-knowledge",
            "--show-validation",
        ],
    )
    assert result.exit_code == 0, result.output
    for content in (
        "Candidates",
        "Coding policies",
        "Decision",
        "Validator findings",
        "Provenance",
        SYNTHETIC_RELEASE,
    ):
        assert content in result.stdout


def test_batch_csv_preserves_all_row_outcomes_and_manifest_hashes(tmp_path, local_cli):
    source = csv_input(
        tmp_path / "input.csv",
        [
            {"value": ALPHA, "id": "one", "route": "oral"},
            {"value": "", "id": "two", "route": "oral"},
            {"value": NO_MATCH, "id": "three", "route": "oral"},
            {"value": ALPHA, "id": "", "route": "oral"},
        ],
    )
    output = tmp_path / "out" / "crosswalk.csv"
    result = runner.invoke(cli.app, batch_args(source, output, "--checkpoint-every", "1"))
    assert result.exit_code == 1, result.output
    crosswalk, artifacts, manifest, partial, checkpoint = output_paths(output)
    records = list(read_models(artifacts, BatchAlignmentRecord))
    assert [record.status for record in records] == ["success", "skipped", "success", "error"]
    assert [record.row_number for record in records] == [1, 2, 3, 4]
    assert records[0].artifact.decision.selected_concepts[0].concept_id == "1001"
    with crosswalk.open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    payload = json.loads(manifest.read_text())
    assert payload["record_counts"] == {"success": 2, "skipped": 1, "error": 1}
    assert payload["input_sha256"] == file_sha256(source)
    assert sorted(payload["output_sha256"].values()) == sorted(
        [file_sha256(crosswalk), file_sha256(artifacts)]
    )
    assert not partial.exists() and not checkpoint.exists()
    assert "outputs were finalized" in result.output


def test_jsonl_batch_records_malformed_and_nonobject_rows(tmp_path, local_cli):
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({"value": ALPHA, "id": "one", "route": "oral"})
        + '\n\n{broken\n[]\n{"id":"missing","route":"oral"}\n'
    )
    output = tmp_path / "crosswalk.csv"
    result = runner.invoke(cli.app, batch_args(source, output))
    assert result.exit_code == 1, result.output
    records = list(read_models(output.with_suffix(".jsonl"), BatchAlignmentRecord))
    assert [record.status for record in records] == ["success", "error", "error", "error"]
    assert records[1].error_code == "JSONDecodeError"
    assert records[2].error_code == "UnsupportedInput"


def test_parquet_batch_input_and_output(tmp_path, local_cli):
    source = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"value": PRODUCT, "id": "one", "route": "oral"}]), source)
    output = tmp_path / "crosswalk.parquet"
    result = runner.invoke(cli.app, batch_args(source, output))
    assert result.exit_code == 0, result.output
    assert pq.read_table(output).column("analysis_level_id").to_pylist() == ["1001"]


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate_header", "duplicate_option", "alias", "symlink"]
)
def test_batch_rejects_invalid_inputs_before_creating_outputs(tmp_path, kind, local_cli):
    source = tmp_path / "input.csv"
    source.write_text("value,id,route\nExampleAlpha,one,oral\n")
    output = tmp_path / "result.csv"
    extra = []
    if kind == "missing":
        source.write_text("value,id\nExampleAlpha,one\n")
    elif kind == "duplicate_header":
        source.write_text("value,id,route,route\nExampleAlpha,one,oral,oral\n")
    elif kind == "duplicate_option":
        extra = ["--context-columns", "id"]
    elif kind == "alias":
        output = source
    else:
        output.symlink_to(tmp_path / "outside.csv")
    result = runner.invoke(cli.app, batch_args(source, output, *extra))
    assert result.exit_code == 2, result.output
    assert not output.with_suffix(".manifest.json").exists()
    assert not output.with_suffix(".jsonl").exists()


def test_empty_batch_has_no_commit_marker(tmp_path, local_cli):
    source = csv_input(tmp_path / "empty.csv", [])
    output = tmp_path / "empty-out.csv"
    result = runner.invoke(cli.app, batch_args(source, output))
    assert result.exit_code == 1 and "no rows" in result.output
    assert not output.with_suffix(".manifest.json").exists()


def interrupted_batch(tmp_path, control):
    source = csv_input(
        tmp_path / "input.csv",
        [{"value": ALPHA, "id": str(index), "route": "oral"} for index in range(3)],
    )
    output = tmp_path / "crosswalk.csv"
    control["interrupt"] = True
    result = runner.invoke(cli.app, batch_args(source, output, "--checkpoint-every", "1"))
    assert result.exit_code != 0
    paths = output_paths(output)
    assert paths[3].is_file() and paths[4].is_file()
    assert not paths[2].exists()
    return source, output, paths


def test_resume_keeps_durable_record_and_run_identity_without_reprocessing(tmp_path, local_cli):
    source, output, (_, artifacts, manifest, partial, checkpoint) = interrupted_batch(
        tmp_path, local_cli
    )
    first = next(iter(read_jsonl(partial)))
    state = json.loads(checkpoint.read_text())
    assert state["processed_rows"] == 1
    result = runner.invoke(cli.app, batch_args(source, output))
    assert result.exit_code == 2 and "partial output exists" in result.output
    result = runner.invoke(
        cli.app, batch_args(source, output, "--resume", "--checkpoint-every", "1")
    )
    assert result.exit_code == 0, result.output
    records = list(read_jsonl(artifacts))
    assert len(records) == 3 and local_cli["calls"] == 3
    assert records[0] == first
    assert {record["artifact"]["run_id"] for record in records} == {state["run_id"]}
    assert json.loads(manifest.read_text())["record_counts"] == {"success": 3}
    assert not partial.exists() and not checkpoint.exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("input", "input content changed"),
        ("profile", "profile changed"),
        ("configuration", "configuration changed"),
        ("ahead", "checkpoint is ahead"),
        ("invalid_count", "row count is invalid"),
        ("run", "different run"),
        ("missing_partial", "does not exist"),
        ("corrupt_partial", "partial output is corrupt"),
    ],
)
def test_resume_rejects_changed_or_corrupt_run(tmp_path, local_cli, mutation, message):
    source, output, (_, _, manifest, partial, checkpoint) = interrupted_batch(tmp_path, local_cli)
    state = json.loads(checkpoint.read_text())
    if mutation == "input":
        source.write_text(source.read_text() + f"{ALPHA},extra,oral\n")
    elif mutation == "missing_partial":
        partial.unlink()
    elif mutation == "corrupt_partial":
        partial.write_text("{broken\n")
    else:
        field, value = {
            "profile": ("profile_sha256", "wrong"),
            "configuration": ("configuration_sha256", "wrong"),
            "ahead": ("processed_rows", 99),
            "invalid_count": ("processed_rows", "invalid"),
            "run": ("run_id", "different"),
        }[mutation]
        state[field] = value
        checkpoint.write_text(json.dumps(state))
    result = runner.invoke(cli.app, batch_args(source, output, "--resume"))
    assert result.exit_code == 2, result.output
    assert message in " ".join(line.strip("│ ") for line in result.output.splitlines())
    assert not manifest.exists()


def test_resume_recovers_when_partial_is_ahead_of_checkpoint(tmp_path, local_cli):
    source, output, (_, artifacts, _, _, checkpoint) = interrupted_batch(tmp_path, local_cli)
    state = json.loads(checkpoint.read_text())
    state["processed_rows"] = 0
    checkpoint.write_text(json.dumps(state))
    result = runner.invoke(cli.app, batch_args(source, output, "--resume"))
    assert result.exit_code == 0, result.output
    assert len(list(read_jsonl(artifacts))) == 3
    assert local_cli["calls"] == 3


@pytest.mark.parametrize("payload", [None, "{broken", "[]", '{"checkpoint_version":"wrong"}'])
def test_checkpoint_reader_rejects_missing_malformed_or_unknown_version(tmp_path, payload):
    path = tmp_path / "checkpoint.json"
    if payload is not None:
        path.write_text(payload)
    with pytest.raises(typer.BadParameter, match="cannot resume"):
        cli._load_checkpoint(path)


def test_input_hash_guard_detects_changed_and_removed_source(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("original")
    digest = file_sha256(path)
    path.write_text("changed")
    with pytest.raises(typer.BadParameter, match="content changed"):
        cli._require_unchanged_input(path, expected_sha256=digest)
    path.unlink()
    with pytest.raises(typer.BadParameter, match="became unavailable"):
        cli._require_unchanged_input(path, expected_sha256=digest)


def test_partial_requires_contiguous_rows_and_single_run(tmp_path):
    with make_pipeline(tmp_path / "cache") as pipeline:
        records = list(
            pipeline.align_batch_records([{"value": ALPHA}, {"value": ALPHA}], value_column="value")
        )
        other = next(
            pipeline.align_batch_records([{"value": ALPHA}], value_column="value", start_row=2)
        )
    path = tmp_path / "partial.jsonl"
    write_models(path, [records[1]])
    with pytest.raises(typer.BadParameter, match="not contiguous"):
        cli._scan_partial(path)
    write_models(path, [records[0], other])
    with pytest.raises(typer.BadParameter, match="run IDs"):
        cli._scan_partial(path)


def test_resource_progress_and_checkpoint_validation(artifact):
    resource = artifact.decision.provenance.resources[0].model_copy(
        update={"cache_hits": 10, "network_calls": 2}
    )
    cli._validate_resource_progress(
        [resource], [resource.model_copy(update={"cache_hits": 11})], row_number=2
    )
    for changed, message in [
        ([], "disappears"),
        ([resource.model_copy(update={"terminology_release": "changed"})], "identity changes"),
        ([resource.model_copy(update={"cache_hits": 9})], "counters decrease"),
    ]:
        with pytest.raises(typer.BadParameter, match=message):
            cli._validate_resource_progress([resource], changed, row_number=2)
    assert cli._checkpoint_resources({}, fallback=[resource]) == [resource]
    for payload in ({"resource_provenance": {}}, {"resource_provenance": [{"bad": "record"}]}):
        with pytest.raises(typer.BadParameter, match="resource provenance is invalid"):
            cli._checkpoint_resources(payload, fallback=[])
    with pytest.raises(typer.BadParameter, match="mixed-resource run"):
        cli._merged_resource_payload(
            [resource], [resource.model_copy(update={"terminology_release": "changed"})]
        )


def test_evaluate_full_artifacts_checks_context_and_writes_report(tmp_path, artifact):
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(
        gold,
        [
            {
                "case_id": artifact.decision.sau_id,
                "source_value": artifact.decision.source_value,
                "context": {"route": artifact.sau.attributes.route},
                "expected_concept_ids": ["1001"],
            }
        ],
    )
    write_models(predictions, [artifact])
    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["complete"] is True
    write_models(predictions, [artifact.decision])
    result = runner.invoke(
        cli.app, ["evaluate", "--gold", str(gold), "--predictions", str(predictions)]
    )
    assert result.exit_code == 2 and "invalid evaluation input" in result.output


def test_evaluate_missing_duplicate_and_legacy_predictions(tmp_path, artifact):
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        gold,
        [
            {
                "case_id": artifact.decision.sau_id,
                "source_value": artifact.decision.source_value,
                "expected_concept_ids": ["1001"],
            }
        ],
    )
    write_models(predictions, [artifact.decision])
    arguments = ["evaluate", "--gold", str(gold), "--predictions", str(predictions)]
    assert runner.invoke(cli.app, arguments).exit_code == 0
    predictions.write_text("")
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 1 and "incomplete" in result.output
    write_models(predictions, [artifact.decision, artifact.decision])
    result = runner.invoke(cli.app, arguments)
    assert result.exit_code == 2 and "invalid evaluation input" in result.output


def test_cache_info_reports_counts_and_releases(tmp_path, synthetic_cache):
    result = runner.invoke(cli.app, ["cache-info", "--cache-dir", str(synthetic_cache.root)])
    assert result.exit_code == 0, result.output
    assert "entries:" in result.stdout
    assert SYNTHETIC_RELEASE in result.stdout
    assert "rxnav." in result.stdout


def test_resume_finishes_after_crosswalk_write_failure_without_reprocessing(
    tmp_path, local_cli, monkeypatch
):
    source = csv_input(tmp_path / "input.csv", [{"value": ALPHA, "id": "one", "route": "oral"}])
    output = tmp_path / "crosswalk.csv"
    _, artifacts, manifest, partial, checkpoint = output_paths(output)
    manifest.write_text('{"old":"commit marker"}')
    original = cli.write_csv

    def fail(*args, **kwargs):
        raise OSError("synthetic crosswalk disk failure")

    monkeypatch.setattr(cli, "write_csv", fail)
    result = runner.invoke(cli.app, batch_args(source, output))
    assert result.exit_code != 0
    assert not manifest.exists()
    assert partial.is_file() and checkpoint.is_file()
    assert json.loads(checkpoint.read_text())["processed_rows"] == 1
    monkeypatch.setattr(cli, "write_csv", original)
    result = runner.invoke(cli.app, batch_args(source, output, "--resume"))
    assert result.exit_code == 0, result.output
    assert local_cli["calls"] == 1
    assert len(list(read_jsonl(artifacts))) == 1
    assert manifest.is_file() and not partial.exists() and not checkpoint.exists()


def test_evaluate_batch_records_ignores_skipped_rows_and_retains_context(tmp_path):
    with make_pipeline(tmp_path / "cache") as pipeline:
        records = list(
            pipeline.align_batch_records(
                [
                    {"value": ALPHA, "id": "one", "route": "oral"},
                    {"value": "", "id": "two", "route": "oral"},
                ],
                value_column="value",
                id_columns=["id"],
                context_columns=["route"],
            )
        )
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        gold,
        [
            {
                "case_id": "one",
                "source_value": ALPHA,
                "context": {"route": "oral"},
                "expected_concept_ids": ["1001"],
            }
        ],
    )
    write_models(predictions, records)
    result = runner.invoke(
        cli.app, ["evaluate", "--gold", str(gold), "--predictions", str(predictions)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["complete"] is True


def test_inspect_unresolved_input_and_summary(local_cli, artifact, capsys):
    result = runner.invoke(
        cli.app, ["inspect", "-p", "faers_drug_normalization", "-t", NO_MATCH, "--show-validation"]
    )
    assert result.exit_code == 0, result.output
    assert "Candidates (0)" in result.stdout
    assert "(none)" in result.stdout
    cli._print_summary([artifact.decision])
    assert "routing:" in capsys.readouterr().out
