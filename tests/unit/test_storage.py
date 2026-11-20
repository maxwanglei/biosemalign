from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel

from biosemalign.exceptions import ConfigurationError
from biosemalign.storage.jsonl import (
    atomic_text_writer,
    read_jsonl,
    read_models,
    write_jsonl,
    write_models,
)
from biosemalign.storage.parquet import CROSSWALK_COLUMNS, decision_to_row, write_csv, write_parquet
from biosemalign.storage.paths import ensure_directory, reject_symlink_components


class Row(BaseModel):
    identifier: str
    note: str | None = None


def test_jsonl_model_roundtrip_and_blank_lines(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    values = [Row(identifier="one", note="café"), Row(identifier="two")]
    assert write_models(path, values) == 2
    assert list(read_models(path, Row)) == values
    assert '"note":null' in path.read_text()
    path.write_text("\n" + path.read_text() + "\n")
    assert len(list(read_jsonl(path))) == 2
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_plain_jsonl_serializes_unicode_and_path_values(tmp_path):
    path = tmp_path / "records.jsonl"
    assert write_jsonl(path, [{"value": "café", "location": Path("fixture")}]) == 1
    assert list(read_jsonl(path)) == [{"value": "café", "location": "fixture"}]
    path.write_text("{}\n\n{broken\n")
    with pytest.raises(ValueError, match=r":3 is not valid JSON"):
        list(read_jsonl(path))


@pytest.mark.parametrize("writer", [write_jsonl, write_models])
def test_failed_stream_does_not_replace_existing_jsonl(tmp_path, writer):
    path = tmp_path / "records.jsonl"
    path.write_text("existing\n")

    def failing():
        yield Row(identifier="one") if writer is write_models else {"identifier": "one"}
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        writer(path, failing())
    assert path.read_text() == "existing\n"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_writer_replace_failure_cleans_temporary_file(tmp_path, monkeypatch):
    import biosemalign.storage.jsonl as module

    path = tmp_path / "result"
    path.write_text("original")
    monkeypatch.setattr(
        module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure"))
    )
    with pytest.raises(OSError, match="disk failure"), atomic_text_writer(path) as handle:
        handle.write("replacement")
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_decision_crosswalk_preserves_original_and_provenance(artifact):
    row = decision_to_row(artifact.decision)
    assert tuple(row) == CROSSWALK_COLUMNS
    assert row["reported_value"] == artifact.decision.source_value
    assert row["run_id"] == artifact.run_id
    assert row["selected_ids"] == "|".join(
        c.concept_id for c in artifact.decision.selected_concepts
    )
    assert row["vocabulary_release"]
    empty = artifact.decision.model_copy(
        update={
            "source_level_concept": None,
            "analysis_level_concept": None,
            "selected_concepts": [],
            "routing_reasons": [],
        }
    )
    row = decision_to_row(empty)
    assert row["source_level_id"] is None and row["analysis_level_id"] is None
    assert row["selected_ids"] == "" and row["review_reasons"] == ""


def test_csv_escapes_spreadsheet_formulas_and_has_stable_columns(tmp_path):
    path = tmp_path / "crosswalk.csv"
    values = ["=1+1", " +cmd", "-1", "@value", "ordinary", None]
    assert write_csv(
        path,
        ({"reported_value": value, "record_id": "one", "extra": "ignored"} for value in values),
    ) == len(values)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(CROSSWALK_COLUMNS)
    assert [row["reported_value"] for row in rows] == [
        "'=1+1",
        "' +cmd",
        "'-1",
        "'@value",
        "ordinary",
        "",
    ]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("writer", [write_csv, write_parquet])
def test_flat_writers_preserve_old_output_when_source_fails(tmp_path, writer):
    path = tmp_path / "crosswalk"
    path.write_text("old")

    def failing():
        yield {"reported_value": "ExampleAlpha"}
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        writer(path, failing())
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_parquet_batches_empty_schema_and_typed_values(tmp_path):
    path = tmp_path / "crosswalk.parquet"
    values = [
        {"reported_value": value, "record_id": str(index)}
        for index, value in enumerate(["=value", "café", None])
    ]
    assert write_parquet(path, values, batch_size=2) == 3
    table = pq.read_table(path)
    assert table.column_names == list(CROSSWALK_COLUMNS)
    assert table.column("reported_value").to_pylist() == ["=value", "café", None]
    assert table.column("record_id").to_pylist() == ["0", "1", "2"]
    assert write_parquet(path, []) == 0
    assert pq.read_table(path).num_rows == 0
    with pytest.raises(ValueError, match="batch_size"):
        write_parquet(path, [], batch_size=0)


def test_private_directory_creation_and_symlink_rejection(tmp_path):
    nested = tmp_path / "private" / "nested"
    ensure_directory(nested, 0o700)
    assert nested.parent.stat().st_mode & 0o777 == 0o700
    assert nested.stat().st_mode & 0o777 == 0o700
    link = tmp_path / "link"
    link.symlink_to(nested, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symbolic link"):
        reject_symlink_components(link / "output")
    with pytest.raises(ConfigurationError, match="symbolic link"):
        write_jsonl(link / "output", [{"private": True}])
