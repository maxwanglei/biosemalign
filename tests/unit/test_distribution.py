from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    specification = importlib.util.spec_from_file_location(
        f"biosemalign_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


checker = load_script("check_distribution")
canary = load_script("rxnorm_release_canary")


def make_wheel(
    path, *, extra=None, metadata="[console_scripts]\nbiosemalign = biosemalign.cli:main\n", omit=()
):
    members = dict.fromkeys(checker.WHEEL_REQUIRED - set(omit), "synthetic")
    if metadata is not None:
        members["biosemalign-0.0.dist-info/entry_points.txt"] = metadata
    members.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


def test_distribution_accepts_complete_wheel_and_reports_missing_files(tmp_path):
    wheel = make_wheel(tmp_path / "complete.whl")
    assert checker.check_distribution(wheel) == []
    wheel = make_wheel(tmp_path / "incomplete.whl", omit={"biosemalign/py.typed"})
    assert "missing required file: biosemalign/py.typed" in checker.check_distribution(wheel)
    assert "missing console-script metadata" in " ".join(
        checker.check_distribution(make_wheel(tmp_path / "missing.whl", metadata=None))
    )
    assert "unsafe console-script target" in " ".join(
        checker.check_distribution(
            make_wheel(tmp_path / "unsafe.whl", metadata="biosemalign = unsafe:main")
        )
    )


@pytest.mark.parametrize(
    "name, message",
    [
        ("../escape", "unsafe archive member"),
        ("/absolute", "unsafe archive member"),
        ("tests/test_leak.py", "wheel must not contain"),
        ("data/restricted/file.txt", "forbidden raw/restricted path"),
        ("tests/fixtures/sample.txt", "forbidden test fixture"),
        ("cache.sqlite", "forbidden data artifact"),
        ("settings/.env.secret", "forbidden environment file"),
    ],
)
def test_distribution_rejects_unsafe_or_restricted_members(tmp_path, name, message):
    wheel = make_wheel(tmp_path / "unsafe.whl", extra={name: "synthetic"})
    assert message in " ".join(checker.check_distribution(wheel))


def test_distribution_accepts_sdist_prefix_and_reads_tar_member(tmp_path):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in checker.SDIST_REQUIRED:
            value = b"synthetic"
            member = tarfile.TarInfo(f"biosemalign-0.0/{name}")
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    assert checker.check_distribution(path) == []
    assert checker._archive_text(path, "biosemalign-0.0/README.md") == "synthetic"
    assert checker._without_sdist_root({"one/a", "two/b"}) == {"one/a", "two/b"}
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("not an archive")
    assert "not a supported" in checker.check_distribution(invalid)[0]


def test_distribution_command_reports_success_and_policy_failure(tmp_path, capsys):
    valid = make_wheel(tmp_path / "valid.whl")
    invalid = make_wheel(tmp_path / "invalid.whl", extra={"file.parquet": "synthetic"})
    assert checker.main([str(valid)]) == 0
    assert "checked" in capsys.readouterr().out
    assert checker.main([str(valid), str(invalid)]) == 1
    captured = capsys.readouterr()
    assert "forbidden data artifact" in captured.err


@pytest.mark.parametrize("status, expected_code", [("current", 0), ("drift", 1)])
def test_canary_command_writes_report_without_touching_resources(
    tmp_path, monkeypatch, status, expected_code
):
    payload = {"status": status, "expected_release": "synthetic-v1"}
    monkeypatch.setattr(canary, "run_canary", lambda **kwargs: payload)
    report = tmp_path / "nested" / "report.json"
    assert (
        canary.main(["--expected-release", "synthetic-v1", "--report", str(report)])
        == expected_code
    )
    assert json.loads(report.read_text()) == payload


def test_canary_command_records_provider_failure(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(canary, "run_canary", fail)
    report = tmp_path / "report.json"
    assert canary.main(["--expected-release", "synthetic-v1", "--report", str(report)]) == 2
    payload = json.loads(report.read_text())
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "synthetic outage"
    assert payload["endpoint"] == canary.OFFICIAL_RXNAV_BASE_URL


@pytest.mark.parametrize(
    "expected_release, expected_ids, status, drift_ids",
    [
        ("synthetic-rxnorm-v1", ["1001"], "current", []),
        ("older-release", ["1001"], "drift", []),
        ("synthetic-rxnorm-v1", ["1002"], "drift", ["one|1"]),
    ],
)
def test_canary_reports_release_and_mapping_drift_using_offline_provider(
    tmp_path, monkeypatch, expected_release, expected_ids, status, drift_ids
):
    from tests.support import ALPHA, make_pipeline

    from biosemalign.enums import ResourceMode
    from biosemalign.storage.jsonl import write_jsonl

    source = tmp_path / "source.csv"
    source.write_text(
        f"prod_ai,primaryid,drug_seq,route,role_cod,indi_pt\n{ALPHA},one,1,oral,PS,\n"
    )
    gold = tmp_path / "gold.jsonl"
    write_jsonl(
        gold, [{"case_id": "one|1", "source_value": ALPHA, "expected_concept_ids": expected_ids}]
    )

    def pipeline_factory(profile, *, settings):
        assert settings.mode is ResourceMode.ONLINE
        assert settings.rxnav_base_url == canary.OFFICIAL_RXNAV_BASE_URL
        return make_pipeline(tmp_path / "synthetic-cache")

    monkeypatch.setattr(canary.BioSemAlign, "from_profile", pipeline_factory)
    payload = canary.run_canary(
        expected_release=expected_release, gold_path=gold, input_path=source
    )
    assert payload["status"] == status
    assert payload["mapping_drift_ids"] == drift_ids
    assert payload["release_changed"] is (expected_release != "synthetic-rxnorm-v1")
    assert payload["evaluation"]["complete"] is True
