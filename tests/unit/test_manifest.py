"""Run fingerprints describe the executed package and durable output evidence."""

import hashlib
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.support import ALPHA, SYNTHETIC_RELEASE, synthetic_settings

import biosemalign.run_manifest as manifests
from biosemalign.enums import ResourceMode
from biosemalign.schemas.manifest import ResearchRunManifest
from biosemalign.schemas.provenance import ResourceProvenance
from biosemalign.util import stable_digest


def resource(**updates):
    values = {
        "resource_name": "RxNorm",
        "provider_name": "rxnav",
        "provider_version": "test",
        "terminology_release": "release",
        "mode": ResourceMode.FROZEN,
        "accessed_at": datetime(2026, 1, 1, tzinfo=UTC),
        "cache_hits": 2,
        "network_calls": 0,
    }
    values.update(updates)
    return ResourceProvenance(**values)


def test_file_and_cache_hashes_ignore_volatile_indexes_and_track_payload_changes(tmp_path):
    assert manifests.cache_snapshot_sha256(tmp_path / "absent") is None
    payload = tmp_path / "entry.json"
    payload.write_bytes(b'{"value":1}')
    assert manifests.file_sha256(payload) == hashlib.sha256(b'{"value":1}').hexdigest()
    initial = manifests.cache_snapshot_sha256(tmp_path)
    (tmp_path / "index.sqlite").write_bytes(b"volatile")
    (tmp_path / ".hidden.json").write_bytes(b"volatile")
    assert manifests.cache_snapshot_sha256(tmp_path) == initial
    payload.write_bytes(b'{"value":2}')
    assert manifests.cache_snapshot_sha256(tmp_path) != initial
    renamed = tmp_path / "renamed.json"
    payload.rename(renamed)
    assert manifests.cache_snapshot_sha256(tmp_path) != initial


def test_code_hash_covers_packaged_config_but_ignores_bytecode(tmp_path, monkeypatch):
    package = tmp_path / "src" / "biosemalign"
    package.mkdir(parents=True)
    module = package / "run_manifest.py"
    module.write_text("# module")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
    monkeypatch.setattr(manifests, "__file__", str(module))
    assert manifests._package_project_root() == tmp_path
    initial = manifests.code_tree_sha256()
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "ignored.pyc").write_bytes(b"ignore")
    (package / "loose.pyo").write_bytes(b"ignore")
    assert manifests.code_tree_sha256() == initial
    config = package / "profile.yaml"
    config.write_text("policy: initial")
    changed = manifests.code_tree_sha256()
    assert changed != initial
    config.write_text("policy: revised")
    assert manifests.code_tree_sha256() != changed
    (tmp_path / "pyproject.toml").unlink()
    assert manifests._package_project_root() is None


def test_settings_hash_excludes_credentials_but_tracks_identifier_key_and_policy(tmp_path):
    plain = synthetic_settings(tmp_path)
    secrets = synthetic_settings(tmp_path, llm_api_key="secret-one", umls_api_key="secret-two")
    assert manifests.effective_settings_sha256(plain) == manifests.effective_settings_sha256(
        secrets
    )
    assert manifests.effective_settings_sha256(plain) != manifests.effective_settings_sha256(
        synthetic_settings(tmp_path, identifier_hmac_key="x" * 32)
    )
    assert manifests.effective_settings_sha256(plain) != manifests.effective_settings_sha256(
        synthetic_settings(tmp_path, output_retention_policy_days=90)
    )


def test_dependency_hash_is_sorted_and_lock_precedence_is_explicit(tmp_path, monkeypatch):
    first = SimpleNamespace(name="a", metadata={"Name": "Alpha"}, version="1")
    second = SimpleNamespace(name="b", metadata={}, version="2")
    monkeypatch.setattr(manifests.importlib.metadata, "distributions", lambda: [second, first])
    assert manifests._dependency_fingerprint() == stable_digest(["Alpha==1", "b==2"])
    assert manifests._dependency_lock_sha256(None) is None
    assert manifests._dependency_lock_sha256(tmp_path) is None
    # Existing Poetry resolution must not silently replace the constrained CI baseline.
    (tmp_path / "poetry.lock").write_text("unrelated")
    assert manifests._dependency_lock_sha256(tmp_path) is None
    for name in ["constraints.txt", "requirements.lock", "uv.lock"]:
        path = tmp_path / name
        path.write_text(name)
        assert manifests._dependency_lock_sha256(tmp_path) == manifests.file_sha256(path)


def test_git_state_reads_package_checkout_and_supports_explicit_commit(tmp_path, monkeypatch):
    assert manifests._git_state(None) == (None, None)
    monkeypatch.setenv("BIOSEMALIGN_CODE_COMMIT", "approved-commit")
    assert manifests._git_state(tmp_path) == ("approved-commit", None)
    monkeypatch.delenv("BIOSEMALIGN_CODE_COMMIT")
    calls = []

    def git(command, **options):
        calls.append((command, options))
        return SimpleNamespace(stdout="abc123\n" if command[1] == "rev-parse" else " M source.py\n")

    monkeypatch.setattr(manifests.subprocess, "run", git)
    assert manifests._git_state(tmp_path) == ("abc123", True)
    assert all(options["cwd"] == tmp_path for _, options in calls)
    monkeypatch.setattr(
        manifests.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="")
    )
    assert manifests._git_state(tmp_path) == (None, False)

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(manifests.subprocess, "run", unavailable)
    assert manifests._git_state(tmp_path) == (None, None)


def test_merge_preserves_inputs_and_accumulates_independent_run_segments():
    first = resource()
    later = resource(
        cache_hits=3, network_calls=1, accessed_at=first.accessed_at + timedelta(days=1)
    )
    other = resource(resource_name="Local", provider_name="local")
    merged = manifests.merge_resource_provenance([first], [later, other])
    assert [entry.resource_name for entry in merged] == ["Local", "RxNorm"]
    assert merged[1].cache_hits == 5 and merged[1].network_calls == 1
    assert merged[1].accessed_at == later.accessed_at
    assert first.cache_hits == 2 and later.cache_hits == 3
    assert manifests.merge_resource_provenance() == []


@pytest.mark.parametrize(
    "updates",
    [
        {"provider_name": "other"},
        {"provider_version": "different"},
        {"terminology_release": "different"},
        {"api_version": "different"},
        {"mode": ResourceMode.ONLINE},
    ],
)
def test_merge_rejects_resource_identity_changes(updates):
    with pytest.raises(ValueError, match="changed across run segments"):
        manifests.merge_resource_provenance([resource()], [resource(**updates)])


def test_run_manifest_fingerprints_outputs_and_freezes_nested_collections(
    tmp_path, monkeypatch, synthetic_pipeline
):
    synthetic_pipeline.align_value(ALPHA)
    output = tmp_path / "output.jsonl"
    output.write_text('{"row":1}\n')
    monkeypatch.setenv("BIOSEMALIGN_CODE_COMMIT", "test-commit")
    started = datetime(2026, 1, 1, tzinfo=UTC)
    manifest = manifests.build_run_manifest(
        pipeline=synthetic_pipeline,
        run_id="run",
        started_at=started,
        input_sha256="input-hash",
        output_paths=[output],
        record_counts={"success": 1, "skipped": 0},
    )
    assert manifest.code_commit == "test-commit"
    assert manifest.working_tree_dirty is None
    assert manifest.output_sha256[str(output)] == manifests.file_sha256(output)
    assert manifest.resources[0].terminology_release == SYNTHETIC_RELEASE
    assert manifest.resources[0].network_calls == 0
    assert manifest.record_counts == {"success": 1, "skipped": 0}
    assert manifest.completed_at >= manifest.started_at
    assert ResearchRunManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(TypeError):
        manifest.record_counts["success"] = 2
    with pytest.raises(ValidationError, match="frozen"):
        manifest.run_id = "changed"
    payload = manifest.model_dump(mode="json")
    payload["resources"] *= 2
    with pytest.raises(ValidationError, match="duplicate resource"):
        ResearchRunManifest.model_validate(payload)
    explicit = manifests.build_run_manifest(
        pipeline=synthetic_pipeline,
        run_id="other-run",
        started_at=started,
        input_sha256="input",
        output_paths=[],
        record_counts={},
        resources=[],
    )
    assert explicit.resources == ()
