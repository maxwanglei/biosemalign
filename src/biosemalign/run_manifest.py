"""Build content-addressed research-run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from biosemalign._version import __version__
from biosemalign.resources.registry import get_registry
from biosemalign.schemas.manifest import ResearchRunManifest
from biosemalign.schemas.provenance import ModelProvenance, ResourceProvenance
from biosemalign.util import stable_digest, utc_now

if TYPE_CHECKING:
    from biosemalign.pipelines.base import BioSemAlign
    from biosemalign.settings import Settings

__all__ = [
    "build_run_manifest",
    "cache_snapshot_sha256",
    "code_tree_sha256",
    "effective_settings_sha256",
    "file_sha256",
    "merge_resource_provenance",
]


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cache_snapshot_sha256(path: str | Path) -> str | None:
    """Hash immutable cache entries without volatile locks and index databases."""
    root = Path(path)
    if not root.is_dir():
        return None
    entries = sorted(
        candidate
        for candidate in root.rglob("*.json")
        if candidate.is_file() and not candidate.name.startswith(".")
    )
    digest = hashlib.sha256()
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(entry).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def code_tree_sha256() -> str:
    """Fingerprint the installed BioSemAlign implementation and packaged configuration."""
    root = Path(__file__).resolve().parent
    entries = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix not in {".pyc", ".pyo"}
    )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(entry).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def effective_settings_sha256(settings: Settings) -> str:
    """Hash effective non-secret settings plus the identifier-key fingerprint."""
    payload = settings.model_dump(
        mode="json", exclude={"identifier_hmac_key", "llm_api_key", "umls_api_key"}
    )
    payload["identifier_hmac_key_fingerprint"] = settings.identifier_hmac_key_fingerprint
    return stable_digest(payload)


def _dependency_fingerprint() -> str:
    installed = sorted(
        f"{distribution.metadata.get('Name', distribution.name)}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    return stable_digest(installed)


def _package_project_root() -> Path | None:
    """Return the source checkout that owns this package, when one is present.

    A command may be launched from an unrelated study repository.  Provenance
    must describe the installed BioSemAlign code, never the caller's current
    working directory.
    """
    module = Path(__file__).resolve()
    for parent in module.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "biosemalign").is_dir():
            return parent
    return None


def _git_state(workspace: Path | None) -> tuple[str | None, bool | None]:
    explicit = os.getenv("BIOSEMALIGN_CODE_COMMIT")
    if explicit:
        return explicit, None
    if workspace is None:
        return None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _dependency_lock_sha256(workspace: Path | None) -> str | None:
    if workspace is None:
        return None
    for name in ("uv.lock", "requirements.lock", "constraints.txt"):
        path = workspace / name
        if path.is_file():
            return file_sha256(path)
    return None


def merge_resource_provenance(
    *snapshots: list[ResourceProvenance],
) -> list[ResourceProvenance]:
    """Combine independent run segments without double-counting snapshots.

    Each input list is one cumulative counter window (for example, the durable
    pre-resume checkpoint and the counters collected after resume).  Mixing a
    terminology release or provider implementation within one resource fails
    closed because that would no longer be a reproducible frozen run.
    """
    merged: dict[str, ResourceProvenance] = {}
    identity_fields = (
        "provider_name",
        "provider_version",
        "terminology_release",
        "api_version",
        "mode",
    )
    for snapshot in snapshots:
        for resource in snapshot:
            existing = merged.get(resource.resource_name)
            if existing is None:
                merged[resource.resource_name] = resource.model_copy(deep=True)
                continue
            mismatches = [
                field
                for field in identity_fields
                if getattr(existing, field) != getattr(resource, field)
            ]
            if mismatches:
                raise ValueError(
                    f"resource {resource.resource_name!r} changed across run segments: "
                    + ", ".join(mismatches)
                )
            merged[resource.resource_name] = existing.model_copy(
                update={
                    "cache_hits": existing.cache_hits + resource.cache_hits,
                    "network_calls": existing.network_calls + resource.network_calls,
                    "accessed_at": max(existing.accessed_at, resource.accessed_at),
                }
            )
    return sorted(merged.values(), key=lambda item: item.resource_name)


def build_run_manifest(
    *,
    pipeline: BioSemAlign,
    run_id: str,
    started_at: datetime,
    input_sha256: str,
    output_paths: list[str | Path],
    record_counts: dict[str, int],
    resources: list[ResourceProvenance] | None = None,
) -> ResearchRunManifest:
    """Snapshot the effective run configuration and finalized outputs."""
    workspace = _package_project_root()
    commit, dirty = _git_state(workspace)
    profile_payload = pipeline.profile.model_dump(mode="json")
    registry_payload = get_registry().model_dump(mode="json")
    resource_snapshot = pipeline.knowledge.resource_provenance() if resources is None else resources
    output_hashes = {str(Path(path)): file_sha256(path) for path in output_paths}

    model: ModelProvenance | None = pipeline.adjudicator.provenance()
    return ResearchRunManifest(
        run_id=run_id,
        started_at=started_at,
        completed_at=utc_now(),
        package_version=__version__,
        code_commit=commit,
        working_tree_dirty=dirty,
        code_tree_sha256=code_tree_sha256(),
        dependency_fingerprint=_dependency_fingerprint(),
        dependency_lock_sha256=_dependency_lock_sha256(workspace),
        profile_name=pipeline.profile.profile_name,
        profile_version=pipeline.profile.profile_version,
        profile_sha256=stable_digest(profile_payload),
        registry_sha256=stable_digest(registry_payload),
        settings_sha256=effective_settings_sha256(pipeline.settings),
        identifier_hmac_key_fingerprint=(pipeline.settings.identifier_hmac_key_fingerprint),
        resource_endpoints={"RxNorm": pipeline.settings.rxnav_base_url},
        resources=resource_snapshot,
        cache_snapshot_sha256=cache_snapshot_sha256(pipeline.settings.cache_dir),
        adjudicator_version=getattr(pipeline.adjudicator, "version", None),
        validation_rules_version=pipeline.validator.version,
        routing_policy_version=pipeline.decision_router.version,
        model=model,
        input_sha256=input_sha256,
        output_sha256=output_hashes,
        record_counts=record_counts,
        cache_retention_days=getattr(pipeline.settings, "cache_retention_days", None),
        output_retention_policy_days=getattr(
            pipeline.settings, "output_retention_policy_days", None
        ),
    )
