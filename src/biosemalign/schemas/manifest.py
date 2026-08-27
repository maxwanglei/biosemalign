"""Immutable manifest for a reproducible research run."""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field, field_validator, model_validator

from biosemalign.schemas.base import VersionedModel, deep_freeze
from biosemalign.schemas.provenance import (
    ModelProvenance,
    ResourceProvenance,
    require_unique_resources,
)

__all__ = ["ResearchRunManifest"]


class ResearchRunManifest(VersionedModel):
    """Configuration and content fingerprints needed to audit a completed run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: str = "1.0.0"
    run_id: str
    started_at: datetime
    completed_at: datetime
    package_version: str
    code_commit: str | None = None
    working_tree_dirty: bool | None = None
    code_tree_sha256: str
    dependency_fingerprint: str
    dependency_lock_sha256: str | None = None
    profile_name: str
    profile_version: str
    profile_sha256: str
    registry_sha256: str
    settings_sha256: str
    identifier_hmac_key_fingerprint: str | None = None
    resource_endpoints: dict[str, str] = Field(default_factory=dict)
    resources: tuple[ResourceProvenance, ...] = Field(default_factory=tuple)
    cache_snapshot_sha256: str | None = None
    adjudicator_version: str | None = None
    validation_rules_version: str | None = None
    routing_policy_version: str | None = None
    model: ModelProvenance | None = None
    input_sha256: str
    output_sha256: dict[str, str] = Field(default_factory=dict)
    record_counts: dict[str, int] = Field(default_factory=dict)
    cache_retention_days: int | None = None
    output_retention_policy_days: int | None = Field(
        default=None,
        description=(
            "Declared site policy for output retention. BioSemAlign records this value but "
            "does not delete research outputs without an explicit operator workflow."
        ),
    )

    @field_validator("resources")
    @classmethod
    def _resources_are_unique(
        cls, resources: tuple[ResourceProvenance, ...]
    ) -> tuple[ResourceProvenance, ...]:
        require_unique_resources(resources)
        return resources

    @model_validator(mode="after")
    def _freeze_nested_collections(self) -> ResearchRunManifest:
        for name in ("extensions", "resource_endpoints", "output_sha256", "record_counts"):
            object.__setattr__(self, name, deep_freeze(dict(getattr(self, name))))
        return self
