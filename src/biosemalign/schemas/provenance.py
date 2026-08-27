"""Provenance contracts.

Section 17.3 lists fifteen things every decision must record. Scattering that
list across the pipeline is how records end up half-stamped, so it is collected
here: ``RunContext`` accumulates provenance as a run proceeds and ``freeze()``
emits the immutable ``Provenance`` record that travels with the output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from biosemalign.enums import ResourceMode
from biosemalign.schemas.base import BioSemAlignModel, VersionedModel, deep_freeze
from biosemalign.util import utc_now

__all__ = [
    "ModelProvenance",
    "Provenance",
    "ResourceProvenance",
    "RunContext",
    "require_unique_resources",
]


class ResourceProvenance(BioSemAlignModel):
    """One external knowledge resource as it was used during this run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_name: str = Field(min_length=1, description="Registry name, e.g. 'RxNorm'.")
    provider_name: str = Field(min_length=1, description="Provider implementation, e.g. 'rxnav'.")
    provider_version: str = Field(min_length=1, description="Version of the provider code.")
    terminology_release: str | None = Field(
        default=None,
        description="Vocabulary release identifier, e.g. the RxNorm release date.",
    )
    api_version: str | None = None
    accessed_at: datetime = Field(default_factory=utc_now)
    mode: ResourceMode = ResourceMode.ONLINE
    cache_hits: int = Field(default=0, ge=0)
    network_calls: int = Field(default=0, ge=0)


def require_unique_resources(
    resources: list[ResourceProvenance] | tuple[ResourceProvenance, ...],
) -> None:
    """Reject ambiguous duplicate resource/provider provenance entries."""
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for resource in resources:
        identity = (resource.resource_name.casefold(), resource.provider_name.casefold())
        if identity in seen:
            duplicates.add(identity)
        seen.add(identity)
    if duplicates:
        rendered = ", ".join(f"{resource}/{provider}" for resource, provider in sorted(duplicates))
        raise ValueError(f"duplicate resource provenance entries: {rendered}")


class ModelProvenance(BioSemAlignModel):
    """The adjudicator that produced the proposal.

    Populated for the deterministic adjudicator too — "which rules ran" is as
    much a reproducibility question as "which checkpoint ran".
    """

    backend: str = Field(description="Adjudicator backend, e.g. 'deterministic', 'vllm', 'mock'.")
    model_name: str | None = None
    model_checkpoint: str | None = None
    prompt_version: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _freeze_nested_configuration(self) -> ModelProvenance:
        object.__setattr__(self, "generation_parameters", deep_freeze(self.generation_parameters))
        return self


class Provenance(VersionedModel):
    """Everything needed to explain or reproduce one decision (§17.3)."""

    run_id: str
    package_version: str
    profile_name: str
    profile_version: str
    source_adapter: str | None = None
    source_adapter_version: str | None = None
    identifier_hmac_key_fingerprint: str | None = Field(
        default=None,
        description="Non-secret fingerprint of the key used for SAU and mapping identifiers.",
    )
    retriever_version: str | None = None
    validation_rules_version: str | None = None
    routing_policy_version: str | None = None
    resources: list[ResourceProvenance] = Field(default_factory=list)
    model: ModelProvenance | None = None
    human_review_status: str | None = Field(
        default=None,
        description="Set once the record has been through AL-MedLit review.",
    )
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("resources")
    @classmethod
    def _resources_are_unique(cls, resources: list[ResourceProvenance]) -> list[ResourceProvenance]:
        require_unique_resources(resources)
        return resources


class RunContext:
    """Mutable provenance accumulator for a single pipeline run.

    Not a Pydantic model: it is a runtime collector whose whole job is to be
    written to as the pipeline proceeds, then frozen into a ``Provenance``.
    """

    def __init__(
        self,
        *,
        run_id: str,
        package_version: str,
        profile_name: str,
        profile_version: str,
        identifier_hmac_key_fingerprint: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.package_version = package_version
        self.profile_name = profile_name
        self.profile_version = profile_version
        self.identifier_hmac_key_fingerprint = identifier_hmac_key_fingerprint
        self.source_adapter: str | None = None
        self.source_adapter_version: str | None = None
        self.retriever_version: str | None = None
        self.validation_rules_version: str | None = None
        self.routing_policy_version: str | None = None
        self.model: ModelProvenance | None = None
        self._resources: dict[str, ResourceProvenance] = {}

    def record_resource(
        self,
        *,
        resource_name: str,
        provider_name: str,
        provider_version: str,
        terminology_release: str | None = None,
        api_version: str | None = None,
        mode: ResourceMode = ResourceMode.ONLINE,
        cache_hit: bool = False,
    ) -> None:
        """Note one use of an external resource, merging repeat uses.

        Counts are aggregated per resource rather than appended per call so a
        batch of ten thousand lookups does not produce ten thousand provenance
        entries.
        """
        existing = self._resources.get(resource_name)
        if existing is None:
            existing = ResourceProvenance(
                resource_name=resource_name,
                provider_name=provider_name,
                provider_version=provider_version,
                terminology_release=terminology_release,
                api_version=api_version,
                mode=mode,
            )
            self._resources[resource_name] = existing
        updates: dict[str, Any] = {}
        if terminology_release and existing.terminology_release is None:
            updates["terminology_release"] = terminology_release
        if cache_hit:
            updates["cache_hits"] = existing.cache_hits + 1
        else:
            updates["network_calls"] = existing.network_calls + 1
        existing = existing.model_copy(update=updates)
        self._resources[resource_name] = existing

    def freeze(self) -> Provenance:
        """Emit the immutable provenance record for this run."""
        return Provenance(
            run_id=self.run_id,
            package_version=self.package_version,
            profile_name=self.profile_name,
            profile_version=self.profile_version,
            source_adapter=self.source_adapter,
            source_adapter_version=self.source_adapter_version,
            identifier_hmac_key_fingerprint=self.identifier_hmac_key_fingerprint,
            retriever_version=self.retriever_version,
            validation_rules_version=self.validation_rules_version,
            routing_policy_version=self.routing_policy_version,
            resources=sorted(self._resources.values(), key=lambda r: r.resource_name),
            model=self.model,
        )
