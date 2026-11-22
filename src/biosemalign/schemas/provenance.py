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

from biosemalign.enums import ConcordanceState, ResourceMode, TierOutcome
from biosemalign.schemas.base import BioSemAlignModel, VersionedModel, deep_freeze
from biosemalign.schemas.codes import EquivalenceGroup
from biosemalign.util import utc_now

__all__ = [
    "MemoProvenance",
    "ModelCallRecord",
    "ModelProvenance",
    "OperatorProvenance",
    "Provenance",
    "ResolutionTrace",
    "ResourceProvenance",
    "RunContext",
    "TierAttempt",
    "TierProvenance",
    "WorkflowProvenance",
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
    content_sha256: str | None = Field(
        default=None,
        description="Digest of the local snapshot served, when the resource is a file bundle.",
    )


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


class OperatorProvenance(BioSemAlignModel):
    """One configured operator in a workflow (v0.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(
        description="Operator kind: clean, match, expand, link, resolve, select, check."
    )
    op_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)
    uses_model: bool = Field(
        default=False,
        description="True when this operator calls a language model (§5.10).",
    )

    @model_validator(mode="after")
    def _freeze_options(self) -> OperatorProvenance:
        object.__setattr__(self, "options", deep_freeze(self.options))
        return self


class ModelCallRecord(BioSemAlignModel):
    """Typed provenance record for one language-model call through ModelService (§5.10, G9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operator_id: str = Field(min_length=1)
    operator_kind: str = Field(min_length=1)
    task: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str | None = None
    prompt_id: str = Field(min_length=1)
    prompt_digest: str = Field(min_length=1)
    request_digest: str = Field(min_length=1)
    from_cache: bool = False
    tier: str | None = None
    candidates_shown: int | None = None
    choice: int | None = None
    declared_relationship: str | None = None
    declared_semantic_loss: tuple[str, ...] = ()
    agrees_with_terminology: bool | None = None
    model_requested_review: bool | None = None
    model_rationale: str | None = None
    error: str | None = None


class TierProvenance(BioSemAlignModel):
    """The operators one cascade tier runs, in execution order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    operators: tuple[OperatorProvenance, ...] = ()


class WorkflowProvenance(BioSemAlignModel):
    """The complete operator configuration a decision was produced under.

    ``workflow_id`` is a digest of everything else here, so two decisions can
    be compared for "same workflow" without diffing operator lists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str = Field(min_length=1)
    tiers: tuple[TierProvenance, ...] = Field(min_length=1)
    validators: tuple[OperatorProvenance, ...] = ()
    target_spec: str | None = Field(
        default=None, description="The target spec decisions followed, e.g. 'rxnorm@1.0' (v0.3)."
    )
    target_spec_sha256: str | None = None


class TierAttempt(BioSemAlignModel):
    """What one tier did with an SAU, and why it passed if it did.

    A tier that assesses concordance also records every exact match it saw —
    each vocabulary's code, grouped by equivalence, linked to the target or
    not. Coding is never lost when a tier passes: the final artifact carries
    only the last tier's package, but every tier's matches stay here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: str = Field(min_length=1)
    outcome: TierOutcome
    reason: str | None = None
    detail: str | None = None
    concordance_state: ConcordanceState | None = None
    exact_matches: tuple[EquivalenceGroup, ...] = Field(
        default=(),
        description="Every exact match of this tier's package, grouped by equivalence.",
    )


class ResolutionTrace(BioSemAlignModel):
    """The path one SAU took through a workflow cascade.

    Recorded per decision rather than per run: tier attribution is what makes
    per-tier coverage and precision measurable, and a pass-through reason is
    the only explanation of why a later, costlier tier was needed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: tuple[TierAttempt, ...] = Field(min_length=1)
    resolved_by_tier: str | None = Field(
        default=None,
        description="Tier whose proposal was validated; None when every tier passed.",
    )
    withheld_context: tuple[str, ...] = Field(
        default=(),
        description="Context fields hidden from proposers by the profile's blinding policy.",
    )

    @model_validator(mode="after")
    def _attempts_are_coherent(self) -> ResolutionTrace:
        *earlier, last = self.attempts
        if any(attempt.outcome is not TierOutcome.PASSED for attempt in earlier):
            raise ValueError("only the final tier attempt may resolve an SAU")
        if last.outcome is TierOutcome.RESOLVED:
            if self.resolved_by_tier != last.tier:
                raise ValueError("resolved_by_tier must name the tier that resolved the SAU")
        elif self.resolved_by_tier is not None:
            raise ValueError("an exhausted cascade cannot name a resolving tier")
        return self


class MemoProvenance(BioSemAlignModel):
    """Marks a decision whose proposal and evidence were reused from the memo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    original_run_id: str = Field(description="Run that resolved the input first.")
    stored_at: datetime


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
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    workflow: WorkflowProvenance | None = None
    resolution: ResolutionTrace | None = None
    memo: MemoProvenance | None = None
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
        self.workflow: WorkflowProvenance | None = None
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
            workflow=self.workflow,
        )
