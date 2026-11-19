"""Workflow combinators: tiers, cascades, and parallel retrieval (v0.2).

A :class:`Cascade` runs its tiers in order. Each tier transforms the SAU,
queries one or more retrievers, fuses their candidates, and asks its proposer
for a proposal; the first proposal is handed to the kernel, and a
:class:`~biosemalign.ops.PassToNext` moves the SAU on to the next tier. Cheap,
high-precision tiers therefore settle the bulk of the input, and only their
residue reaches costlier ones.

Combinators describe *what* runs. Running it, and everything that must happen
after a proposal exists, belongs to the kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from biosemalign.enums import ResourceMode
from biosemalign.exceptions import ConfigurationError
from biosemalign.ops import OpContext, build_op
from biosemalign.ops.base import FuseOp, ProposeOp, RetrieveOp, TransformOp, ValidateOp
from biosemalign.profiles.schema import OperatorConfig, TaskProfile, TierConfig
from biosemalign.schemas.provenance import OperatorProvenance, TierProvenance, WorkflowProvenance
from biosemalign.settings import Settings
from biosemalign.targets.spec import TargetSpec
from biosemalign.util import stable_digest

__all__ = ["Cascade", "ParallelUnion", "Tier", "workflow_from_profile"]

#: Name of the single tier that reproduces the v0.1 pipeline.
DEFAULT_TIER = "default"


@dataclass(frozen=True, slots=True, init=False)
class ParallelUnion:
    """Query several retrievers and hand every candidate set to the fuser.

    Retrievers run in the order given. Concurrency would buy little against
    rate-limited providers and would make provider counters order-dependent.
    """

    retrievers: tuple[RetrieveOp, ...]

    def __init__(self, retrievers: Iterable[RetrieveOp]) -> None:
        object.__setattr__(self, "retrievers", tuple(retrievers))
        if not self.retrievers:
            raise ConfigurationError("ParallelUnion needs at least one retriever")


@dataclass(frozen=True, slots=True)
class Tier:
    """One stage of a cascade."""

    name: str
    retrieval: RetrieveOp | ParallelUnion
    fuser: FuseOp
    proposer: ProposeOp
    transforms: tuple[TransformOp, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        checks: list[tuple[str, object, type]] = [
            ("fuser", self.fuser, FuseOp),
            ("proposer", self.proposer, ProposeOp),
            *(("retriever", op, RetrieveOp) for op in self.retrievers),
            *(("transform", op, TransformOp) for op in self.transforms),
        ]
        for role, op, protocol in checks:
            if not isinstance(op, protocol):
                raise ConfigurationError(
                    f"tier {self.name!r}: {role} {op!r} does not implement {protocol.__name__}"
                )

    @property
    def retrievers(self) -> tuple[RetrieveOp, ...]:
        if isinstance(self.retrieval, ParallelUnion):
            return self.retrieval.retrievers
        return (self.retrieval,)

    def provenance(self) -> TierProvenance:
        return TierProvenance(
            name=self.name,
            operators=(
                *(_operator("transform", op) for op in self.transforms),
                *(_operator("retrieve", op) for op in self.retrievers),
                _operator("fuse", self.fuser),
                _operator("propose", self.proposer),
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class Cascade:
    """Tiers tried in order until one proposes."""

    tiers: tuple[Tier, ...]

    def __init__(self, tiers: Iterable[Tier]) -> None:
        object.__setattr__(self, "tiers", tuple(tiers))
        if not self.tiers:
            raise ConfigurationError("a cascade needs at least one tier")
        names = [tier.name for tier in self.tiers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigurationError(f"cascade tier names must be unique: {duplicates}")

    def verify(self, profile: TaskProfile, settings: Settings) -> None:
        """Fail at startup on configurations that cannot run safely.

        Checked before any record is processed, so a misassembled workflow
        costs a traceback rather than a half-written batch.
        """
        if settings.mode is ResourceMode.FROZEN:
            live: list[str] = []
            for tier in self.tiers:
                operators: list[RetrieveOp | ProposeOp] = [*tier.retrievers, tier.proposer]
                live.extend(f"{tier.name}/{op.op_id}" for op in operators if op.requires_network)
            if live:
                raise ConfigurationError(
                    "FROZEN mode forbids operators that require live network access: "
                    + ", ".join(live)
                )
        declared = [*profile.knowledge.primary_resources, *profile.knowledge.secondary_resources]
        retrievers = [op for tier in self.tiers for op in tier.retrievers]
        if declared and all(hasattr(op, "resources") for op in retrievers):
            queried = {
                str(name).casefold() for op in retrievers for name in getattr(op, "resources", ())
            }
            unqueried = [name for name in declared if name.casefold() not in queried]
            if unqueried:
                raise ConfigurationError(
                    f"profile resources {unqueried} are declared but no workflow retriever "
                    "queries them; their evidence would be silently omitted"
                )
        for tier in self.tiers:
            limit = getattr(tier.proposer, "max_candidates", None)
            if limit is not None and profile.retrieval.maximum_candidates > limit:
                raise ConfigurationError(
                    f"tier {tier.name!r}: proposer {tier.proposer.op_id!r} accepts at most "
                    f"{limit} candidates, but retrieval.maximum_candidates is "
                    f"{profile.retrieval.maximum_candidates}"
                )

    def provenance(
        self, validators: Iterable[ValidateOp] = (), target_spec: TargetSpec | None = None
    ) -> WorkflowProvenance:
        """The workflow's identity: its operators, validators, and target spec.

        The target spec decides what reaches which level and what is lost, so
        a different spec is a different workflow, whatever the operators.
        """
        tiers = tuple(tier.provenance() for tier in self.tiers)
        checks = tuple(_operator("validate", op) for op in validators)
        payload: dict[str, object] = {
            "tiers": [tier.model_dump(mode="json") for tier in tiers],
            "validators": [check.model_dump(mode="json") for check in checks],
        }
        if target_spec is not None:
            payload["target_spec"] = {"label": target_spec.label, "sha256": target_spec.digest}
        return WorkflowProvenance(
            workflow_id=stable_digest(payload, length=16),
            tiers=tiers,
            validators=checks,
            target_spec=target_spec.label if target_spec is not None else None,
            target_spec_sha256=target_spec.digest if target_spec is not None else None,
        )


def workflow_from_profile(context: OpContext) -> Cascade:
    """Assemble the cascade a profile declares, or the single v0.1 tier.

    The default tier queries RxNav, or the local lexicon when the deployment's
    bundle serves the target terminology.
    """
    workflow = context.profile.workflow
    if workflow is not None:
        tiers = workflow.tiers
    else:
        local = context.knowledge is not None and context.knowledge.serves_locally(
            context.profile.task.target_terminology
        )
        retriever = OperatorConfig(op="lexicon" if local else "rxnav")
        tiers = [TierConfig(name=DEFAULT_TIER, retrievers=[retriever])]
    return Cascade(_build_tier(config, context) for config in tiers)


def _build_tier(config: TierConfig, context: OpContext) -> Tier:
    retrievers = [build_op("retrieve", op.op, context, op.options) for op in config.retrievers]
    return Tier(
        name=config.name,
        transforms=tuple(
            build_op("transform", op.op, context, op.options) for op in config.transforms
        ),
        retrieval=retrievers[0] if len(retrievers) == 1 else ParallelUnion(retrievers),
        fuser=build_op("fuse", config.fuser.op, context, config.fuser.options),
        proposer=build_op("propose", config.proposer.op, context, config.proposer.options),
    )


def _operator(kind: str, op: Any) -> OperatorProvenance:
    return OperatorProvenance(
        kind=kind, op_id=op.op_id, version=op.version, options=dict(getattr(op, "options", {}))
    )
