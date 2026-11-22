"""Workflow cascade combinators run by the kernel (§5.3, §5.4)."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from biosemalign.enums import DecisionType, ResourceMode
from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import Check, Cleaner, Expander, Linker, Matcher, PassToNext, Resolver
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.provenance import OperatorProvenance, TierProvenance, WorkflowProvenance
from biosemalign.settings import Settings
from biosemalign.targets.spec import TargetSpec
from biosemalign.util import stable_digest

__all__ = ["Cascade", "ParallelUnion", "Tier", "operator_provenance"]


@dataclass(frozen=True, slots=True, init=False)
class ParallelUnion:
    """Query several matchers and hand every candidate set to the linker.

    Matchers run in the order given. Concurrency would buy little against
    rate-limited providers and would make provider counters order-dependent.
    """

    matchers: tuple[Matcher, ...]

    def __init__(self, matchers: Iterable[Matcher]) -> None:
        object.__setattr__(self, "matchers", tuple(matchers))
        if not self.matchers:
            raise ConfigurationError("ParallelUnion needs at least one matcher")


@dataclass(frozen=True, slots=True)
class Tier:
    """One tier of a cascade, including tier-level escalation settings."""

    name: str
    match: Matcher | ParallelUnion
    linker: Linker
    resolver: Resolver
    cleaners: tuple[Cleaner, ...] = ()
    expander: Expander | None = None
    pass_on: frozenset[DecisionType] | Collection[DecisionType] = field(default_factory=frozenset)
    pass_on_review_request: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cleaners", tuple(self.cleaners))
        object.__setattr__(self, "pass_on", frozenset(self.pass_on))
        checks: list[tuple[str, object, type]] = [
            ("linker", self.linker, Linker),
            ("resolver", self.resolver, Resolver),
            *(("matcher", op, Matcher) for op in self.matchers),
            *(("cleaner", op, Cleaner) for op in self.cleaners),
        ]
        if self.expander is not None:
            checks.append(("expander", self.expander, Expander))
        for role, op, protocol in checks:
            if not isinstance(op, protocol):
                raise ConfigurationError(
                    f"tier {self.name!r}: {role} {op!r} does not implement {protocol.__name__}"
                )

    @property
    def matchers(self) -> tuple[Matcher, ...]:
        if isinstance(self.match, ParallelUnion):
            return self.match.matchers
        return (self.match,)

    def apply_escalation(
        self, result: AlignmentProposal | PassToNext
    ) -> AlignmentProposal | PassToNext:
        """Apply this tier's ``pass_on`` and ``pass_on_review_request`` policy."""
        if not isinstance(result, AlignmentProposal):
            return result
        if result.decision in self.pass_on:
            return PassToNext(reason=f"decision:{result.decision.value}", detail=result.rationale)
        if self.pass_on_review_request and result.requested_human_review:
            return PassToNext(reason="review_requested", detail=result.rationale)
        return result

    def provenance(self) -> TierProvenance:
        ops: list[OperatorProvenance] = [
            *(_operator("clean", op) for op in self.cleaners),
            *(_operator("match", op) for op in self.matchers),
        ]
        if self.expander is not None:
            ops.append(_operator("expand", self.expander))
        ops.extend([_operator("link", self.linker), _operator("resolve", self.resolver)])
        return TierProvenance(name=self.name, operators=tuple(ops))


@dataclass(frozen=True, slots=True, init=False)
class Cascade:
    """Tiers tried in order until one resolves."""

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
        """Fail at startup on configurations that cannot run safely."""
        if settings.mode is ResourceMode.FROZEN:
            live: list[str] = []
            for tier in self.tiers:
                operators: list[Matcher | Resolver] = [*tier.matchers, tier.resolver]
                live.extend(f"{tier.name}/{op.op_id}" for op in operators if op.requires_network)
            if live:
                raise ConfigurationError(
                    "FROZEN mode forbids operators that require live network access: "
                    + ", ".join(live)
                )
        declared = list(profile.knowledge.resources)
        matchers = [op for tier in self.tiers for op in tier.matchers]
        if declared and all(hasattr(op, "resources") for op in matchers):
            queried = {
                str(name).casefold() for op in matchers for name in getattr(op, "resources", ())
            }
            unqueried = [name for name in declared if name.casefold() not in queried]
            if unqueried:
                raise ConfigurationError(
                    f"profile resources {unqueried} are declared but no workflow matcher "
                    "queries them; their evidence would be silently omitted"
                )
        for tier in self.tiers:
            limit = getattr(tier.resolver, "max_candidates", None)
            if limit is not None and profile.limits.candidates > limit:
                raise ConfigurationError(
                    f"tier {tier.name!r}: resolver {tier.resolver.op_id!r} accepts at most "
                    f"{limit} candidates, but limits.candidates is "
                    f"{profile.limits.candidates}"
                )

    def provenance(
        self, checks: Iterable[Check] = (), target_spec: TargetSpec | None = None
    ) -> WorkflowProvenance:
        """The workflow's identity: its operators, checks, and target spec."""
        tiers = tuple(tier.provenance() for tier in self.tiers)
        check_ops = tuple(_operator("check", op) for op in checks)
        payload: dict[str, object] = {
            "tiers": [tier.model_dump(mode="json") for tier in tiers],
            "validators": [check.model_dump(mode="json") for check in check_ops],
        }
        escalation = [
            {
                "tier": tier.name,
                "pass_on": sorted(decision.value for decision in tier.pass_on),
                "pass_on_review_request": tier.pass_on_review_request,
            }
            for tier in self.tiers
            if tier.pass_on or tier.pass_on_review_request
        ]
        if escalation:
            payload["escalation"] = escalation
        if target_spec is not None:
            payload["target_spec"] = {"label": target_spec.label, "sha256": target_spec.digest}
        return WorkflowProvenance(
            workflow_id=stable_digest(payload, length=16),
            tiers=tiers,
            validators=check_ops,
            target_spec=target_spec.label if target_spec is not None else None,
            target_spec_sha256=target_spec.digest if target_spec is not None else None,
        )


def operator_provenance(kind: str, op: Any) -> OperatorProvenance:
    """Build immutable :class:`OperatorProvenance` for one workflow operator."""
    return OperatorProvenance(
        kind=kind,
        op_id=op.op_id,
        version=op.version,
        options=dict(getattr(op, "options", {})),
        uses_model=bool(op.uses_model),
    )


_operator = operator_provenance
