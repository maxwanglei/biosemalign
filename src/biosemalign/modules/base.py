"""Composable module protocols (Schema 2, §5.3).

Seven extension points connect an SAU to an artifact: six workflow modules
(``clean``, ``match``, ``expand``, ``link``, ``resolve``, ``select``) plus
``check``. None of them can produce a final decision: whatever workflow is
plugged in, the kernel (``kernel/``) is the only code path that validates,
routes, and mints decisions.

Operators are matched structurally, so a third-party class needs no base
class — only the attributes and method its protocol names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding
from biosemalign.schemas.expansion import ExpansionSet
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

if TYPE_CHECKING:
    from biosemalign.knowledge.router import KnowledgeRouter
    from biosemalign.modules.model import ModelService
    from biosemalign.profiles.schema import TaskProfile
    from biosemalign.settings import Settings

__all__ = [
    "CandidateOrigin",
    "CandidateSet",
    "Check",
    "Cleaner",
    "Expander",
    "Linker",
    "Matcher",
    "OpContext",
    "Operator",
    "PassToNext",
    "Resolver",
    "Selector",
]

CandidateOrigin = Literal["match", "expand", "refinement"]


@dataclass(frozen=True, slots=True)
class PassToNext:
    """A resolver or selector declining to decide, so that the next tier may try.

    The reason is recorded in the decision's resolution trace. An exhausted
    cascade is never dropped: the kernel turns it into an abstention routed to
    human review.
    """

    reason: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Candidates from one matcher, in that matcher's own ranked order."""

    matcher: str
    candidates: list[CandidateConcept] = field(default_factory=list)
    origin: CandidateOrigin = "match"


@dataclass(frozen=True, slots=True)
class OpContext:
    """What an operator factory may draw on when a workflow is assembled."""

    profile: TaskProfile
    settings: Settings | None = None
    knowledge: KnowledgeRouter | None = None
    resolver: Resolver | None = None
    model_service: ModelService | None = None


@runtime_checkable
class Operator(Protocol):
    """Common metadata shared by every registered operator."""

    op_id: str
    version: str
    uses_model: bool


@runtime_checkable
class Cleaner(Protocol):
    """Non-destructive canonicalization and slot extraction: ``(SAU) -> SAU``.

    A cleaner may set ``mention.retrieval_text`` and fill attributes the
    source left empty. The kernel rejects any other change, so source text,
    identity, and asserted context cannot be rewritten.
    """

    op_id: str
    version: str
    uses_model: bool

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit: ...


@runtime_checkable
class Matcher(Protocol):
    """Finds candidate codes for one SAU: ``(SAU) -> CandidateSet``.

    ``requires_network`` is True only for operators that cannot honour FROZEN
    mode; a provider that replays a release-bound cache does not require it.
    """

    op_id: str
    version: str
    uses_model: bool
    requires_network: bool

    def match(self, sau: SemanticAlignmentUnit) -> CandidateSet: ...


@runtime_checkable
class Expander(Protocol):
    """Reaches related codes from anchor concepts and records the path."""

    op_id: str
    version: str
    uses_model: bool

    def expand(self, sau: SemanticAlignmentUnit, anchors: Sequence[CandidateConcept]) -> Any: ...


@runtime_checkable
class Linker(Protocol):
    """Merges candidate sets and builds the bounded evidence a resolver sees."""

    op_id: str
    version: str
    uses_model: bool

    def link(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage: ...


@runtime_checkable
class Resolver(Protocol):
    """Proposes an alignment from the package, or passes to the next tier."""

    op_id: str
    version: str
    uses_model: bool
    requires_network: bool

    def resolve(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext: ...

    def provenance(self) -> ModelProvenance:
        """How the resolver is described in decision provenance."""
        ...


@runtime_checkable
class Selector(Protocol):
    """Proposes which expanded codes belong in a concept set, or passes."""

    op_id: str
    version: str
    uses_model: bool
    requires_network: bool

    def select(self, sau: SemanticAlignmentUnit, expansion: ExpansionSet) -> Any: ...

    def provenance(self) -> ModelProvenance:
        """How the selector is described in provenance."""
        ...


@runtime_checkable
class Check(Protocol):
    """Domain or task check appended after the mandatory kernel checks."""

    op_id: str
    version: str
    uses_model: bool

    def check(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]: ...
