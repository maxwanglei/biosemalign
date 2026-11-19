"""Composable operator protocols (v0.2).

Five stateless, versioned operator kinds fill the stages between an SAU and a
proposal. None of them can produce an ``AlignmentDecision``: whatever workflow
is plugged in, the kernel (``pipelines/kernel.py``) is the only code path that
validates, routes, and mints decisions. That is what keeps unretrieved-ID
rejection, outcome-leakage filtering, and provenance non-bypassable while the
stages themselves stay replaceable.

Operators are matched structurally, so a third-party class needs no base
class — only the attributes and method its protocol names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

if TYPE_CHECKING:
    from biosemalign.adjudication.backend import Adjudicator
    from biosemalign.knowledge.router import KnowledgeRouter
    from biosemalign.profiles.schema import TaskProfile
    from biosemalign.settings import Settings

__all__ = [
    "CandidateSet",
    "CodeSet",
    "Coder",
    "FuseOp",
    "Linker",
    "OpContext",
    "PassToNext",
    "ProposeOp",
    "Resolver",
    "RetrieveOp",
    "TransformOp",
    "ValidateOp",
]


@dataclass(frozen=True, slots=True)
class PassToNext:
    """A proposer declining to decide, so that the next tier may try.

    The reason is recorded in the decision's resolution trace. An exhausted
    cascade is never dropped: the kernel turns it into an abstention routed to
    human review.
    """

    reason: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Candidates from one retriever, in that retriever's own ranked order."""

    retriever: str
    candidates: list[CandidateConcept] = field(default_factory=list)


#: The codes one coder attached to an SAU: candidates with their match
#: evidence (method, score, matched name, asserting source), in the coder's
#: ranked order. The module name for :class:`CandidateSet`.
CodeSet = CandidateSet


@dataclass(frozen=True, slots=True)
class OpContext:
    """What an operator factory may draw on when a workflow is assembled."""

    profile: TaskProfile
    settings: Settings | None = None
    knowledge: KnowledgeRouter | None = None
    adjudicator: Adjudicator | None = None


@runtime_checkable
class TransformOp(Protocol):
    """Non-destructive canonicalization and slot extraction: ``(SAU) -> SAU``.

    A transform may set ``mention.retrieval_text`` and fill attributes the
    source left empty. The kernel rejects any other change, so source text,
    identity, and asserted context cannot be rewritten.
    """

    op_id: str
    version: str

    def transform(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit: ...


@runtime_checkable
class RetrieveOp(Protocol):
    """Queries one terminology index or API: ``(SAU) -> CandidateSet``.

    ``requires_network`` is True only for operators that cannot honour FROZEN
    mode; a provider that replays a release-bound cache does not require it.
    """

    op_id: str
    version: str
    requires_network: bool

    def retrieve(self, sau: SemanticAlignmentUnit) -> CandidateSet: ...


@runtime_checkable
class FuseOp(Protocol):
    """Merges candidate sets into the bounded evidence a proposer may use."""

    op_id: str
    version: str

    def fuse(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage: ...


@runtime_checkable
class ProposeOp(Protocol):
    """Proposes an alignment from the package, or passes to the next tier."""

    op_id: str
    version: str
    requires_network: bool

    def propose(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext: ...

    def provenance(self) -> ModelProvenance:
        """How the proposer is described in decision provenance."""
        ...


@runtime_checkable
class ValidateOp(Protocol):
    """Domain or task checks appended after the mandatory kernel checks."""

    op_id: str
    version: str

    def validate(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]: ...


# -- the three modules (v0.3) ---------------------------------------------------
#
# Coding, linking, and resolution are the module names of the retrieve, fuse,
# and propose stages. An operator may implement either spelling; the registry
# adapts one that implements only the module name. The kernel contract is
# unchanged: whatever the spelling, only the kernel validates and mints.


@runtime_checkable
class Coder(Protocol):
    """Attaches codes to an SAU: ``(SAU) -> CodeSet``. Exact match, BM25, dense, ...

    A coder returns evidence, never a decision: every code it finds, with how
    it found it.
    """

    op_id: str
    version: str
    requires_network: bool

    def code(self, sau: SemanticAlignmentUnit) -> CodeSet: ...


@runtime_checkable
class Linker(Protocol):
    """Links codes toward the target: ``(SAU, CodeSets) -> KnowledgePackage``.

    The package carries the codes, their equivalence groups, and the edges
    that lead to the target granularity. The kernel derives the typed
    ``LinkSet`` from it, after its own leakage filter, so a linker cannot
    assert a link its evidence does not support.
    """

    op_id: str
    version: str

    def link(
        self, sau: SemanticAlignmentUnit, code_sets: Sequence[CodeSet]
    ) -> KnowledgePackage: ...


@runtime_checkable
class Resolver(Protocol):
    """Decides from codes and links, or passes: ``(SAU, package) -> proposal | pass``."""

    op_id: str
    version: str
    requires_network: bool

    def resolve(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext: ...

    def provenance(self) -> ModelProvenance:
        """How the resolver is described in decision provenance."""
        ...
