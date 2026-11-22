"""Expansion and slot-refinement contracts (§5.3, §5.4, WP4)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.codes import QualifiedCode, RefinementOutcome
from biosemalign.schemas.knowledge import CandidateConcept

__all__ = [
    "ExpansionMember",
    "ExpansionSet",
    "ExpansionStep",
    "RefinementOutcome",
]


class ExpansionStep(BioSemAlignModel):
    """One relationship traversal (or lexical step) connecting an anchor to an expanded concept."""

    from_code: QualifiedCode
    predicate: str
    direction: Literal["outgoing", "incoming", "lexical"] = "incoming"
    to_code: QualifiedCode


class ExpansionMember(BioSemAlignModel):
    """One candidate concept reached from an anchor concept by an expander."""

    concept: CandidateConcept
    anchor: QualifiedCode
    hops: int = Field(ge=0)
    path: list[ExpansionStep] = Field(default_factory=list)
    via_operator: str
    matched_slots: dict[str, str] = Field(default_factory=dict)
    from_model_suggestion: bool = Field(
        default=False,
        description="True when this member was reached through a model's suggestion (M4, G10).",
    )


class ExpansionSet(BioSemAlignModel):
    """The bounded result of an :class:`~biosemalign.modules.base.Expander`."""

    expander: str
    anchors: list[QualifiedCode] = Field(default_factory=list)
    members: list[ExpansionMember] = Field(default_factory=list)
    refinement: RefinementOutcome | None = None
    truncated: bool = False
