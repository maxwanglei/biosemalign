"""Adjudicator and LLM-backend interfaces (§13.2).

Two protocols, on purpose. ``Adjudicator`` is what the pipeline calls;
``LLMBackend`` is what a model-driven adjudicator calls underneath. Keeping
them separate is what makes the deterministic adjudicator and a future vLLM
adjudicator interchangeable at the pipeline seam without the pipeline knowing
whether a model was involved.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["Adjudicator", "GenerationConfig", "LLMBackend"]


class GenerationConfig(BioSemAlignModel):
    """Inference settings, recorded in provenance for every decision (§7.4)."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = 1024
    seed: int | None = None
    prompt_version: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    """A model that returns structured output conforming to a schema."""

    backend_name: str

    def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        generation_config: GenerationConfig,
    ) -> BaseModel: ...


@runtime_checkable
class Adjudicator(Protocol):
    """Turns an SAU plus its Knowledge Package into a proposed alignment."""

    adjudicator_name: str

    def adjudicate(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal: ...

    def provenance(self) -> ModelProvenance:
        """How this adjudicator should be described in the output record."""
        ...
