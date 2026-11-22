"""LLM backend protocol and generation parameters (§13.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from biosemalign.schemas.base import BioSemAlignModel

__all__ = ["GenerationConfig", "LLMBackend"]


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
