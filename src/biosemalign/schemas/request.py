"""The caller's request, and the mapping task it implies."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from biosemalign.enums import MentionMode, SourceType
from biosemalign.schemas.base import BioSemAlignModel, VersionedModel

__all__ = ["AlignmentRequest", "MappingTask"]


class MappingTask(BioSemAlignModel):
    """What this mapping is *for*.

    Mapping purpose and desired granularity are explicit rather than implied
    (goal 3, §3.1): "metoprolol tartrate 50 MG oral tablet" is the right answer
    for product-level exposure and the wrong answer for ingredient-level signal
    detection, and only the task can distinguish them.
    """

    operation: str = Field(description="Task name, e.g. 'drug_normalization'.")
    mapping_intent: str = Field(description="Analytical purpose, e.g. 'ingredient_level_analysis'.")
    target_terminology: str = Field(description="Target vocabulary, e.g. 'RxNorm'.")
    desired_granularity: str | None = Field(
        default=None, description="Requested level, e.g. 'ingredient', 'precise_ingredient'."
    )


class AlignmentRequest(VersionedModel):
    """A request to align one input against a target terminology (§11.1)."""

    source_type: SourceType
    value: str | dict[str, Any] = Field(
        description="A single value for field input, or a mapping for record input."
    )
    mention_mode: MentionMode = MentionMode.USER_SPECIFIED
    task: str
    mapping_intent: str
    target_terminology: str
    source_system: str | None = None
    desired_granularity: str | None = None
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual fields such as route, drug_role, indication.",
    )

    @field_validator("value")
    @classmethod
    def _reject_blank_value(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(value, str) and not value.strip():
            raise ValueError("value must not be blank")
        if isinstance(value, dict) and not value:
            raise ValueError("value must not be an empty mapping")
        return value

    def to_task(self) -> MappingTask:
        return MappingTask(
            operation=self.task,
            mapping_intent=self.mapping_intent,
            target_terminology=self.target_terminology,
            desired_granularity=self.desired_granularity,
        )
