"""Mock backends for deterministic tests (Milestone 5 exit criterion 4).

These exist so the pipeline's plumbing can be tested without a model endpoint,
and so the LLM path has something to exercise before ``VLLMBackend`` lands.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from biosemalign.adjudication.backend import GenerationConfig
from biosemalign.enums import DecisionType, RelationshipType
from biosemalign.exceptions import AdjudicationFailure
from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["MockAdjudicator", "MockLLMBackend"]


class MockLLMBackend:
    """Returns pre-supplied structured responses in order.

    Deliberately raises when exhausted rather than repeating the last response:
    a test that makes more calls than it scripted has changed behaviour, and
    should say so.
    """

    backend_name = "mock"

    def __init__(self, responses: Sequence[BaseModel]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, object]] = []

    def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        generation_config: GenerationConfig,
    ) -> BaseModel:
        self.calls.append(
            {
                "messages": messages,
                "output_schema": output_schema.__name__,
                "generation_config": generation_config.model_dump(),
            }
        )
        if self._index >= len(self._responses):
            raise AdjudicationFailure(
                f"MockLLMBackend exhausted after {len(self._responses)} responses"
            )
        response = self._responses[self._index]
        self._index += 1
        if not isinstance(response, output_schema):
            raise AdjudicationFailure(
                f"scripted response is {type(response).__name__}, expected {output_schema.__name__}"
            )
        return response


class MockAdjudicator:
    """Selects the first candidate unconditionally.

    Useful for exercising validation and routing against a known-shaped
    proposal without involving retrieval logic.
    """

    adjudicator_name = "mock"

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(backend=self.adjudicator_name, model_name="mock@0.1.0")

    def adjudicate(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal:
        if not package.candidates:
            return AlignmentProposal(
                sau_id=sau.sau_id,
                decision=DecisionType.NO_VALID_MAPPING,
                relationship_to_source=RelationshipType.UNSUPPORTED_MAPPING,
                rationale="mock adjudicator: no candidates",
            )
        top = package.candidates[0]
        selected = SelectedConcept(
            concept_id=top.candidate_id,
            label=top.preferred_label,
            vocabulary=top.vocabulary,
            concept_class=top.concept_class,
            status=top.status,
            relationship_to_source=RelationshipType.EXACT,
        )
        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=DecisionType.MAPPED,
            selected_concepts=[selected],
            source_level_concept=selected,
            analysis_level_concept=selected,
            relationship_to_source=RelationshipType.EXACT,
            rationale="mock adjudicator: selected the top-ranked candidate",
        )
