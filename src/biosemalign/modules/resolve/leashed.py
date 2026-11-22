"""The candidate-constrained ("leashed") LLM resolver (`llm`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from biosemalign.enums import DecisionType, ReadinessStatus, RelationshipType, SemanticLossKind
from biosemalign.exceptions import AdjudicationFailure, ConfigurationError
from biosemalign.modules.model import (
    ModelService,
    PromptTemplate,
    load_prompt,
    render_declared_context,
    render_numbered_candidates,
)
from biosemalign.modules.resolve.backend import GenerationConfig, LLMBackend
from biosemalign.modules.resolve.deterministic import DeterministicResolver
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit, canonical_preserved_attributes

__all__ = ["DEFAULT_MAX_CANDIDATES", "PROMPT_VERSION", "LeashedLLMResolver"]

PROMPT_VERSION = "leashed-1"

#: Candidate list length past which reranking accuracy is known to degrade.
DEFAULT_MAX_CANDIDATES = 15


class _ChoiceBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _choice_model(size: int, relations: list[str]) -> type[BaseModel]:
    """The reply schema: a candidate number in 0..size, nothing else identifying."""
    numbers: Any = Literal.__getitem__(tuple(range(size + 1)))
    relation: Any = Literal.__getitem__(tuple(relations))
    loss: Any = Literal.__getitem__(tuple(kind.value for kind in SemanticLossKind))
    reasons: Any = Literal["none", "no_valid_candidate", "insufficient_context"]
    model: type[BaseModel] = create_model(
        "LeashedChoice",
        __base__=_ChoiceBase,
        candidate=(numbers, Field(description="Number of the chosen candidate, or 0.")),
        abstain_reason=(reasons, ...),
        relationship=(relation, ...),
        semantic_loss=(list[loss], ...),
        needs_review=(bool, ...),
        rationale=(str, Field(max_length=600)),
    )
    return model


class LeashedLLMResolver:
    """Model-chosen identity, rule-derived projection, disagreement to review."""

    op_id = "llm"
    version = "0.3.0"
    uses_model = True
    requires_network = False

    def __init__(
        self,
        profile: TaskProfile,
        backend: ModelService | LLMBackend,
        *,
        model_name: str | None = None,
        model_revision: str | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        temperature: float = 0.0,
        prompt_version: str | None = None,
        seed: int | None = 0,
        examples_path: str | Path | None = None,
    ) -> None:
        self.profile = profile
        if isinstance(backend, ModelService):
            self.model_service = backend
            self.backend: LLMBackend = backend.backend
            self.model_name = model_name or backend.model_name
            self.model_revision = (
                model_revision if model_revision is not None else backend.model_revision
            )
        else:
            if not model_name:
                raise ConfigurationError(
                    "model_name is required when constructing LeashedLLMResolver from a backend"
                )
            self.model_service = ModelService(
                backend=backend,
                model_name=model_name,
                model_revision=model_revision,
            )
            self.backend = backend
            self.model_name = model_name
            self.model_revision = model_revision
        self.rules = DeterministicResolver(profile)
        self.max_candidates = max_candidates
        effective_prompt = prompt_version or PROMPT_VERSION
        self.generation = GenerationConfig(
            temperature=temperature,
            seed=seed,
            prompt_version=effective_prompt,
        )
        self.prompt: PromptTemplate = load_prompt(
            "choose_candidate", domain=profile.domain, examples_path=examples_path
        )
        self._template_digest = self.prompt.digest

    @property
    def options(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.generation.temperature,
            "max_candidates": self.max_candidates,
            "prompt_version": self.generation.prompt_version,
            "prompt_digest": self.prompt.digest,
        }

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            backend=self.model_service.backend_name,
            model_name=self.model_name,
            model_checkpoint=self.model_revision,
            prompt_version=f"{self.generation.prompt_version}#{self._template_digest}",
            generation_parameters={
                **self.generation.model_dump(),
                "max_candidates": self.max_candidates,
                "prompt_id": self.prompt.prompt_id,
                "projection": f"rules via {self.rules.provenance().model_name}",
            },
        )

    def resolve(self, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> AlignmentProposal:
        target = sau.task.target_terminology.casefold()
        candidates = [c for c in package.candidates if c.vocabulary.casefold() == target]
        candidates = candidates[: self.max_candidates]
        if sau.readiness.status is ReadinessStatus.INSUFFICIENT_CONTEXT or not candidates:
            # Nothing for a model to choose between; the rules abstain.
            return self.rules.resolve(sau, package)

        relations = [
            relation.value
            for relation in (self.profile.task.relation_types or list(RelationshipType))
        ]
        messages = self._messages(sau, package, candidates)
        prompt_id = f"{self.generation.prompt_version}:{self.prompt.prompt_id}"
        try:
            call_result = self.model_service.call(
                operator_id=self.op_id,
                operator_kind="resolve",
                task="choose_candidate",
                messages=messages,
                output_schema=_choice_model(len(candidates), relations),
                generation_config=self.generation,
                prompt_id=prompt_id,
                prompt_digest=self.prompt.digest,
            )
        except AdjudicationFailure:
            self.model_service.update_last_record(candidates_shown=len(candidates))
            return self._abstention(
                sau,
                DecisionType.ABSTAINED,
                rationale="the model reply could not be used; a reviewer must decide",
                review=True,
            )

        choice = call_result.reply.model_dump()
        declared_loss = tuple(sorted(set(choice["semantic_loss"])))
        if choice["candidate"] == 0:
            insufficient = choice["abstain_reason"] == "insufficient_context"
            self.model_service.update_record(
                call_result.record_index,
                candidates_shown=len(candidates),
                choice=0,
                declared_relationship=choice["relationship"],
                declared_semantic_loss=declared_loss,
                model_requested_review=choice["needs_review"],
                model_rationale=choice["rationale"],
            )
            # M3: declining every candidate routes to human review unless profile opts in.
            review = (
                insufficient
                or choice["needs_review"]
                or not self.profile.routing.allow_model_auto_abstain
            )
            return self._abstention(
                sau,
                DecisionType.INSUFFICIENT_CONTEXT
                if insufficient
                else DecisionType.NO_VALID_MAPPING,
                rationale=f"the model declined every candidate: {choice['rationale']}",
                review=review,
                missing=["context sufficient to choose a candidate"] if insufficient else [],
            )

        chosen = candidates[choice["candidate"] - 1]
        anchor: CandidateConcept | None = None
        if (
            package.refinement is not None
            and package.refinement.status == "unique"
            and package.refinement.selected is not None
            and package.refinement.anchor is not None
            and (chosen.vocabulary.casefold(), chosen.candidate_id)
            == (
                package.refinement.selected.vocabulary.casefold(),
                package.refinement.selected.concept_code,
            )
        ):
            anchor = package.candidate_by_qualified_id(
                package.refinement.anchor.vocabulary,
                package.refinement.anchor.concept_code,
            )
        projected = self.rules.project(
            sau,
            chosen,
            links=package.links,
            refinement=package.refinement,
            anchor=anchor,
        )
        evidence_loss = sorted({loss.kind.value for loss in projected.semantic_loss})
        agrees = (
            choice["relationship"] == projected.relationship_to_source.value
            and list(declared_loss) == evidence_loss
        )
        self.model_service.update_record(
            call_result.record_index,
            candidates_shown=len(candidates),
            choice=choice["candidate"],
            declared_relationship=choice["relationship"],
            declared_semantic_loss=declared_loss,
            agrees_with_terminology=agrees,
            model_requested_review=choice["needs_review"],
            model_rationale=choice["rationale"],
        )
        unsupported = list(projected.unsupported_inferences)
        if not agrees:
            unsupported.append(
                f"model declared {choice['relationship']} with loss "
                f"{list(declared_loss)}; terminology evidence supports "
                f"{projected.relationship_to_source.value} with loss {evidence_loss}"
            )
        return projected.model_copy(
            update={
                "rationale": (
                    f"model selected candidate {choice['candidate']} "
                    f"({chosen.candidate_id}); {projected.rationale}"
                ),
                "unsupported_inferences": unsupported,
                "requested_human_review": (
                    projected.requested_human_review or choice["needs_review"] or not agrees
                ),
            }
        )

    # -- helpers ---------------------------------------------------------

    def _messages(
        self,
        sau: SemanticAlignmentUnit,
        package: KnowledgePackage,
        candidates: list[CandidateConcept],
    ) -> list[dict[str, str]]:
        policies = "\n".join(f"- {policy.rule}" for policy in package.coding_policies)
        system = self.prompt.text.format(
            target=sau.task.target_terminology,
            granularity=sau.task.desired_granularity or "the most precise available",
            policies=f"\nProject policies:\n{policies}" if policies else "",
        )
        if self.prompt.examples_text:
            system = f"{system}\n\nWorked examples:\n{self.prompt.examples_text}"
        lines = [f"Mention: {sau.mention.text}"]
        if sau.mention.retrieval_text:
            lines.append(f"Canonicalized: {sau.mention.retrieval_text}")
        context = render_declared_context(sau, "resolve", self.profile)
        if context:
            lines.append("Context:")
            lines.extend(context)
        lines.append("Candidates:")
        lines.extend(render_numbered_candidates(candidates))
        lines.append("Reply with the JSON object only.")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def _abstention(
        self,
        sau: SemanticAlignmentUnit,
        decision: DecisionType,
        *,
        rationale: str,
        review: bool,
        missing: list[str] | None = None,
    ) -> AlignmentProposal:
        relationship = (
            RelationshipType.INSUFFICIENT_CONTEXT
            if decision is DecisionType.INSUFFICIENT_CONTEXT
            and self.profile.permits_relation(RelationshipType.INSUFFICIENT_CONTEXT)
            else RelationshipType.UNSUPPORTED_MAPPING
        )
        if not self.profile.permits_relation(relationship):
            raise AdjudicationFailure(
                f"profile {self.profile.profile_name} cannot represent abstention relation "
                f"{relationship.value}"
            )
        return AlignmentProposal(
            sau_id=sau.sau_id,
            decision=decision,
            relationship_to_source=relationship,
            preserved_attributes=canonical_preserved_attributes(sau.attributes),
            missing_information=list(missing or []),
            rationale=rationale,
            requested_human_review=review,
        )
