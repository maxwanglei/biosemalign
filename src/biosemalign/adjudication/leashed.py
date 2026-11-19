"""The candidate-constrained ("leashed") LLM adjudicator (v0.2).

A model is good at the part rules cannot do — reading "MTX" next to "ectopic
pregnancy" as methotrexate — and bad at the part rules do well: inventing no
identifiers and reporting exactly what a mapping discards. So the division of
labour is strict:

* The model sees a numbered list of retrieved target-terminology candidates
  (at most ``max_candidates``) and answers through a JSON schema whose
  ``candidate`` field admits only ``0..K``. It cannot name an identifier; the
  engine maps the number back to a retrieved candidate.
* The chosen identity is projected to the requested granularity by the
  deterministic rules, which derive the relationship and semantic loss from
  terminology evidence.
* The model also declares a relationship and the attributes it believes are
  lost. Where that declaration disagrees with the evidence, or the model asks
  for review, the proposal requests human review rather than trusting either.

An unusable reply becomes an abstention routed to review, recorded as such.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from biosemalign.adjudication.backend import GenerationConfig, LLMBackend
from biosemalign.adjudication.deterministic import DeterministicAdjudicator
from biosemalign.enums import DecisionType, ReadinessStatus, RelationshipType, SemanticLossKind
from biosemalign.exceptions import AdjudicationFailure
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import stable_digest

__all__ = ["PROMPT_VERSION", "LeashedLLMAdjudicator"]

PROMPT_VERSION = "leashed-1"

#: Candidate list length past which reranking accuracy is known to degrade.
DEFAULT_MAX_CANDIDATES = 15

_SYSTEM = """\
You decide which concept a source mention denotes, choosing from a numbered list of \
candidates retrieved from {target}. Requested granularity: {granularity}.

Rules:
- Choose only by number. Answer 0 when no candidate is defensible \
(abstain_reason "no_valid_candidate") or when the mention cannot be resolved \
without more context (abstain_reason "insufficient_context"). Otherwise use \
abstain_reason "none".
- Never introduce a concept that is not in the list.
- Declare the relationship of the chosen concept to the mention, and list every \
attribute of the mention (strength, dose form, salt, brand, route, ...) that the \
chosen concept does not carry.
- Set needs_review when you are not confident.
{policies}"""

_CONTEXT_FIELDS = (
    "route",
    "dose",
    "unit",
    "drug_role",
    "indication",
    "sentence",
    "paragraph",
    "section_heading",
    "document_type",
    "study_design",
    "table_caption",
)


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


class LeashedLLMAdjudicator:
    """Model-chosen identity, rule-derived projection, disagreement to review."""

    adjudicator_name = "leashed_llm"

    def __init__(
        self,
        profile: TaskProfile,
        backend: LLMBackend,
        *,
        model_name: str,
        model_revision: str | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        seed: int | None = 0,
    ) -> None:
        self.profile = profile
        self.backend = backend
        self.rules = DeterministicAdjudicator(profile)
        self.model_name = model_name
        self.model_revision = model_revision
        self.max_candidates = max_candidates
        self.generation = GenerationConfig(
            temperature=profile.adjudication.temperature,
            seed=seed,
            prompt_version=PROMPT_VERSION,
        )
        self._template_digest = stable_digest(_SYSTEM, length=12)

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            backend=self.backend.backend_name,
            model_name=self.model_name,
            model_checkpoint=self.model_revision,
            prompt_version=f"{PROMPT_VERSION}#{self._template_digest}",
            generation_parameters={
                **self.generation.model_dump(),
                "max_candidates": self.max_candidates,
                "projection": f"drug_normalization_rules via {self.rules.provenance().model_name}",
            },
        )

    def adjudicate(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal:
        target = sau.task.target_terminology.casefold()
        candidates = [c for c in package.candidates if c.vocabulary.casefold() == target]
        candidates = candidates[: self.max_candidates]
        if sau.readiness.status is ReadinessStatus.INSUFFICIENT_CONTEXT or not candidates:
            # Nothing for a model to choose between; the rules abstain.
            return self.rules.adjudicate(sau, package)

        relations = [
            relation.value
            for relation in (self.profile.adjudication.relation_types or list(RelationshipType))
        ]
        messages = self._messages(sau, package, candidates)
        record: dict[str, Any] = {
            "model": self.model_name,
            "prompt_version": PROMPT_VERSION,
            "request_digest": stable_digest(messages, length=16),
            "candidates_shown": len(candidates),
        }
        try:
            reply = self.backend.generate_structured(
                messages=messages,
                output_schema=_choice_model(len(candidates), relations),
                generation_config=self.generation,
            )
        except AdjudicationFailure as exc:
            record["error"] = str(exc)
            return self._abstention(
                sau,
                DecisionType.ABSTAINED,
                rationale="the model reply could not be used; a reviewer must decide",
                review=True,
                record=record,
            )

        choice = reply.model_dump()
        record.update(
            choice=choice["candidate"],
            declared_relationship=choice["relationship"],
            declared_semantic_loss=sorted(set(choice["semantic_loss"])),
            model_requested_review=choice["needs_review"],
            model_rationale=choice["rationale"],
        )
        if choice["candidate"] == 0:
            insufficient = choice["abstain_reason"] == "insufficient_context"
            return self._abstention(
                sau,
                DecisionType.INSUFFICIENT_CONTEXT
                if insufficient
                else DecisionType.NO_VALID_MAPPING,
                rationale=f"the model declined every candidate: {choice['rationale']}",
                review=insufficient or choice["needs_review"],
                record=record,
                missing=["context sufficient to choose a candidate"] if insufficient else [],
            )

        chosen = candidates[choice["candidate"] - 1]
        projected = self.rules.project(sau, chosen)
        evidence_loss = sorted({loss.kind.value for loss in projected.semantic_loss})
        agrees = (
            choice["relationship"] == projected.relationship_to_source.value
            and record["declared_semantic_loss"] == evidence_loss
        )
        record["agrees_with_terminology"] = agrees
        unsupported = list(projected.unsupported_inferences)
        if not agrees:
            unsupported.append(
                f"model declared {choice['relationship']} with loss "
                f"{record['declared_semantic_loss']}; terminology evidence supports "
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
                "extensions": {**projected.extensions, "llm": record},
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
        system = _SYSTEM.format(
            target=sau.task.target_terminology,
            granularity=sau.task.desired_granularity or "the most precise available",
            policies=f"\nProject policies:\n{policies}" if policies else "",
        )
        lines = [f"Mention: {sau.mention.text}"]
        if sau.mention.retrieval_text:
            lines.append(f"Canonicalized: {sau.mention.retrieval_text}")
        context = [
            f"  {name}: {value}"
            for name in _CONTEXT_FIELDS
            if (value := getattr(sau.context, name, None) or getattr(sau.attributes, name, None))
        ]
        context.extend(f"  {key}: {value}" for key, value in sorted(sau.context.extensions.items()))
        if context:
            lines.append("Context:")
            lines.extend(context)
        lines.append("Candidates:")
        for number, candidate in enumerate(candidates, start=1):
            details = [candidate.concept_class or "concept", candidate.status.value]
            entry = (
                f"{number}. {candidate.preferred_label} "
                f"[{candidate.vocabulary} {candidate.candidate_id}; {', '.join(details)}]"
            )
            if candidate.synonyms:
                entry += f"; also: {'; '.join(candidate.synonyms[:3])}"
            related = [
                f"{s.predicate}: {s.object_label or s.object_id}"
                for s in candidate.relationships[:5]
            ]
            if related:
                entry += f"; {'; '.join(related)}"
            lines.append(entry)
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
        record: dict[str, Any],
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
            preserved_attributes=sau.attributes.model_dump(
                exclude_none=True, exclude={"extensions"}
            ),
            missing_information=list(missing or []),
            rationale=rationale,
            requested_human_review=review,
            extensions={"llm": record},
        )
