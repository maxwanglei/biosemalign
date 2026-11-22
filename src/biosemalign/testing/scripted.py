"""Test-only scripted backend and model-backed operators (§5.10, WP8).

These exist solely to verify kernel model-readiness guarantees (M1–M9, G1–G10)
across every workflow module without shipping production model operators or
making network calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from biosemalign.enums import Authority, RelationshipType, RetrievalChannel, Severity
from biosemalign.exceptions import AdjudicationFailure
from biosemalign.knowledge.providers.lexicon import LexiconProvider
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules.base import CandidateSet, Linker
from biosemalign.modules.model import ModelService, render_declared_context
from biosemalign.modules.resolve.backend import GenerationConfig
from biosemalign.modules.select.hierarchy import HierarchySelector
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet, ExpansionStep
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage, KnowledgeStatement
from biosemalign.schemas.phenotyping import ConceptSetMember, ConceptSetProposal, ConceptSetScope
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import stable_digest

__all__ = [
    "JudgeMembersReply",
    "MemberJudgment",
    "ScriptedLLMBackend",
    "ScriptedModelCheck",
    "ScriptedModelCleaner",
    "ScriptedModelExpander",
    "ScriptedModelLinker",
    "ScriptedModelMatcher",
    "ScriptedModelSelector",
    "scripted_model_service",
]

ScriptedResponseSource = (
    dict[str, Any]
    | BaseModel
    | Exception
    | Sequence[dict[str, Any] | BaseModel | Exception]
    | Callable[[list[dict[str, str]], type[BaseModel], GenerationConfig], Any]
)


class _StrictReply(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _QuerySuggestionsReply(_StrictReply):
    queries: list[str] = Field(default_factory=list, max_length=10)


class _LinkSuggestionItem(_StrictReply):
    subject_code: str
    predicate: str
    object_code: str
    object_vocabulary: str | None = None
    object_label: str | None = None


class _LinkSuggestionsReply(_StrictReply):
    assertions: list[_LinkSuggestionItem] = Field(default_factory=list, max_length=10)


class _CheckFindingItem(_StrictReply):
    code: str
    severity: Literal["warning", "info", "error"] = "warning"
    message: str
    concept_id: str | None = None


class _CheckReply(_StrictReply):
    findings: list[_CheckFindingItem] = Field(default_factory=list, max_length=10)


class MemberJudgment(_StrictReply):
    """Single integer-indexed member judgment; cannot carry terminology codes (M1)."""

    member: int = Field(ge=1)
    included: bool = True
    relationship: RelationshipType
    scope: ConceptSetScope
    rationale: str = Field(min_length=1)


class JudgeMembersReply(_StrictReply):
    """Paged selector reply containing integer-indexed judgments only (M1)."""

    judgments: list[MemberJudgment] = Field(default_factory=list)


class ScriptedLLMBackend:
    """Offline :class:`~biosemalign.modules.resolve.backend.LLMBackend` driven by test scripts."""

    backend_name = "scripted"

    def __init__(self, responses: ScriptedResponseSource) -> None:
        self._source = responses
        self._queue: list[dict[str, Any] | BaseModel | Exception] | None = (
            list(responses) if isinstance(responses, (list, tuple)) else None
        )
        self._index = 0
        self.calls: list[dict[str, Any]] = []
        self.messages: list[list[dict[str, str]]] = []
        self.cache_hits = 0

    def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        generation_config: GenerationConfig,
    ) -> BaseModel:
        self.messages.append(messages)
        self.calls.append(
            {
                "messages": messages,
                "output_schema": output_schema.__name__,
                "generation_config": generation_config.model_dump(),
            }
        )
        raw: Any
        if callable(self._source):
            raw = self._source(messages, output_schema, generation_config)
        elif self._queue is not None:
            if self._index >= len(self._queue):
                raise AdjudicationFailure(
                    f"ScriptedLLMBackend exhausted after {len(self._queue)} responses"
                )
            raw = self._queue[self._index]
            self._index += 1
        else:
            raw = self._source

        if isinstance(raw, Exception):
            raise raw
        if isinstance(raw, BaseModel):
            if isinstance(raw, output_schema):
                return raw
            raw = raw.model_dump()
        try:
            return output_schema.model_validate(raw)
        except ValidationError as exc:
            raise AdjudicationFailure(
                f"scripted reply does not conform to {output_schema.__name__}: "
                f"{exc.error_count()} error(s)"
            ) from exc


def scripted_model_service(
    responses: ScriptedResponseSource,
    *,
    model_name: str = "scripted-model",
    model_revision: str | None = "scripted-rev-1",
) -> ModelService:
    """Build a :class:`ModelService` wrapping a :class:`ScriptedLLMBackend`."""
    return ModelService(
        backend=ScriptedLLMBackend(responses),
        model_name=model_name,
        model_revision=model_revision,
    )


class ScriptedModelCleaner:
    """Test-only model-backed ``clean`` operator routed via :class:`ModelService`."""

    version = "0.1.0"
    uses_model = True
    options: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        profile: TaskProfile,
        model_service: ModelService,
        *,
        op_id: str = "testing.model_cleaner",
    ) -> None:
        self.profile = profile
        self.model_service = model_service
        self.op_id = op_id

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        lines = [f"Mention: {sau.mention.text}"]
        context = render_declared_context(sau, "clean", self.profile)
        if context:
            lines.append("Context:")
            lines.extend(context)
        messages = [
            {
                "role": "system",
                "content": "Suggest a clean search query string from the mention and declared context.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        result = self.model_service.call(
            operator_id=self.op_id,
            operator_kind="clean",
            task="suggest_queries",
            messages=messages,
            output_schema=_QuerySuggestionsReply,
            prompt_id="suggest_queries:testing.clean",
            prompt_digest=stable_digest(messages[0]["content"], length=12),
        )
        queries = [q.strip() for q in result.reply.queries if q and q.strip()]
        if not queries:
            return sau
        suggestion = queries[0]
        if suggestion == sau.mention.text:
            return sau
        return sau.model_copy(
            update={
                "mention": sau.mention.model_copy(
                    update={"retrieval_text": suggestion, "retrieval_text_from_model": True}
                )
            }
        )


class ScriptedModelMatcher:
    """Test-only model-backed ``match`` operator routed via :class:`ModelService`."""

    version = "0.1.0"
    uses_model = True
    requires_network = False

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        model_service: ModelService,
        *,
        vocabularies: Sequence[str] | None = None,
        op_id: str = "testing.model_matcher",
    ) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.model_service = model_service
        self.op_id = op_id
        self.vocabularies = list(vocabularies or profile.knowledge.resources)
        self.options: dict[str, Any] = {"vocabularies": list(self.vocabularies)}

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(self.vocabularies)

    def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
        query_base = sau.mention.retrieval_text or sau.mention.text
        lines = [f"Mention: {query_base}"]
        context = render_declared_context(sau, "match", self.profile)
        if context:
            lines.append("Context:")
            lines.extend(context)
        messages = [
            {
                "role": "system",
                "content": "Suggest synonymous terminology search strings.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        result = self.model_service.call(
            operator_id=self.op_id,
            operator_kind="match",
            task="suggest_queries",
            messages=messages,
            output_schema=_QuerySuggestionsReply,
            prompt_id="suggest_queries:testing.match",
            prompt_digest=stable_digest(messages[0]["content"], length=12),
        )
        store = self.knowledge.lexicon
        if store is None:
            return CandidateSet(matcher=self.op_id, candidates=[])
        candidates: list[CandidateConcept] = []
        providers = [
            p
            for name in self.vocabularies
            if isinstance((p := self.knowledge.provider_for(name)), LexiconProvider)
        ]
        for query in result.reply.queries:
            text = query.strip()
            if not text:
                continue
            hits = store.exact(text, list(self.vocabularies))
            for provider in providers:
                for cand in provider.candidates(
                    hits, channel=RetrievalChannel.NORMALIZED_LOOKUP, query=text
                ):
                    candidates.append(
                        cand.model_copy(
                            update={
                                "retrieval_evidence": [
                                    ev.model_copy(update={"from_model_suggestion": True})
                                    for ev in cand.retrieval_evidence
                                ]
                            }
                        )
                    )
        return CandidateSet(matcher=self.op_id, candidates=candidates)


class ScriptedModelExpander:
    """Test-only model-backed ``expand`` operator routed via :class:`ModelService`."""

    version = "0.1.0"
    uses_model = True
    options: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        knowledge: KnowledgeRouter,
        profile: TaskProfile,
        model_service: ModelService,
        *,
        op_id: str = "testing.model_expander",
    ) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.model_service = model_service
        self.op_id = op_id

    def expand(
        self, sau: SemanticAlignmentUnit, anchors: Sequence[CandidateConcept]
    ) -> ExpansionSet:
        anchor_codes = [
            QualifiedCode(
                vocabulary=c.vocabulary,
                concept_code=c.candidate_id,
                label=c.preferred_label,
            )
            for c in anchors
        ]
        lines = [f"Mention: {sau.mention.text}"]
        context = render_declared_context(sau, "expand", self.profile)
        if context:
            lines.append("Context:")
            lines.extend(context)
        messages = [
            {
                "role": "system",
                "content": "Suggest related terminology names to expand the anchor set.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        result = self.model_service.call(
            operator_id=self.op_id,
            operator_kind="expand",
            task="suggest_queries",
            messages=messages,
            output_schema=_QuerySuggestionsReply,
            prompt_id="suggest_queries:testing.expand",
            prompt_digest=stable_digest(messages[0]["content"], length=12),
        )
        store = self.knowledge.lexicon
        if store is None:
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)
        target_vocab = self.profile.target.terminology
        provider = self.knowledge.provider_for(target_vocab)
        if not isinstance(provider, LexiconProvider):
            return ExpansionSet(expander=self.op_id, anchors=anchor_codes)

        anchor_ref = (
            anchor_codes[0]
            if anchor_codes
            else QualifiedCode(
                vocabulary=target_vocab, concept_code="ANCHOR", label=sau.mention.text
            )
        )
        members: list[ExpansionMember] = []
        seen: set[tuple[str, str]] = set()
        limit = self.profile.limits.members
        for query in result.reply.queries:
            text = query.strip()
            if not text:
                continue
            hits = store.exact(text, [target_vocab])
            for cand in provider.candidates(
                hits, channel=RetrievalChannel.ONTOLOGY_NEIGHBORHOOD, query=text
            ):
                key = (cand.vocabulary.casefold(), cand.candidate_id)
                if key in seen:
                    continue
                seen.add(key)
                marked = cand.model_copy(
                    update={
                        "retrieval_evidence": [
                            ev.model_copy(update={"from_model_suggestion": True})
                            for ev in cand.retrieval_evidence
                        ]
                    }
                )
                to_code = QualifiedCode(
                    vocabulary=cand.vocabulary,
                    concept_code=cand.candidate_id,
                    label=cand.preferred_label,
                )
                members.append(
                    ExpansionMember(
                        concept=marked,
                        anchor=anchor_ref,
                        hops=1,
                        path=[
                            ExpansionStep(
                                from_code=anchor_ref,
                                predicate="lexical_suggestion",
                                direction="lexical",
                                to_code=to_code,
                            )
                        ],
                        via_operator=self.op_id,
                        from_model_suggestion=True,
                    )
                )
        return ExpansionSet(
            expander=self.op_id,
            anchors=anchor_codes,
            members=members[:limit],
            truncated=len(members) > limit,
        )


class ScriptedModelLinker:
    """Test-only model-backed ``link`` operator attaching ``llm_internal_knowledge`` edges."""

    version = "0.1.0"
    uses_model = True
    options: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        inner: Linker,
        profile: TaskProfile,
        model_service: ModelService,
        *,
        op_id: str = "testing.model_linker",
    ) -> None:
        self.inner = inner
        self.profile = profile
        self.model_service = model_service
        self.op_id = op_id

    def link(
        self, sau: SemanticAlignmentUnit, candidate_sets: Sequence[CandidateSet]
    ) -> KnowledgePackage:
        package = self.inner.link(sau, candidate_sets)
        lines = [f"Mention: {sau.mention.text}"]
        lines.extend(
            f"Candidate: {c.vocabulary}:{c.candidate_id} ({c.preferred_label})"
            for c in package.candidates
        )
        messages = [
            {
                "role": "system",
                "content": "Optionally propose relationship assertions carried as LLM internal knowledge.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        result = self.model_service.call(
            operator_id=self.op_id,
            operator_kind="link",
            task="suggest_links",
            messages=messages,
            output_schema=_LinkSuggestionsReply,
            prompt_id="suggest_links:testing.link",
            prompt_digest=stable_digest(messages[0]["content"], length=12),
        )
        extra_guidance = list(package.kg_guidance)
        candidates = []
        assertions = result.reply.assertions
        for candidate in package.candidates:
            added = [
                KnowledgeStatement(
                    subject_id=item.subject_code,
                    predicate=item.predicate,
                    object_id=item.object_code,
                    object_vocabulary=item.object_vocabulary or candidate.vocabulary,
                    subject_label=candidate.preferred_label,
                    object_label=item.object_label or item.object_code,
                    source_resource=f"model:{self.model_service.model_name}",
                    authority=Authority.LLM_INTERNAL_KNOWLEDGE,
                )
                for item in assertions
                if item.subject_code == candidate.candidate_id
            ]
            if added:
                extra_guidance.extend(added)
                candidates.append(
                    candidate.model_copy(
                        update={"relationships": [*candidate.relationships, *added]}
                    )
                )
            else:
                candidates.append(candidate)
        return package.model_copy(update={"candidates": candidates, "kg_guidance": extra_guidance})


class ScriptedModelCheck:
    """Test-only model-backed ``check`` operator routed via :class:`ModelService`."""

    version = "0.1.0"
    uses_model = True
    options: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        profile: TaskProfile,
        model_service: ModelService,
        *,
        op_id: str = "testing.model_check",
    ) -> None:
        self.profile = profile
        self.model_service = model_service
        self.op_id = op_id

    def check(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]:
        del package
        lines = [
            f"Proposal decision: {proposal.decision.value}",
            "Selected: " + ", ".join(c.concept_id for c in proposal.selected_concepts),
        ]
        if sau is not None:
            lines.insert(0, f"Mention: {sau.mention.text}")
            context = render_declared_context(sau, "check", self.profile)
            if context:
                lines.append("Context:")
                lines.extend(context)
        messages = [
            {
                "role": "system",
                "content": "Review the proposed mapping and report any optional warnings or notes.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        result = self.model_service.call(
            operator_id=self.op_id,
            operator_kind="check",
            task="review_proposal",
            messages=messages,
            output_schema=_CheckReply,
            prompt_id="review_proposal:testing.check",
            prompt_digest=stable_digest(messages[0]["content"], length=12),
        )
        return [
            ValidationFinding(
                code=item.code,
                severity=Severity(item.severity),
                message=item.message,
                concept_id=item.concept_id,
            )
            for item in result.reply.findings
        ]


class ScriptedModelSelector:
    """Test-only model-backed ``select`` operator routed via :class:`ModelService` (S13)."""

    version = "0.1.0"
    uses_model = True
    requires_network = False

    def __init__(
        self,
        knowledge: KnowledgeRouter | None,
        profile: TaskProfile,
        model_service: ModelService,
        *,
        page_size: int = 25,
        op_id: str = "testing.model_selector",
    ) -> None:
        self.knowledge = knowledge
        self.profile = profile
        self.model_service = model_service
        self.page_size = page_size
        self.op_id = op_id
        self.options: dict[str, Any] = {"page_size": page_size}
        self._baseline = HierarchySelector(knowledge, profile)

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            backend=self.model_service.backend_name,
            model_name=self.model_service.model_name,
            model_checkpoint=self.model_service.model_revision,
        )

    def select(self, sau: SemanticAlignmentUnit, expansion: ExpansionSet) -> ConceptSetProposal:
        baseline = self._baseline.select(sau, expansion)
        if not baseline.members:
            return baseline

        context_lines = render_declared_context(sau, "select", self.profile)
        judgments: dict[int, MemberJudgment] = {}
        total = len(baseline.members)
        for start in range(0, total, self.page_size):
            chunk = baseline.members[start : start + self.page_size]
            lines = [f"Idea: {sau.mention.text}"]
            if context_lines:
                lines.append("Context:")
                lines.extend(context_lines)
            lines.append("Members:")
            for offset, member in enumerate(chunk, start=1):
                index = start + offset
                lines.append(
                    f"  {index}. {member.label} [{member.concept_class or '?'}] "
                    f"(hierarchy={member.relation.value}, scope={member.scope})"
                )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Judge which numbered candidate members belong in the concept set, "
                        "their relationship to the idea, and narrow/broad scope."
                    ),
                },
                {"role": "user", "content": "\n".join(lines)},
            ]
            result = self.model_service.call(
                operator_id=self.op_id,
                operator_kind="select",
                task="judge_members",
                messages=messages,
                output_schema=JudgeMembersReply,
                prompt_id="judge_members:testing.select",
                prompt_digest=stable_digest(messages[0]["content"], length=12),
            )
            for item in result.reply.judgments:
                judgments[item.member] = item

        updated_members: list[ConceptSetMember] = []
        for idx, base_member in enumerate(baseline.members, start=1):
            choice = judgments.get(idx)
            if choice is None:
                updated_members.append(
                    base_member.model_copy(update={"hierarchy_relation": base_member.relation})
                )
                continue
            disagrees = choice.relationship is not base_member.relation
            if disagrees:
                reason = (
                    f"Model stated {choice.relationship.value} ({choice.scope}) while "
                    f"terminology hierarchy stated {base_member.relation.value} "
                    f"({base_member.scope}); flagged for reviewer."
                )
            else:
                reason = base_member.reason
            updated_members.append(
                base_member.model_copy(
                    update={
                        "included": choice.included,
                        "relation": choice.relationship,
                        "scope": choice.scope,
                        "reason": reason,
                        "hierarchy_relation": base_member.relation,
                        "model_disagreement": disagrees,
                        "model_rationale": choice.rationale,
                    }
                )
            )
        return baseline.model_copy(
            update={"members": updated_members, "requested_human_review": True}
        )
