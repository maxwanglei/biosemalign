"""Task-profile schema (§14).

Project behaviour is configured, not hard-coded. A profile is validated
strictly on load so that a typo in ``allowed_relationships`` fails loudly at
startup rather than silently widening what the adjudicator may consider.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from biosemalign.enums import MentionMode, RelationshipType, SemanticType, SourceType
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.sau import ContextEnvelope, ContextualAttributes

__all__ = [
    "AdjudicationConfig",
    "InputConfig",
    "KnowledgeConfig",
    "RetrievalConfig",
    "RoutingConfig",
    "TaskConfig",
    "TaskProfile",
    "ValidationConfig",
]

_OCCURRENCE_ONLY_MAPPING_KEY_FIELDS = frozenset(
    {"record_id", "evidence_block_id", "neighboring_fields", "extensions"}
)
_SUPPORTED_MAPPING_KEY_CONTEXT_FIELDS = frozenset(
    (set(ContextEnvelope.model_fields) | set(ContextualAttributes.model_fields))
    - _OCCURRENCE_ONLY_MAPPING_KEY_FIELDS
)


class TaskConfig(BioSemAlignModel):
    operation: str
    mapping_intent: str
    target_terminology: str
    desired_granularity: str | None = None


class InputConfig(BioSemAlignModel):
    source_type: SourceType = SourceType.STRUCTURED_FIELD
    default_semantic_type: SemanticType = SemanticType.UNKNOWN
    automatic_mention_detection: bool = False
    value_field: str | None = Field(
        default=None, description="For record input, which key holds the value to map."
    )
    required_context: list[str] = Field(
        default_factory=list,
        description="Context fields whose absence downgrades SAU readiness (§5.8).",
    )
    optional_context: list[str] = Field(
        default_factory=list,
        description="Context fields captured when present; absence is not a warning.",
    )
    mapping_key_context: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical context or attribute fields that distinguish reusable mappings. "
            "Source-row identifiers are deliberately excluded."
        ),
    )

    @field_validator("mapping_key_context")
    @classmethod
    def _validate_mapping_key_context(cls, fields: list[str]) -> list[str]:
        duplicates = sorted({field for field in fields if fields.count(field) > 1})
        if duplicates:
            raise ValueError(f"mapping_key_context contains duplicate fields: {duplicates}")

        occurrence_only = sorted(set(fields) & _OCCURRENCE_ONLY_MAPPING_KEY_FIELDS)
        if occurrence_only:
            raise ValueError(
                "mapping_key_context contains occurrence-only fields that prevent safe reuse: "
                f"{occurrence_only}"
            )

        unknown = sorted(set(fields) - _SUPPORTED_MAPPING_KEY_CONTEXT_FIELDS)
        if unknown:
            raise ValueError(
                f"mapping_key_context contains unknown canonical fields: {unknown}; "
                f"supported fields: {sorted(_SUPPORTED_MAPPING_KEY_CONTEXT_FIELDS)}"
            )
        return fields

    @model_validator(mode="after")
    def _reject_unimplemented_mention_detection(self) -> InputConfig:
        if self.automatic_mention_detection:
            raise ValueError(
                "automatic_mention_detection is not implemented in v0.1; callers must "
                "supply the mention explicitly"
            )
        return self

    @property
    def mention_mode(self) -> MentionMode:
        return (
            MentionMode.AUTOMATIC_DETECTION
            if self.automatic_mention_detection
            else MentionMode.USER_SPECIFIED
        )


class KnowledgeConfig(BioSemAlignModel):
    primary_resources: list[str] = Field(default_factory=list)
    secondary_resources: list[str] = Field(default_factory=list)
    allowed_relationships: list[str] = Field(default_factory=list)
    prohibited_relationships: list[str] = Field(
        default_factory=list,
        description=(
            "Predicates that must never enter the Knowledge Package. This is the "
            "outcome-leakage guard: during PHPT signal discovery, drug-to-PHPT "
            "associations would bias candidate discovery (§6.2)."
        ),
    )

    @model_validator(mode="after")
    def _no_predicate_in_both_lists(self) -> KnowledgeConfig:
        overlap = set(self.allowed_relationships) & set(self.prohibited_relationships)
        if overlap:
            raise ValueError(
                f"predicates appear in both allowed and prohibited lists: {sorted(overlap)}"
            )
        if self.secondary_resources:
            raise ValueError(
                "secondary_resources are not queried in v0.1; remove them rather than "
                "silently omitting their evidence"
            )
        return self

    def permits(self, predicate: str) -> bool:
        """Whether a predicate may enter the Knowledge Package.

        Prohibition always wins. An empty allowlist means "no restriction",
        which keeps profiles for exploratory tasks short.
        """
        if predicate in self.prohibited_relationships:
            return False
        return not self.allowed_relationships or predicate in self.allowed_relationships


class RetrievalConfig(BioSemAlignModel):
    exact_lookup: bool = True
    terminology_approximate_match: bool = True
    lexical_retrieval: bool = False
    embedding_retrieval: bool = False
    include_historical_candidates: bool = Field(
        default=True,
        description=(
            "Search obsolete and non-primary-source concepts as a last resort. "
            "They are flagged, never silently mixed with current concepts, and "
            "are the only way to resolve withdrawn products in historical FAERS "
            "reports."
        ),
    )
    maximum_candidates: int = Field(default=10, ge=1, le=100)
    progressive_enrichment_batch_size: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of ranked candidates enriched before the stopping rule is re-evaluated.",
    )

    @model_validator(mode="after")
    def _reject_unimplemented_channels(self) -> RetrievalConfig:
        unsupported = [
            name
            for name, enabled in (
                ("lexical_retrieval", self.lexical_retrieval),
                ("embedding_retrieval", self.embedding_retrieval),
            )
            if enabled
        ]
        if unsupported:
            raise ValueError(
                f"retrieval channels are not implemented in v0.1: {', '.join(unsupported)}"
            )
        return self


class AdjudicationConfig(BioSemAlignModel):
    candidate_constrained: bool = True
    backend: str = Field(
        default="deterministic",
        description="Adjudicator to use: 'deterministic', 'mock', or 'vllm'.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    relation_types: list[RelationshipType] = Field(
        default_factory=list,
        description="Relationship classes the adjudicator may emit for this task.",
    )
    prompt_version: str | None = None

    @model_validator(mode="after")
    def _require_candidate_constraint(self) -> AdjudicationConfig:
        if not self.candidate_constrained:
            raise ValueError(
                "candidate_constrained=false is not implemented in v0.1; adjudication "
                "must remain bounded by retrieved evidence"
            )
        return self


class ValidationConfig(BioSemAlignModel):
    require_active_concept: bool = True
    preserve_combination_ingredients: bool = True
    prohibit_unretrieved_identifiers: bool = True
    flag_salt_active_moiety_conflation: bool = True

    @model_validator(mode="after")
    def _require_bounded_identifiers(self) -> ValidationConfig:
        if not self.prohibit_unretrieved_identifiers:
            raise ValueError(
                "prohibit_unretrieved_identifiers=false is not supported in v0.1; "
                "every final concept must be owned by bounded canonical evidence"
            )
        return self


class RoutingConfig(BioSemAlignModel):
    exact_synonym_auto_accept: bool = True
    combination_products_require_review: bool = False
    unresolved_salt_ambiguity_requires_review: bool = True
    historical_candidate_requires_review: bool = Field(
        default=True,
        description="A mapping resting on an obsolete concept is never auto-accepted.",
    )
    minimum_auto_accept_score: float | None = Field(
        default=None,
        description="Optional provider score floor for automatic acceptance; None disables it.",
    )
    minimum_lexical_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Lexical agreement a fuzzy-only match must reach before it can be accepted "
            "automatically. Below this it is routed to review rather than rejected."
        ),
    )
    rejection_lexical_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Lexical agreement below which the adjudicator rejects every candidate. "
            "RxNav's approximate matcher always returns its nearest neighbour, so "
            "without a floor an unmappable string maps confidently to something."
        ),
    )

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> RoutingConfig:
        if self.rejection_lexical_similarity > self.minimum_lexical_similarity:
            raise ValueError(
                "rejection_lexical_similarity must not exceed minimum_lexical_similarity; "
                "otherwise candidates would be rejected outright before ever being "
                "eligible for review"
            )
        return self


class TaskProfile(BioSemAlignModel):
    """One project's complete configuration of the framework (§14)."""

    profile_name: str
    profile_version: str
    description: str | None = None
    task: TaskConfig
    input: InputConfig = Field(default_factory=InputConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    adjudication: AdjudicationConfig = Field(default_factory=AdjudicationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @model_validator(mode="after")
    def _target_terminology_is_a_declared_resource(self) -> TaskProfile:
        resources = {r.casefold() for r in self.knowledge.primary_resources}
        if resources and self.task.target_terminology.casefold() not in resources:
            raise ValueError(
                f"target_terminology {self.task.target_terminology!r} is not among "
                f"primary_resources {self.knowledge.primary_resources!r}; the profile "
                "would ask for concepts from a vocabulary it never queries"
            )
        return self

    def permits_relation(self, relation: RelationshipType) -> bool:
        """Whether the adjudicator may emit this relationship class for this task."""
        return not self.adjudication.relation_types or relation in self.adjudication.relation_types
