"""Task-profile schema (Schema 2, §5.8).

Project behaviour is configured, not hard-coded. A profile is validated
strictly on load so that a typo in ``allowed_relationships`` or an unknown
operator fails loudly at startup rather than silently altering behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator

from biosemalign.enums import (
    DecisionType,
    MentionMode,
    RelationshipType,
    SemanticType,
    Severity,
    SourceType,
)
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.sau import ContextEnvelope, ContextualAttributes

if TYPE_CHECKING:
    from biosemalign.targets.spec import TargetSpec

__all__ = [
    "ContextConfig",
    "ContextFieldConfig",
    "InputConfig",
    "KnowledgeConfig",
    "LimitsConfig",
    "OperatorConfig",
    "RoutingConfig",
    "TargetConfig",
    "TaskConfig",
    "TaskProfile",
    "TierConfig",
    "domain_fields",
    "register_domain_fields",
    "unregister_domain_fields",
]

ContextUse = Literal["key", "clean", "match", "expand", "resolve", "select", "check"]
ContextKind = Literal["attribute", "context"]
WorkflowKind = Literal["mapping", "coding", "phenotyping"]

_OCCURRENCE_ONLY_MAPPING_KEY_FIELDS = frozenset(
    {"record_id", "evidence_block_id", "neighboring_fields", "extensions"}
)
_SUPPORTED_MAPPING_KEY_CONTEXT_FIELDS = frozenset(
    (set(ContextEnvelope.model_fields) | set(ContextualAttributes.model_fields))
    - _OCCURRENCE_ONLY_MAPPING_KEY_FIELDS
)
#: Context a resolver or selector can be blinded to. Fields mirrored as contextual
#: attributes are excluded: a decision must preserve those, so hiding them
#: would make every proposal fail attribute validation.
_BLINDABLE_CONTEXT_FIELDS = frozenset(
    set(ContextEnvelope.model_fields) - set(ContextualAttributes.model_fields)
)


class OperatorConfig(BioSemAlignModel):
    """A registered operator and its options, e.g. ``{op: drug.rxnav}``."""

    op: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: Any) -> Any:
        return {"op": value} if isinstance(value, str) else value


class TaskConfig(BioSemAlignModel):
    operation: str
    mapping_intent: str
    relation_types: list[RelationshipType] = Field(
        default_factory=list,
        description="Relationship classes a resolver may emit for this task.",
    )


class TargetConfig(BioSemAlignModel):
    terminology: str
    spec: str | None = Field(
        default=None,
        description=(
            "Target spec: a built-in id such as 'rxnorm', or a path to a spec YAML. "
            "None uses the built-in spec of terminology."
        ),
    )
    level: str | None = Field(
        default=None,
        description="Requested granularity / concept level in the target terminology.",
    )
    source_terminologies: list[str] = Field(
        default_factory=list,
        description=(
            "Vocabularies besides the target that a source-level concept may come from, "
            "e.g. ['RxNorm Extension'] for non-US brands whose ingredients are RxNorm "
            "concepts. Analysis-level concepts always come from the target terminology."
        ),
    )
    require_active_concept: bool = True

    @model_validator(mode="before")
    @classmethod
    def _default_terminology_from_spec(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("terminology") or not value.get("spec"):
            return value
        from biosemalign.targets import builtin_specs

        builtin = builtin_specs().get(str(value["spec"]))
        if builtin is None:
            return value
        return {**value, "terminology": builtin.target}

    def source_vocabulary(self, vocabulary: str | None) -> str:
        """Where a source-level concept of ``vocabulary`` is looked up.

        A permitted source terminology is looked up in itself; anything else
        in the target, where an impostor finds no evidence and fails.
        """
        if vocabulary is not None and vocabulary.casefold() in {
            v.casefold() for v in self.source_terminologies
        }:
            return vocabulary
        return self.terminology


class InputConfig(BioSemAlignModel):
    source_type: SourceType = SourceType.STRUCTURED_FIELD
    default_semantic_type: SemanticType = SemanticType.UNKNOWN
    automatic_mention_detection: bool = False
    value_field: str | None = Field(
        default=None, description="For record input, which key holds the value to map."
    )
    mention_not_in_source_severity: Severity = Field(
        default=Severity.WARNING,
        description="Severity emitted when Mention.fidelity is 'not_found' (D12).",
    )

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


class ContextFieldConfig(BioSemAlignModel):
    """Declaration of one context or attribute field."""

    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    kind: ContextKind = "context"
    use: list[ContextUse] = Field(default_factory=list)
    required: bool = False
    blinded: bool = Field(
        default=False,
        description=(
            "Context fields withheld from resolvers/selectors because they can carry the "
            "studied outcome (e.g. an indication)."
        ),
    )

    @model_validator(mode="after")
    def _validate_blinding_and_key_rules(self) -> ContextFieldConfig:
        if "key" in self.use and self.name in _OCCURRENCE_ONLY_MAPPING_KEY_FIELDS:
            raise ValueError(
                "context field with use='key' cannot be an occurrence-only field that "
                f"prevents safe reuse: {self.name!r}"
            )
        if self.blinded:
            if (
                self.kind == "attribute"
                or self.name in ContextualAttributes.model_fields
                or self.name in {"neighboring_fields", "extensions"}
            ):
                raise ValueError(
                    f"blinded field {self.name!r} cannot be blinded; contextual-attribute fields "
                    "(route, dose, unit, species) must be preserved by every decision. "
                    f"Blindable fields: {sorted(_BLINDABLE_CONTEXT_FIELDS)}"
                )
            disallowed = [u for u in self.use if u in {"resolve", "select"}]
            if disallowed:
                raise ValueError(f"blinded field {self.name!r} cannot list {disallowed!r} in use")
        return self


class ContextConfig(BioSemAlignModel):
    """Declared context and attribute fields."""

    fields: list[ContextFieldConfig] = Field(default_factory=list)

    @field_validator("fields")
    @classmethod
    def _unique_field_names(cls, fields: list[ContextFieldConfig]) -> list[ContextFieldConfig]:
        names = [field.name for field in fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"context.fields contains duplicate fields: {duplicates}")
        return fields

    @property
    def required_context(self) -> list[str]:
        return [field.name for field in self.fields if field.required]

    @property
    def optional_context(self) -> list[str]:
        return [field.name for field in self.fields if not field.required and field.use != ["key"]]

    @property
    def mapping_key_context(self) -> list[str]:
        return [field.name for field in self.fields if "key" in field.use]

    @property
    def blinded_fields(self) -> list[str]:
        return sorted(field.name for field in self.fields if field.blinded)


class KnowledgeConfig(BioSemAlignModel):
    resources: list[str] = Field(default_factory=list)
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
        return self

    def permits(self, predicate: str) -> bool:
        """Whether a predicate may enter the Knowledge Package.

        Prohibition always wins. An empty allowlist means "no restriction",
        which keeps profiles for exploratory tasks short.
        """
        if predicate in self.prohibited_relationships:
            return False
        return not self.allowed_relationships or predicate in self.allowed_relationships


class LimitsConfig(BioSemAlignModel):
    """Bounds on knowledge packages, expansions, and progressive enrichment."""

    candidates: int = Field(default=10, ge=1, le=100)
    members: int = Field(default=2000, ge=1)
    enrichment_batch_size: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of ranked candidates enriched before the stopping rule is re-evaluated.",
    )


class RoutingConfig(BioSemAlignModel):
    exact_synonym_auto_accept: bool = True
    combination_products_require_review: bool = False
    unresolved_salt_ambiguity_requires_review: bool = True
    historical_candidate_requires_review: bool = Field(
        default=True,
        description="A mapping resting on an obsolete concept is never auto-accepted.",
    )
    allow_model_auto_abstain: bool = Field(
        default=False,
        description=(
            "When False (M3), terms declined by a model route to human review instead of "
            "automatic abstention."
        ),
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
    calibration: str | None = Field(
        default=None,
        description=(
            "Path to a calibration fitted by 'biosemalign calibrate'. When set, an eligible "
            "decision is accepted automatically only above its tier's certified threshold."
        ),
    )
    rejection_lexical_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Lexical agreement below which the resolver rejects every candidate. "
            "RxNav's approximate matcher always returns its nearest neighbour, so "
            "without a floor an unmappable string maps confidently to something."
        ),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-pack routing options.",
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


class TierConfig(BioSemAlignModel):
    """One tier of a workflow cascade."""

    name: str = Field(min_length=1)
    clean: list[OperatorConfig] = Field(default_factory=list)
    match: list[OperatorConfig] = Field(
        default_factory=lambda: [OperatorConfig(op="exact")],
        min_length=1,
        description="Queried in order; more than one forms a parallel union.",
    )
    expand: OperatorConfig | list[OperatorConfig] | None = None
    link: OperatorConfig | None = Field(default_factory=lambda: OperatorConfig(op="standard"))
    resolve: OperatorConfig | None = Field(default_factory=lambda: OperatorConfig(op="rules"))
    select: OperatorConfig | None = None
    pass_on: list[DecisionType] = Field(
        default_factory=list,
        description="Proposal outcomes handed to the next tier instead of being finalized.",
    )
    pass_on_review_request: bool = Field(
        default=False,
        description="Hand proposals that request human review to the next tier.",
    )

    @property
    def expand_ops(self) -> list[OperatorConfig]:
        if self.expand is None:
            return []
        if isinstance(self.expand, list):
            return list(self.expand)
        return [self.expand]

    @field_validator("clean", "match", mode="before")
    @classmethod
    def _coerce_single_operator(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, list):
            return [value]
        return value


_DOMAIN_CONTEXT_FIELDS: dict[str, tuple[ContextFieldConfig, ...]] = {}


def register_domain_fields(domain: str, fields: list[ContextFieldConfig]) -> None:
    """Register default context/attribute field declarations for ``domain``."""
    _DOMAIN_CONTEXT_FIELDS[domain.casefold()] = tuple(fields)


def unregister_domain_fields(domain: str) -> None:
    """Remove registered context field declarations for ``domain``."""
    _DOMAIN_CONTEXT_FIELDS.pop(domain.casefold(), None)


def domain_fields(domain: str | None) -> list[ContextFieldConfig]:
    """Copy of context field declarations registered by ``domain``."""
    if not domain:
        return []
    return [
        field.model_copy(deep=True) for field in _DOMAIN_CONTEXT_FIELDS.get(domain.casefold(), ())
    ]


class TaskProfile(BioSemAlignModel):
    """One project's complete Schema 2 configuration of the framework (§5.8)."""

    profile_schema: Literal[2] = 2
    profile_name: str
    profile_version: str
    description: str | None = None
    domain: str | None = None
    workflow: WorkflowKind = "mapping"

    task: TaskConfig
    target: TargetConfig
    input: InputConfig = Field(default_factory=InputConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    tiers: list[TierConfig] | None = Field(
        default=None,
        description="Operator cascade; omitted means the default single-tier mapping cascade.",
    )
    checks: list[OperatorConfig] = Field(
        default_factory=list,
        description=(
            "Check operators run after the mandatory kernel checks, in order. "
            "The mandatory terminology checks cannot be removed."
        ),
    )
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @field_validator("tiers")
    @classmethod
    def _validate_tiers(cls, tiers: list[TierConfig] | None) -> list[TierConfig] | None:
        if tiers is None:
            return None
        if not tiers:
            raise ValueError("tiers must not be empty; omit or set to null for default cascade")
        names = [tier.name for tier in tiers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"workflow tier names must be unique: {duplicates}")
        return tiers

    @model_validator(mode="after")
    def _target_terminology_is_a_declared_resource(self) -> TaskProfile:
        resources = {r.casefold() for r in self.knowledge.resources}
        if resources and self.target.terminology.casefold() not in resources:
            raise ValueError(
                f"target.terminology {self.target.terminology!r} is not among "
                f"knowledge.resources {self.knowledge.resources!r}; the profile "
                "would ask for concepts from a vocabulary it never queries"
            )
        return self

    def target_spec(self) -> TargetSpec:
        """The target terminology spec this profile maps into."""
        from biosemalign.targets import load_target_spec

        return load_target_spec(self.target.spec, terminology=self.target.terminology)

    def effective_context_fields(self) -> list[ContextFieldConfig]:
        """Declared context/attribute fields merged with domain-pack defaults."""
        base: list[ContextFieldConfig] = []
        if self.domain:
            base = domain_fields(self.domain)
            if not base:
                from biosemalign.plugins import ensure_plugins_loaded

                ensure_plugins_loaded()
                base = domain_fields(self.domain)
        profile_by_name = {field.name: field for field in self.context.fields}
        seen: set[str] = set()
        merged: list[ContextFieldConfig] = []
        for field in base:
            if field.name in profile_by_name:
                merged.append(profile_by_name[field.name])
            else:
                merged.append(field)
            seen.add(field.name)
        for field in self.context.fields:
            if field.name not in seen:
                merged.append(field)
                seen.add(field.name)
        return merged

    def effective_mapping_key_fields(self) -> list[str]:
        """Derived mapping-key fields: every attribute field plus any non-empty-use field."""
        return [
            field.name
            for field in self.effective_context_fields()
            if (field.kind == "attribute" or bool(field.use))
            and field.name not in _OCCURRENCE_ONLY_MAPPING_KEY_FIELDS
        ]

    def effective_blinded_fields(self) -> list[str]:
        """Sorted names of effective context fields blinded from resolvers/selectors."""
        return sorted(field.name for field in self.effective_context_fields() if field.blinded)

    def permits_relation(self, relation: RelationshipType) -> bool:
        """Whether a resolver may emit this relationship class for this task."""
        return not self.task.relation_types or relation in self.task.relation_types
