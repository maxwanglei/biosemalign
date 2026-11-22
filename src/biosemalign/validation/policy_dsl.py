"""Declarative policy packs: domain guidance as data, not engine code (v0.2).

A client states clinical guidance — distinctions that must not be collapsed,
context that contradicts a formulation, mentions that name a control rather
than a drug — as YAML, and the engine applies it as a validation operator.
The rule types are a closed set validated strictly on load; there are no
expressions to evaluate, so a policy pack cannot execute anything.

Every finding is a WARNING by default, which routes the decision to human
review without discarding it; a rule may raise ERROR to reject outright. The
pack's content digest is part of the operator version, so the validation
rules version recorded on every decision changes whenever a rule does.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from biosemalign.enums import Authority, ErrorCategory, RelationshipType, Severity
from biosemalign.exceptions import ConfigurationError
from biosemalign.input.context import context_view
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept, ValidationFinding
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.util import stable_digest

__all__ = [
    "POLICY_DSL_VERSION",
    "CategoryTagRule",
    "ConceptSelector",
    "ContextCompatibilityRule",
    "PolicyCheck",
    "PolicyPack",
    "PreserveDistinctionRule",
    "load_policy_pack",
]

POLICY_DSL_VERSION = "1"

#: Relationships that roll a source up to something coarser.
_COLLAPSING = frozenset(
    {RelationshipType.TARGET_BROADER_THAN_SOURCE, RelationshipType.MULTIPLE_COMPONENTS_REQUIRED}
)


def _compile(pattern: str | None) -> re.Pattern[str] | None:
    return re.compile(pattern, re.IGNORECASE) if pattern else None


class _Pattern(BioSemAlignModel):
    @staticmethod
    def _check(pattern: str | None) -> str | None:
        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from exc
        return pattern


class ConceptSelector(_Pattern):
    """Matches a concept by code, by label pattern, or both."""

    vocabulary: str | None = None
    codes: list[str] = Field(default_factory=list)
    label_pattern: str | None = None

    @field_validator("label_pattern")
    @classmethod
    def _valid_pattern(cls, value: str | None) -> str | None:
        return cls._check(value)

    @model_validator(mode="after")
    def _selects_something(self) -> ConceptSelector:
        if not self.codes and not self.label_pattern:
            raise ValueError("a concept selector needs codes, a label_pattern, or both")
        return self

    def matches(self, concept: SelectedConcept) -> bool:
        if self.vocabulary and concept.vocabulary.casefold() != self.vocabulary.casefold():
            return False
        if self.codes and concept.concept_id not in self.codes:
            return False
        pattern = _compile(self.label_pattern)
        return pattern is None or bool(pattern.search(concept.label))


class _Rule(_Pattern):
    rule_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["error", "warning", "info"] = "warning"

    def finding(self, code: str, concept_id: str | None = None) -> ValidationFinding:
        return ValidationFinding(
            code=code,
            severity=Severity(self.severity),
            message=f"[{self.rule_id}] {self.message}",
            concept_id=concept_id,
            error_category=ErrorCategory.UNSUPPORTED_SPECIFICITY
            if code == "CRITICAL_SEMANTIC_LOSS"
            else ErrorCategory.VALIDATION_ERROR,
        )


class PreserveDistinctionRule(_Rule):
    """A source whose rollup to a broader concept loses a material distinction.

    For example, caffeine citrate dosed for neonatal apnea is not
    interchangeable with caffeine base, so mapping it to the ingredient must be
    looked at rather than accepted.
    """

    source: ConceptSelector


class ContextCompatibilityRule(_Rule):
    """Context that contradicts the formulation a candidate names.

    Fires when the SAU's ``context_field`` matches ``context_pattern`` and the
    source concept's label or dose forms match ``incompatible_pattern`` — an
    intramuscular route against a topical cream, say.
    """

    context_field: str = Field(min_length=1)
    context_pattern: str
    incompatible_pattern: str

    @field_validator("context_pattern", "incompatible_pattern")
    @classmethod
    def _valid_patterns(cls, value: str) -> str:
        cls._check(value)
        return value


class CategoryTagRule(_Rule):
    """Mentions that name a non-target category, such as a placebo or a class."""

    tag: str = Field(min_length=1, description="e.g. CONTROL_VEHICLE, DRUG_CLASS")
    mention_pattern: str

    @field_validator("mention_pattern")
    @classmethod
    def _valid_pattern(cls, value: str) -> str:
        cls._check(value)
        return value


class PolicyPack(BioSemAlignModel):
    """A versioned set of declarative rules."""

    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    description: str | None = None
    preserve_distinction_pairs: list[PreserveDistinctionRule] = Field(default_factory=list)
    context_compatibility_rules: list[ContextCompatibilityRule] = Field(default_factory=list)
    non_target_category_tags: list[CategoryTagRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> PolicyPack:
        ids = [
            rule.rule_id
            for rules in (
                self.preserve_distinction_pairs,
                self.context_compatibility_rules,
                self.non_target_category_tags,
            )
            for rule in rules
        ]
        duplicates = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"policy rule ids must be unique: {duplicates}")
        return self

    @property
    def digest(self) -> str:
        return stable_digest(self.model_dump(mode="json"), length=12)


def load_policy_pack(source: str | Path | dict[str, Any]) -> PolicyPack:
    """Load a pack from a YAML file or an inline mapping."""
    if isinstance(source, dict):
        payload: Any = source
    else:
        path = Path(source)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"cannot read policy pack {path}: {exc}") from exc
    try:
        return PolicyPack.model_validate(payload)
    except ValueError as exc:
        raise ConfigurationError(f"invalid policy pack: {exc}") from exc


class PolicyCheck:
    """Apply a policy pack after the mandatory terminology checks."""

    op_id = "policy"
    uses_model = False

    def __init__(
        self,
        pack: PolicyPack,
        *,
        target_terminology: str,
        profile: TaskProfile | None = None,
    ) -> None:
        self.pack = pack
        self.target = target_terminology
        self.profile = profile
        self.version = f"{POLICY_DSL_VERSION}:{pack.policy_id}@{pack.policy_version}#{pack.digest}"

    @property
    def options(self) -> dict[str, Any]:
        return {"policy_id": self.pack.policy_id, "policy_version": self.pack.policy_version}

    def check(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        source = proposal.source_level_concept

        if source is not None and proposal.relationship_to_source in _COLLAPSING:
            for rule in self.pack.preserve_distinction_pairs:
                if rule.source.matches(source):
                    findings.append(rule.finding("CRITICAL_SEMANTIC_LOSS", source.concept_id))

        if source is not None and sau is not None:
            candidate = package.candidate_by_qualified_id(self.target, source.concept_id)
            forms = [source.label]
            if candidate is not None:
                forms.extend(
                    statement.object_label or statement.object_id
                    for statement in candidate.relationships
                    if statement.predicate == "has_dose_form"
                    and statement.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
                )
            for context_rule in self.pack.context_compatibility_rules:
                value = _context_value(sau, context_rule.context_field, self.profile)
                if value is None or not re.search(
                    context_rule.context_pattern, value, re.IGNORECASE
                ):
                    continue
                if any(
                    re.search(context_rule.incompatible_pattern, form, re.IGNORECASE)
                    for form in forms
                ):
                    findings.append(context_rule.finding("CONTEXT_INCOMPATIBLE", source.concept_id))

        if sau is not None:
            for tag_rule in self.pack.non_target_category_tags:
                if re.search(tag_rule.mention_pattern, sau.mention.text, re.IGNORECASE):
                    finding = tag_rule.finding("NON_TARGET_CATEGORY")
                    finding.message = f"[{tag_rule.rule_id}] {tag_rule.tag}: {tag_rule.message}"
                    findings.append(finding)
        return findings


def _context_value(
    sau: SemanticAlignmentUnit, field: str, profile: TaskProfile | None = None
) -> str | None:
    if profile is not None:
        value = context_view(sau, "check", profile).get(field)
        return str(value) if value not in (None, "") else None
    for container in (sau.attributes, sau.context):
        value = getattr(container, field, None)
        if isinstance(value, str) and value:
            return value
    extension = sau.context.extensions.get(field)
    return str(extension) if extension not in (None, "") else None
