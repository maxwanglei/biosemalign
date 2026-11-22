"""Target terminology specifications (v0.3).

A target spec declares, as data, what the engine used to hard-code about its
target terminology: which concept classes sit at which granularity, which
relationship walks from a source concept down to a requested level, what a
coarser target loses relative to its source, and which edges are
release-bound terminology evidence. ``rxnorm.yaml`` reproduces the v0.2 drug
rules exactly; other targets (ATC, MeSH, ...) are further specs, not code.

Loss rules are a closed set of rule types, so a spec stays declarative and
the adjudicator and the validator share one implementation: the losses a
mapping reports and the losses the validator expects come from the same
function and cannot drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from biosemalign.enums import Authority, SemanticLossKind
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.util import stable_digest

if TYPE_CHECKING:
    from biosemalign.schemas.decision import SemanticLoss
    from biosemalign.schemas.knowledge import CandidateConcept

__all__ = [
    "GranularitySpec",
    "LossRule",
    "RefinementRule",
    "SaltSpec",
    "TargetSpec",
]


class GranularitySpec(BioSemAlignModel):
    """One analysis level of the target and how a source concept reaches it."""

    via: list[str] = Field(
        default_factory=list,
        description="Predicates walked from a source concept to concepts at this level.",
    )
    max_hops: int = Field(
        default=1,
        ge=1,
        description="Maximum relationship hops walked to reach a terminal class.",
    )
    terminal: list[str] = Field(
        min_length=1,
        description="Concept classes that already sit at this level: a source of one is its own target.",
    )
    accepts: list[str] = Field(
        default_factory=list,
        description=(
            "Classes a target reached through ``via`` may have; defaults to ``terminal``. "
            "Wider when the edge can also lead to other classes (RxNorm's has_boss can "
            "lead to a precise ingredient)."
        ),
    )
    multi: Literal["components", "alternatives"] = Field(
        default="components",
        description=(
            "Meaning of several targets: 'components' — all are required (a combination "
            "product); 'alternatives' — any may apply (a drug in several classes)."
        ),
    )
    combination_description: str | None = Field(
        default=None,
        description=(
            "Semantic-loss text when a source resolves to several components; "
            "'{n}' is the number of components."
        ),
    )


class LossRule(BioSemAlignModel):
    """One way a coarser target loses information present in the source.

    Rule types (closed):

    ``pattern``
        The source class carries a value matched by ``pattern`` in its label,
        or else in the mention; lost when the target label lacks one.
    ``relationship_labels``
        The source class has ``predicate`` edges with labels; lost always.
        With ``fallback_to_source``, emitted even without edges, valued with
        the source label.
    ``source_not_selected``
        The source class is itself the more specific concept, and the source
        is not among the selected targets.
    ``unselected_objects``
        The source (not of ``exclude_classes``) has ``predicate`` objects that
        are not among the selected targets.
    ``context``
        A context or attribute field of the SAU (``field``) is not carried.

    Rules are applied in order; once a kind is emitted, later rules of the
    same kind are skipped, so alternatives are written as successive rules.
    """

    kind: SemanticLossKind
    rule: Literal[
        "pattern",
        "relationship_labels",
        "source_not_selected",
        "unselected_objects",
        "context",
        "ancestor",
    ]
    description: str = Field(min_length=1)
    classes: list[str] = Field(default_factory=list)
    exclude_classes: list[str] = Field(default_factory=list)
    predicate: str | None = None
    pattern: str | None = None
    field: str | None = None
    fallback_to_source: bool = False

    @model_validator(mode="after")
    def _complete(self) -> LossRule:
        needs = {
            "pattern": ("pattern",),
            "relationship_labels": ("predicate",),
            "unselected_objects": ("predicate",),
            "context": ("field",),
            "source_not_selected": (),
            "ancestor": (),
        }[self.rule]
        missing = [name for name in needs if getattr(self, name) in (None, "")]
        if missing:
            raise ValueError(f"loss rule {self.rule!r} needs {missing}")
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"loss rule pattern {self.pattern!r} is invalid: {exc}") from exc
        return self


class SaltSpec(BioSemAlignModel):
    """Where precise forms (salts, esters) sit relative to their base."""

    precise_classes: list[str] = Field(min_length=1)
    base_classes: list[str] = Field(min_length=1)
    precise_predicate: str


class RefinementRule(BioSemAlignModel):
    """Rule describing how a coarse concept refines into specific concepts using SAU slots."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_classes: list[str] = Field(alias="from", min_length=1)
    to_classes: list[str] = Field(alias="to", min_length=1)
    inverse_of: list[str] = Field(min_length=1)
    slots: list[str] = Field(min_length=1)


class TargetSpec(BioSemAlignModel):
    """A target terminology's granularity model, as data."""

    spec_id: str = Field(min_length=1, description="Short identifier, e.g. 'rxnorm'.")
    version: str = Field(min_length=1)
    target: str = Field(min_length=1, description="Target vocabulary, e.g. 'RxNorm'.")
    family: list[str] = Field(
        default_factory=list,
        description="Vocabularies whose concepts and edges count as the target's own.",
    )
    class_aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Other spellings of concept classes, e.g. OMOP's 'Ingredient' for 'IN'.",
    )
    subsumption: list[str] = Field(
        default_factory=list,
        description="Predicates pointing from narrower concepts to broader concepts, e.g. ['is_a'].",
    )
    granularities: dict[str, GranularitySpec] = Field(default_factory=dict)
    release_bound_predicates: list[str] = Field(
        default_factory=list,
        description=(
            "Edges whose targets are, by the terminology's own release, active concepts "
            "(official relationships of a current release point only at current concepts)."
        ),
    )
    edge_target_classes: dict[str, str] = Field(
        default_factory=dict,
        description="The class a release-bound edge's target always has, e.g. has_ingredient: IN.",
    )
    losses: list[LossRule] = Field(default_factory=list)
    salt: SaltSpec | None = None
    refinement: list[RefinementRule] = Field(
        default_factory=list,
        description="Rules for refining coarse matched concepts down to specific concepts using slots.",
    )

    @field_validator("family")
    @classmethod
    def _family_is_unique(cls, family: list[str]) -> list[str]:
        folded = [v.casefold() for v in family]
        if len(folded) != len(set(folded)):
            raise ValueError("target family lists a vocabulary twice")
        return family

    @model_validator(mode="after")
    def _family_includes_target(self) -> TargetSpec:
        if self.target.casefold() not in {v.casefold() for v in self.family}:
            self.family.insert(0, self.target)
        return self

    # -- identity --------------------------------------------------------

    @cached_property
    def digest(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True)
        if not payload.get("subsumption"):
            payload.pop("subsumption", None)
        if not payload.get("refinement"):
            payload.pop("refinement", None)
        for level in payload.get("granularities", {}).values():
            if isinstance(level, dict) and level.get("max_hops") == 1:
                level.pop("max_hops", None)
        return stable_digest(payload)

    @property
    def label(self) -> str:
        return f"{self.spec_id}@{self.version}"

    def in_family(self, vocabulary: str | None) -> bool:
        return vocabulary is not None and vocabulary.casefold() in {
            v.casefold() for v in self.family
        }

    # -- classes and levels ----------------------------------------------

    def canonical_class(self, concept_class: str | None) -> str | None:
        """A class in the spec's own spelling (aliases resolved)."""
        if concept_class is None:
            return None
        return self.class_aliases.get(concept_class, concept_class)

    def spellings(self, classes: Iterable[str]) -> list[str]:
        """Every spelling of the given classes: themselves and their aliases."""
        wanted = set(classes)
        return sorted(
            wanted
            | {alias for alias, canonical in self.class_aliases.items() if canonical in wanted}
        )

    def refinement_rules_for(self, concept_class: str | None) -> list[RefinementRule]:
        """Refinement rules applicable when an anchor concept has ``concept_class``."""
        canonical = self.canonical_class(concept_class)
        if canonical is None:
            return []
        matched: list[RefinementRule] = []
        for rule in self.refinement:
            allowed = {self.canonical_class(item) or item for item in rule.from_classes}
            if canonical in allowed:
                matched.append(rule)
        return matched

    def granularity(self, name: str | None) -> GranularitySpec | None:
        if not name:
            return None
        exact = self.granularities.get(name)
        if exact is not None:
            return exact
        folded = name.casefold()
        for key, spec in self.granularities.items():
            if key.casefold() == folded:
                return spec
        return None

    def has_granularity(self, name: str | None) -> bool:
        """Whether ``name`` is a valid requested level (``None`` means identity mapping)."""
        return name is None or self.granularity(name) is not None

    def max_hops_for(self, granularity: str | None) -> int:
        """Maximum relationship hops required to reach ``granularity``."""
        level = self.granularity(granularity)
        return level.max_hops if level is not None else 1

    def predicate(self, granularity: str | None) -> str | None:
        """The predicate walked to reach a granularity (its first ``via``)."""
        level = self.granularity(granularity)
        return level.via[0] if level is not None and level.via else None

    def terminal_classes(self, granularity: str | None) -> frozenset[str]:
        level = self.granularity(granularity)
        return frozenset(level.terminal) if level is not None else frozenset()

    def accepted_classes(self, granularity: str | None) -> frozenset[str]:
        level = self.granularity(granularity)
        if level is None:
            return frozenset()
        return frozenset(level.accepts or level.terminal)

    def is_terminal(self, concept_class: str | None, granularity: str | None) -> bool:
        return self.canonical_class(concept_class) in self.terminal_classes(granularity)

    def inferred_class(self, predicate: str) -> str | None:
        """The class a release-bound edge's target always has, if the spec says."""
        if predicate not in self.release_bound_predicates:
            return None
        return self.edge_target_classes.get(predicate)

    # -- semantic loss ---------------------------------------------------

    @cached_property
    def _patterns(self) -> dict[int, re.Pattern[str]]:
        return {
            index: re.compile(rule.pattern, re.IGNORECASE)
            for index, rule in enumerate(self.losses)
            if rule.pattern is not None
        }

    def expected_losses(
        self,
        source: CandidateConcept,
        *,
        selected_ids: Sequence[str],
        target_label: str | None,
        mention_text: str | None,
        context: dict[str, object],
    ) -> list[SemanticLoss]:
        """Losses of mapping ``source`` to the selected targets, in rule order.

        ``target_label`` is the first target's label; ``context`` holds the
        SAU's context and attribute values (``route``, ...).
        """
        from biosemalign.schemas.decision import SemanticLoss

        losses: list[SemanticLoss] = []
        emitted: set[SemanticLossKind] = set()
        concept_class = self.canonical_class(source.concept_class) or ""
        selected = set(selected_ids)
        for index, rule in enumerate(self.losses):
            if rule.kind in emitted:
                continue
            if rule.classes and concept_class not in rule.classes:
                continue
            if concept_class in rule.exclude_classes:
                continue
            value = self._apply(index, rule, source, selected, target_label, mention_text, context)
            if value is None:
                continue
            losses.append(
                SemanticLoss(kind=rule.kind, description=rule.description, source_value=value)
            )
            emitted.add(rule.kind)
        return losses

    def _apply(
        self,
        index: int,
        rule: LossRule,
        source: CandidateConcept,
        selected: set[str],
        target_label: str | None,
        mention_text: str | None,
        context: dict[str, object],
    ) -> str | None:
        if rule.rule == "pattern":
            pattern = self._patterns[index]
            found = pattern.search(source.preferred_label) or (
                pattern.search(mention_text) if mention_text else None
            )
            if found is None or (target_label is not None and pattern.search(target_label)):
                return None
            return found.group(0).strip()
        if rule.rule == "relationship_labels":
            non_llm = [
                s
                for s in source.relationships
                if s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
            ]
            labels = [
                s.object_label for s in non_llm if s.predicate == rule.predicate and s.object_label
            ]
            if rule.fallback_to_source:
                labels = [
                    s.object_label or s.object_id for s in non_llm if s.predicate == rule.predicate
                ]
                return ", ".join(labels) or source.preferred_label
            return ", ".join(labels) if labels else None
        if rule.rule in {"source_not_selected", "ancestor"}:
            return None if source.candidate_id in selected else source.preferred_label
        if rule.rule == "unselected_objects":
            objects = [
                s.object_label or s.object_id
                for s in source.relationships
                if s.authority is not Authority.LLM_INTERNAL_KNOWLEDGE
                and s.predicate == rule.predicate
                and s.object_id not in selected
            ]
            return ", ".join(objects) if objects else None
        value = context.get(rule.field or "")
        return str(value) if value else None

    def combination_loss(
        self, granularity: str | None, *, count: int, source_label: str
    ) -> SemanticLoss | None:
        """The loss of resolving to ``count`` required components, if the level says so."""
        from biosemalign.schemas.decision import SemanticLoss

        level = self.granularity(granularity)
        if level is None or level.multi != "components" or not level.combination_description:
            return None
        return SemanticLoss(
            kind=SemanticLossKind.COMBINATION_COMPONENT,
            description=level.combination_description.format(n=count),
            source_value=source_label,
        )
