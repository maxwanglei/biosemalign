"""Concordant cross-vocabulary resolver (`concordant`)."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from biosemalign.enums import ConcordanceState
from biosemalign.modules.base import PassToNext
from biosemalign.modules.resolve.deterministic import DeterministicResolver
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import KnowledgePackage, QualifiedCode
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["ConcordantResolver"]


class ConcordantResolver:
    """Resolve only when exact matches agree on one identity that reaches the target.

    Uses target-neutral pass reasons (``no_target_link``, ``non_target_code``).
    """

    op_id = "concordant"
    version = "0.3.0"
    uses_model = False
    requires_network = False

    REASONS = ("no_target_link", "non_target_code")

    def __init__(
        self,
        rules: DeterministicResolver,
        profile: TaskProfile,
        *,
        authority_sources: Collection[str] = (),
        minimum_mention_length: int = 5,
    ) -> None:
        self.no_link_reason, self.foreign_reason = self.REASONS
        self.rules = rules
        self.target = profile.target.terminology
        self.identity_vocabularies = [self.target] + [
            vocabulary
            for vocabulary in profile.target.source_terminologies
            if vocabulary.casefold() != self.target.casefold()
        ]
        self.authority_sources = frozenset(authority_sources)
        self.minimum_mention_length = minimum_mention_length

    @property
    def options(self) -> dict[str, Any]:
        return {
            "authority_sources": sorted(self.authority_sources),
            "minimum_mention_length": self.minimum_mention_length,
            "model": self.rules.provenance().model_name,
        }

    def resolve(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal | PassToNext:
        assessment = package.concordance
        if assessment is None:
            return PassToNext("no_concordance_assessment", "the linker did not assess concordance")
        if len(sau.mention.normalized_text.replace(" ", "")) < self.minimum_mention_length:
            return PassToNext(
                "short_mention",
                f"mentions shorter than {self.minimum_mention_length} characters are too "
                "ambiguous to resolve by name alone",
            )
        passes = {
            ConcordanceState.NO_EXACT_MATCH: "no_exact_match",
            ConcordanceState.CROSS_VOCAB_COLLISION: "lexical_collision",
            ConcordanceState.EQUIVALENCE_CONFLICT: "equivalence_conflict",
        }
        if assessment.state in passes:
            return PassToNext(passes[assessment.state], _describe_groups(package))
        group = assessment.groups[0]
        if assessment.state is ConcordanceState.SINGLE_SOURCE_EXACT and not (
            set(group.sources) & self.authority_sources
        ):
            return PassToNext(
                "single_source",
                f"only {', '.join(group.sources)} asserts this name, which is not an "
                "authority source",
            )
        member = self._identity(group.members)
        if isinstance(member, PassToNext):
            return member
        identity = package.candidate_by_qualified_id(member.vocabulary, member.concept_code)
        if identity is None:
            return PassToNext(
                "target_not_in_package",
                f"{member.vocabulary} {member.concept_code} fell outside the bounded package",
            )
        granularity = sau.task.desired_granularity
        targets = (
            [
                (t.vocabulary, t.concept_code)
                for t in package.links.targets_of(identity.vocabulary, identity.candidate_id)
            ]
            if package.links is not None
            else [
                (t.vocabulary, t.concept_id) for t in self.rules.targets_for(identity, granularity)
            ]
        )
        described = (
            f"{identity.vocabulary}:{identity.candidate_id} ({identity.concept_class or 'concept'}"
            f", {identity.preferred_label!r})"
        )
        if not targets:
            return PassToNext(
                self.no_link_reason,
                f"{described} has no path to a {self.target} {granularity or 'target'}",
            )
        foreign = [t for t in targets if t[0].casefold() != self.target.casefold()]
        if foreign:
            return PassToNext(
                self.foreign_reason,
                f"{described} reaches "
                + ", ".join(f"{vocabulary}:{code}" for vocabulary, code in foreign)
                + f", outside {self.target}",
            )
        return self.rules.project(sau, identity, links=package.links, refinement=package.refinement)

    def _identity(self, members: Sequence[QualifiedCode]) -> QualifiedCode | PassToNext:
        """The group's one concept from the most preferred identity vocabulary."""
        for vocabulary in self.identity_vocabularies:
            found = [m for m in members if m.vocabulary.casefold() == vocabulary.casefold()]
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                return PassToNext(
                    "no_target_concept",
                    f"the agreed entity has {len(found)} {vocabulary} concepts",
                )
        return PassToNext(
            "no_target_concept",
            f"the agreed entity has no {' or '.join(self.identity_vocabularies)} concept",
        )

    def provenance(self) -> ModelProvenance:
        return self.rules.provenance()


def _describe_groups(package: KnowledgePackage) -> str | None:
    if package.concordance is None or not package.concordance.groups:
        return None
    return " | ".join(
        ", ".join(f"{m.vocabulary}:{m.concept_code}" for m in group.members)
        for group in package.concordance.groups
    )
