"""Routing policy (§9.2).

Transparent prespecified rules, not a learned calibration model. The initial
objective is precision of automatically accepted mappings rather than maximum
automatic coverage: a wrong auto-accepted mapping silently corrupts a
disproportionality analysis, while an unnecessary review costs a reviewer a
minute.

A learned model trained on AL-MedLit adjudications can replace this later; it
needs adjudications to exist first.
"""

from __future__ import annotations

from collections.abc import Callable

from biosemalign.enums import (
    DecisionType,
    RelationshipType,
    RetrievalChannel,
    RoutingStatus,
    ValidationStatus,
)
from biosemalign.plugins import ensure_plugins_loaded
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, RoutingFeatures, ValidationResult
from biosemalign.schemas.knowledge import KnowledgePackage

__all__ = [
    "ROUTING_POLICY_VERSION",
    "RoutingRuleHook",
    "register_routing_rule_hook",
    "route_decision",
    "unregister_routing_rule_hook",
]

RoutingRuleHook = Callable[
    [AlignmentProposal, ValidationResult, RoutingFeatures, KnowledgePackage, TaskProfile],
    list[str],
]
_ROUTING_RULE_HOOKS: dict[str, RoutingRuleHook] = {}


def register_routing_rule_hook(name: str, hook: RoutingRuleHook) -> None:
    """Register domain-specific routing review triggers."""
    _ROUTING_RULE_HOOKS[name] = hook


def unregister_routing_rule_hook(name: str) -> None:
    """Remove a registered domain routing hook."""
    _ROUTING_RULE_HOOKS.pop(name, None)


ROUTING_POLICY_VERSION = "0.3.0"

_EXACT_CHANNELS = frozenset({RetrievalChannel.EXACT_LOOKUP, RetrievalChannel.NORMALIZED_LOOKUP})

_BASE_AUTO_ACCEPTABLE_RELATIONS = frozenset(
    {RelationshipType.EXACT, RelationshipType.TARGET_BROADER_THAN_SOURCE}
)


def _auto_acceptable_relations(profile: TaskProfile) -> frozenset[RelationshipType]:
    """Relationship classes eligible for automatic acceptance under this profile.

    A combination product whose every ingredient survived validation is no less
    trustworthy than a single-ingredient broader mapping, so the profile flag
    decides rather than a hard-coded exclusion — otherwise
    ``combination_products_require_review: false`` would silently do nothing.
    """
    if profile.routing.combination_products_require_review:
        return _BASE_AUTO_ACCEPTABLE_RELATIONS
    return _BASE_AUTO_ACCEPTABLE_RELATIONS | {RelationshipType.MULTIPLE_COMPONENTS_REQUIRED}


def route_decision(
    *,
    proposal: AlignmentProposal,
    validation: ValidationResult,
    features: RoutingFeatures,
    package: KnowledgePackage,
    profile: TaskProfile,
) -> tuple[RoutingStatus, list[str]]:
    """Decide where this decision goes, and say why."""
    reasons: list[str] = []
    routing = profile.routing

    # --- rejected proposals always need a person ---
    if validation.status is ValidationStatus.REJECTED:
        return RoutingStatus.HUMAN_REVIEW, [
            f"validation rejected the proposal: {f.code}" for f in validation.errors
        ]

    # --- abstain only after integrity validation has accepted the abstention ---
    if proposal.decision in {DecisionType.NO_VALID_MAPPING, DecisionType.ABSTAINED}:
        if package.candidates:
            reason = "retrieved candidates were considered but none was accepted"
        else:
            reason = "no candidate concept was retrieved on any channel"
        if proposal.rationale:
            reason = f"{reason}: {proposal.rationale}"
        # An abstention the adjudicator flagged for review, or one carrying
        # validator warnings, has not been accepted; a person must see it.
        reasons.append(reason)
        if proposal.requested_human_review:
            reasons.append("the adjudicator requested review")
        reasons.extend(f"validator warning: {f.code}" for f in validation.warnings)
        if len(reasons) > 1:
            return RoutingStatus.HUMAN_REVIEW, reasons
        return RoutingStatus.ABSTAIN, reasons

    if proposal.decision is DecisionType.INSUFFICIENT_CONTEXT:
        return RoutingStatus.HUMAN_REVIEW, [
            "insufficient context to map; a reviewer may be able to supply it"
        ]

    if proposal.requested_human_review:
        reasons.append("the adjudicator requested review")

    # --- profile-driven review triggers ---
    if not features.concept_is_active and (
        profile.target.require_active_concept or routing.historical_candidate_requires_review
    ):
        reasons.append(
            "a final concept is not current or was not verified active in the target terminology"
        )

    if features.multiple_concepts_required and routing.combination_products_require_review:
        reasons.append("combination products are configured for review")

    # Two target concepts that each carry the mention as an exact name are a
    # homonym, not a match; ranking would only pick one by tie-break.
    target = profile.target.terminology.casefold()
    exact_targets = {
        candidate.candidate_id
        for candidate in package.candidates
        if candidate.vocabulary.casefold() == target and candidate.channels & _EXACT_CHANNELS
    }
    if len(exact_targets) > 1:
        reasons.append(
            f"the mention is an exact name of {len(exact_targets)} "
            f"{profile.target.terminology} concepts: {', '.join(sorted(exact_targets))}"
        )

    ensure_plugins_loaded()
    for hook in _ROUTING_RULE_HOOKS.values():
        reasons.extend(hook(proposal, validation, features, package, profile))

    # A match resting only on fuzzy scoring needs lexical corroboration before
    # it is accepted without a person looking at it.
    similarity = features.lexical_similarity
    if (
        similarity is not None
        and similarity < routing.minimum_lexical_similarity
        and not (features.exact_terminology_match or features.official_synonym_match)
    ):
        reasons.append(
            f"lexical similarity {similarity:.2f} to the retrieved concept is below the "
            f"floor {routing.minimum_lexical_similarity:.2f} for automatic acceptance"
        )

    if features.relationship_type not in _auto_acceptable_relations(profile):
        reasons.append(
            f"relationship {features.relationship_type.value} is not eligible for "
            "automatic acceptance"
        )

    if validation.warnings:
        reasons.extend(f"validator warning: {f.code}" for f in validation.warnings)

    if routing.minimum_auto_accept_score is not None:
        source_id = (
            proposal.source_level_concept.concept_id
            if proposal.source_level_concept is not None
            else None
        )
        source = (
            package.candidate_by_qualified_id(
                profile.target.source_vocabulary(
                    proposal.source_level_concept.vocabulary
                    if proposal.source_level_concept is not None
                    else None
                ),
                source_id,
            )
            if source_id is not None
            else None
        )
        if source is None:
            source = next(
                (
                    candidate
                    for selected in proposal.selected_concepts
                    if (
                        candidate := package.candidate_by_qualified_id(
                            profile.target.terminology, selected.concept_id
                        )
                    )
                    is not None
                ),
                None,
            )
        score = source.best_score if source is not None else 0.0
        if score < routing.minimum_auto_accept_score:
            reasons.append(
                f"selected source candidate score {score:.1f} is below the configured floor "
                f"{routing.minimum_auto_accept_score:.1f}"
            )

    if (
        not routing.exact_synonym_auto_accept
        and features.relationship_type is RelationshipType.EXACT
    ):
        reasons.append("profile disables automatic acceptance of all EXACT mappings")

    # G10 / M4: a candidate reached solely via model suggestion (written query text,
    # model-backed matcher, or model-backed expander) is never auto-accepted.
    selected_source = None
    if proposal.source_level_concept is not None:
        selected_source = package.candidate_by_qualified_id(
            profile.target.source_vocabulary(proposal.source_level_concept.vocabulary),
            proposal.source_level_concept.concept_id,
        )
    if selected_source is None:
        for selected in proposal.selected_concepts:
            selected_source = package.candidate_by_qualified_id(
                profile.target.terminology, selected.concept_id
            )
            if selected_source is not None:
                break
    if (
        selected_source is not None
        and selected_source.retrieval_evidence
        and all(evidence.from_model_suggestion for evidence in selected_source.retrieval_evidence)
    ):
        reasons.append(
            "selected candidate was reached through a model suggestion and requires human review"
        )

    if reasons:
        return RoutingStatus.HUMAN_REVIEW, reasons

    # --- automatic acceptance ---
    if routing.exact_synonym_auto_accept and (
        features.exact_terminology_match or features.official_synonym_match
    ):
        return RoutingStatus.AUTO_ACCEPT, [
            "exact terminology or official synonym match on an active concept"
        ]

    if features.relationship_type is RelationshipType.TARGET_BROADER_THAN_SOURCE:
        # The expected, correct outcome for ingredient-level analysis — provided
        # the loss it entails was actually reported.
        if proposal.semantic_loss:
            return RoutingStatus.AUTO_ACCEPT, [
                "resolved to the requested granularity with semantic loss fully reported"
            ]
        return RoutingStatus.HUMAN_REVIEW, ["broader mapping with no reported semantic loss"]

    if features.relationship_type is RelationshipType.MULTIPLE_COMPONENTS_REQUIRED:
        return RoutingStatus.AUTO_ACCEPT, [
            "combination product mapped to every component, integrity checks passed"
        ]

    if features.relationship_type is RelationshipType.EXACT:
        return RoutingStatus.AUTO_ACCEPT, ["exact mapping with no validator findings"]

    return RoutingStatus.HUMAN_REVIEW, ["no automatic-acceptance rule applied"]
