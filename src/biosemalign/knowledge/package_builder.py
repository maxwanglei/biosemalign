"""Assemble the Knowledge Package handed to the adjudicator (§6.1, §11.4).

Two invariants are enforced here rather than trusted to earlier stages:

* No prohibited predicate enters the package. Enrichment already filters, but
  this is the boundary the adjudicator sees, and outcome leakage during PHPT
  signal discovery is not a failure worth risking on one filter (§6.2).
* Every statement carries an authority label, so the adjudicator can be told
  plainly that a project coding policy does not outrank the target
  terminology's own concept status (§6.8).
"""

from __future__ import annotations

from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    CodingPolicy,
    KnowledgePackage,
    KnowledgeStatement,
)
from biosemalign.schemas.provenance import ResourceProvenance

__all__ = ["build_knowledge_package", "derive_coding_policies"]

log = get_logger(__name__)


def derive_coding_policies(profile: TaskProfile) -> list[CodingPolicy]:
    """Turn profile configuration into policies the adjudicator can be shown.

    Derived rather than hard-coded so that a second project changes behaviour
    by editing YAML, not by forking the package (§14, Milestone 9).
    """
    task = profile.task
    policies: list[CodingPolicy] = [
        CodingPolicy(
            policy_id="granularity",
            description="Requested analytical granularity",
            rule=(
                f"Map to {task.target_terminology} at "
                f"{task.desired_granularity or 'the most precise available'} level for "
                f"{task.mapping_intent}. When the target concept is coarser than the "
                "source, classify the relationship as TARGET_BROADER_THAN_SOURCE and "
                "record every detail the target does not carry."
            ),
            applies_to=[task.operation],
        ),
        CodingPolicy(
            policy_id="candidate_constrained",
            description="Selection is limited to retrieved candidates",
            rule=(
                "Select only from the retrieved candidate list. Never introduce an "
                "identifier that does not appear in it. If no candidate is defensible, "
                "reject all of them."
            ),
            applies_to=[task.operation],
        ),
    ]

    if profile.validation.preserve_combination_ingredients:
        policies.append(
            CodingPolicy(
                policy_id="combination_integrity",
                description="Combination products keep every component",
                rule=(
                    "A combination product must map to every one of its ingredients. "
                    "Silently dropping a component misclassifies exposure."
                ),
                applies_to=[task.operation],
            )
        )

    if profile.validation.flag_salt_active_moiety_conflation:
        policies.append(
            CodingPolicy(
                policy_id="salt_active_moiety",
                description="Salt forms are not interchangeable",
                rule=(
                    "Do not treat one salt form as equivalent to another. Metoprolol "
                    "succinate and metoprolol tartrate share an ingredient but are "
                    "distinct precise ingredients with different formulations."
                ),
                applies_to=[task.operation],
            )
        )

    if profile.knowledge.prohibited_relationships:
        policies.append(
            CodingPolicy(
                policy_id="no_outcome_associations",
                description="Outcome associations are withheld from this task",
                rule=(
                    "Relationships asserting an association between a drug and the "
                    "outcome under study are deliberately excluded, because using them "
                    "to choose candidates would bias signal discovery toward the "
                    "hypothesis being tested."
                ),
                applies_to=[task.operation],
            )
        )

    return policies


def build_knowledge_package(
    *,
    sau_id: str,
    candidates: list[CandidateConcept],
    profile: TaskProfile,
    resource_provenance: list[ResourceProvenance],
) -> KnowledgePackage:
    """Build the bounded evidence bundle for one SAU."""
    ontology_guidance: list[KnowledgeStatement] = []
    blocked: list[str] = []

    for candidate in candidates:
        kept: list[KnowledgeStatement] = []
        for statement in candidate.relationships:
            if not profile.knowledge.permits(statement.predicate):
                blocked.append(statement.predicate)
                continue
            kept.append(statement)
        candidate.relationships = kept
        ontology_guidance.extend(kept)

    if blocked:
        log.warning(
            "knowledge_package.blocked_predicates",
            sau_id=sau_id,
            predicates=sorted(set(blocked)),
            note="prohibited by profile; excluded from the package",
        )

    return KnowledgePackage(
        sau_id=sau_id,
        candidates=candidates,
        ontology_guidance=ontology_guidance,
        coding_policies=derive_coding_policies(profile),
        resource_provenance=resource_provenance,
    )
