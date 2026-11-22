"""Assemble the Knowledge Package handed to the resolver (§6.1, §11.4).

Two invariants are enforced here rather than trusted to earlier stages:

* No prohibited predicate enters the package. Enrichment already filters, but
  this is the boundary the resolver sees, and outcome leakage during PHPT
  signal discovery is not a failure worth risking on one filter (§6.2).
* Every statement carries an authority label, so the resolver can be told
  plainly that a project coding policy does not outrank the target
  terminology's own concept status (§6.8).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from biosemalign.logging import get_logger
from biosemalign.plugins import ensure_plugins_loaded, load_domain
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    CodingPolicy,
    KnowledgePackage,
    KnowledgeStatement,
)
from biosemalign.schemas.provenance import ResourceProvenance

__all__ = [
    "CheckPolicyHook",
    "build_knowledge_package",
    "derive_coding_policies",
    "enforce_relationship_policy",
    "register_check_policies",
    "unregister_check_policies",
]

log = get_logger(__name__)

CheckPolicyHook = Callable[[Mapping[str, Any], TaskProfile], list[CodingPolicy]]
_CHECK_POLICY_HOOKS: dict[str, CheckPolicyHook] = {}


def register_check_policies(op_id: str, hook: CheckPolicyHook) -> None:
    """Register domain coding-policy derivation for check operator ``op_id``."""
    _CHECK_POLICY_HOOKS[op_id] = hook


def unregister_check_policies(op_id: str) -> None:
    """Remove coding-policy derivation registered for ``op_id``."""
    _CHECK_POLICY_HOOKS.pop(op_id, None)


def derive_coding_policies(profile: TaskProfile) -> list[CodingPolicy]:
    """Turn profile configuration into policies the resolver can be shown.

    Derived rather than hard-coded so that a second project changes behaviour
    by editing YAML, not by forking the package (§14, Milestone 9).
    """
    ensure_plugins_loaded()
    if profile.domain:
        load_domain(profile.domain)

    task = profile.task
    target = profile.target
    policies: list[CodingPolicy] = [
        CodingPolicy(
            policy_id="granularity",
            description="Requested analytical granularity",
            rule=(
                f"Map to {target.terminology} at "
                f"{target.level or 'the most precise available'} level for "
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

    for check in profile.checks:
        hook = _CHECK_POLICY_HOOKS.get(check.op)
        if hook is not None:
            policies.extend(hook(check.options, profile))

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


def enforce_relationship_policy(
    package: KnowledgePackage, profile: TaskProfile
) -> KnowledgePackage:
    """Remove every statement the profile does not permit, wherever it appears.

    The kernel applies this to whatever a linker operator returns. Linking is
    replaceable, so the outcome-leakage guard cannot live only inside it; this
    second filter is what keeps a custom linker from reintroducing prohibited
    predicates (§6.2).
    """
    permits = profile.knowledge.permits
    blocked: set[str] = set()

    def keep(statements: list[KnowledgeStatement]) -> list[KnowledgeStatement]:
        kept = [statement for statement in statements if permits(statement.predicate)]
        blocked.update(s.predicate for s in statements if not permits(s.predicate))
        return kept

    candidates = [
        candidate.model_copy(update={"relationships": keep(candidate.relationships)})
        for candidate in package.candidates
    ]
    conflicts = [
        conflict.model_copy(update={"statements": keep(conflict.statements)})
        for conflict in package.conflicts
    ]
    filtered = package.model_copy(
        update={
            "candidates": candidates,
            "ontology_guidance": keep(package.ontology_guidance),
            "umls_guidance": keep(package.umls_guidance),
            "kg_guidance": keep(package.kg_guidance),
            "conflicts": conflicts,
        }
    )
    if not blocked:
        return package
    log.warning(
        "knowledge_package.blocked_predicates",
        sau_id=package.sau_id,
        predicates=sorted(blocked),
        note="prohibited by profile; removed by the kernel after linking",
    )
    return filtered
