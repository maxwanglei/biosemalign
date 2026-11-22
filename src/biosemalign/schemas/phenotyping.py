"""Phenotyping workflow contracts: concept set proposal, review, and approved export (§5.4, WP6)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from biosemalign.enums import ConceptStatus, RelationshipType, RoutingStatus, ValidationStatus
from biosemalign.schemas.base import (
    BioSemAlignModel,
    VersionedModel,
    canonical_digest,
    canonical_id,
)
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.coding import CodingArtifact
from biosemalign.schemas.decision import ValidationResult
from biosemalign.schemas.expansion import ExpansionSet, ExpansionStep
from biosemalign.schemas.provenance import Provenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = [
    "ConceptSet",
    "ConceptSetArtifact",
    "ConceptSetDecision",
    "ConceptSetMember",
    "ConceptSetProposal",
    "ConceptSetReview",
    "ConceptSetScope",
    "MemberReviewAction",
]

ConceptSetScope = Literal["narrow", "broad"]


class ConceptSetMember(BioSemAlignModel):
    """One candidate or approved member of a phenotype concept set."""

    vocabulary: str = Field(min_length=1)
    concept_code: str = Field(min_length=1)
    label: str
    concept_class: str | None = None
    status: ConceptStatus = ConceptStatus.UNKNOWN
    included: bool = True
    relation: RelationshipType
    scope: ConceptSetScope
    reason: str = Field(min_length=1)
    anchor: QualifiedCode | None = None
    hops: int = Field(default=0, ge=0)
    path: list[ExpansionStep] = Field(default_factory=list)
    via_operator: str | None = None
    from_model_suggestion: bool = False
    added_by_reviewer: bool = False
    hierarchy_relation: RelationshipType | None = None
    model_disagreement: bool = False
    model_rationale: str | None = None


class ConceptSetProposal(VersionedModel):
    """A selector's proposed members for a phenotype idea, awaiting validation and review."""

    sau_id: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    target_terminology: str = Field(min_length=1)
    level: str | None = None
    members: list[ConceptSetMember] = Field(default_factory=list)
    rationale: str | None = None
    requested_human_review: bool = True


class ConceptSetDecision(VersionedModel):
    """Validated and routed concept-set decision emitted by the phenotyping finalizer."""

    sau_id: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    target_terminology: str = Field(min_length=1)
    level: str | None = None
    members: list[ConceptSetMember] = Field(default_factory=list)
    validation: ValidationResult
    routing_status: RoutingStatus = RoutingStatus.HUMAN_REVIEW
    routing_reasons: list[str] = Field(default_factory=list)
    provenance: Provenance


class MemberReviewAction(BioSemAlignModel):
    """One human-reviewer action on a proposed or additional terminology code."""

    action: Literal["include", "exclude", "add"]
    vocabulary: str = Field(min_length=1)
    concept_code: str = Field(min_length=1)
    relation: RelationshipType | None = None
    scope: ConceptSetScope | None = None
    reason: str | None = None


class ConceptSetReview(VersionedModel):
    """Recorded human review approval/edits for a proposed :class:`ConceptSetArtifact` (G8)."""

    review_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    approved: bool
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actions: list[MemberReviewAction] = Field(default_factory=list)
    notes: str | None = None


class ConceptSet(VersionedModel):
    """Immutable exported concept set produced only after recorded human approval (§5.4, G8)."""

    concept_set_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(default="1.0.0", min_length=1)
    terminology: str = Field(min_length=1)
    terminology_release: str | None = None
    level: str | None = None
    artifact_id: str = Field(min_length=1)
    artifact_digest: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    members: list[ConceptSetMember] = Field(default_factory=list)

    @staticmethod
    def compute_concept_set_id(
        *,
        name: str,
        version: str,
        terminology: str,
        terminology_release: str | None,
        level: str | None,
        artifact_id: str,
        artifact_digest: str,
        review_id: str,
        approved_by: str,
        approved_at: datetime,
        members: list[ConceptSetMember],
    ) -> str:
        """Compute deterministic content identifier for an exported :class:`ConceptSet`."""
        payload = {
            "name": name,
            "version": version,
            "terminology": terminology,
            "terminology_release": terminology_release,
            "level": level,
            "artifact_id": artifact_id,
            "artifact_digest": artifact_digest,
            "review_id": review_id,
            "approved_by": approved_by,
            "approved_at": approved_at.isoformat(),
            "members": [member.model_dump(mode="json") for member in members],
        }
        return canonical_id("CSET", payload, length=32)

    @model_validator(mode="after")
    def _validate_concept_set(self) -> ConceptSet:
        seen: set[tuple[str, str]] = set()
        for member in self.members:
            if not member.included:
                raise ValueError("exported ConceptSet cannot include excluded members")
            key = (member.vocabulary.casefold(), member.concept_code)
            if key in seen:
                raise ValueError(
                    f"duplicate member in ConceptSet: {member.vocabulary}:{member.concept_code}"
                )
            seen.add(key)
        expected_id = ConceptSet.compute_concept_set_id(
            name=self.name,
            version=self.version,
            terminology=self.terminology,
            terminology_release=self.terminology_release,
            level=self.level,
            artifact_id=self.artifact_id,
            artifact_digest=self.artifact_digest,
            review_id=self.review_id,
            approved_by=self.approved_by,
            approved_at=self.approved_at,
            members=self.members,
        )
        if self.concept_set_id != expected_id:
            raise ValueError("concept_set_id does not match its canonical payload")
        return self


class ConceptSetArtifact(VersionedModel):
    """Complete auditable phenotyping artifact: coding anchors, expansion, proposal, decision, review."""

    artifact_id: str = Field(description="Content-addressed identifier of the proposed artifact.")
    run_id: str = Field(min_length=1)
    mapping_key: str = Field(min_length=1)
    sau: SemanticAlignmentUnit
    coding: CodingArtifact
    expansion: ExpansionSet
    proposal: ConceptSetProposal
    validation: ValidationResult
    decision: ConceptSetDecision
    review: ConceptSetReview | None = None
    approved_members: list[ConceptSetMember] = Field(default_factory=list)

    def compute_digest(self) -> str:
        """Compute full content digest covering proposal, decision, review, and approved members."""
        payload: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "sau": self.sau.model_dump(mode="json"),
            "coding": self.coding.model_dump(mode="json"),
            "expansion": self.expansion.model_dump(mode="json"),
            "proposal": self.proposal.model_dump(mode="json"),
            "validation": self.validation.model_dump(mode="json"),
            "decision": self.decision.model_dump(mode="json"),
            "review": self.review.model_dump(mode="json") if self.review is not None else None,
            "approved_members": [
                member.model_dump(mode="json") for member in self.approved_members
            ],
        }
        return canonical_digest(payload, length=64)

    @staticmethod
    def compute_artifact_id(
        *,
        run_id: str,
        sau: SemanticAlignmentUnit,
        coding: CodingArtifact,
        expansion: ExpansionSet,
        proposal: ConceptSetProposal,
        validation: ValidationResult,
        decision: ConceptSetDecision,
    ) -> str:
        """Compute deterministic content-addressed identifier for a :class:`ConceptSetArtifact`."""
        payload = {
            "run_id": run_id,
            "sau": sau.model_dump(mode="json"),
            "coding": coding.model_dump(mode="json"),
            "expansion": expansion.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }
        return canonical_id("SET", payload, length=32)

    @model_validator(mode="after")
    def _validate_concept_set_artifact(self) -> ConceptSetArtifact:
        if self.mapping_key != self.sau.mapping_key or self.mapping_key != self.coding.mapping_key:
            raise ValueError("concept set artifact mapping_key does not match its SAU and coding")
        if (
            self.run_id != self.sau.provenance.run_id
            or self.run_id != self.coding.run_id
            or self.run_id != self.decision.provenance.run_id
        ):
            raise ValueError("concept set artifact run_id does not match provenance")
        if self.proposal.sau_id != self.sau.sau_id or self.decision.sau_id != self.sau.sau_id:
            raise ValueError("concept set proposal/decision sau_id does not match the SAU")
        if self.validation != self.decision.validation:
            raise ValueError("concept set artifact validation does not match decision validation")

        for field_name in (
            "package_version",
            "profile_name",
            "profile_version",
            "source_adapter",
            "source_adapter_version",
            "identifier_hmac_key_fingerprint",
        ):
            if getattr(self.sau.provenance, field_name) != getattr(
                self.decision.provenance, field_name
            ):
                raise ValueError(
                    f"concept set decision provenance {field_name} does not match SAU provenance"
                )

        # G8: a proposed code set is never automatically accepted.
        if self.decision.routing_status is not RoutingStatus.HUMAN_REVIEW:
            raise ValueError("concept set proposals must always route to HUMAN_REVIEW (G8)")

        # G4: every member in proposal and decision must come from the anchors or expansion.
        expansion_keys = {
            (anchor.vocabulary.casefold(), anchor.concept_code) for anchor in self.expansion.anchors
        } | {
            (member.concept.vocabulary.casefold(), member.concept.candidate_id)
            for member in self.expansion.members
        }

        def check_unique(members: list[ConceptSetMember], label: str) -> list[tuple[str, str]]:
            keys: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for member in members:
                key = (member.vocabulary.casefold(), member.concept_code)
                if key in seen:
                    raise ValueError(
                        f"duplicate {label} member {member.vocabulary}:{member.concept_code}"
                    )
                seen.add(key)
                keys.append(key)
            return keys

        decision_keys = check_unique(self.decision.members, "decision")
        for member in self.decision.members:
            if (member.vocabulary.casefold(), member.concept_code) not in expansion_keys:
                raise ValueError(
                    f"decision member {member.vocabulary}:{member.concept_code} is outside the "
                    "expansion set (G4)"
                )

        if self.validation.status is ValidationStatus.REJECTED:
            if self.decision.members:
                raise ValueError("rejected concept set proposal must have empty decision members")
        else:
            proposal_keys = check_unique(self.proposal.members, "proposal")
            for member in self.proposal.members:
                if (member.vocabulary.casefold(), member.concept_code) not in expansion_keys:
                    raise ValueError(
                        f"proposed member {member.vocabulary}:{member.concept_code} is outside the "
                        "expansion set (G4)"
                    )
            if set(decision_keys) != set(proposal_keys):
                raise ValueError("decision members do not match validated proposal members")

        # Review consistency (G8 + reviewer-added mark + action replay).
        if self.review is None:
            if self.approved_members:
                raise ValueError("approved_members must be empty when review is absent")
        else:
            if self.review.artifact_id != self.artifact_id:
                raise ValueError("concept set review artifact_id does not match artifact")
            if not self.review.approved and self.approved_members:
                raise ValueError("unapproved review cannot record approved_members")
            check_unique(self.approved_members, "approved")
            approved_by_key = {
                (member.vocabulary.casefold(), member.concept_code): member
                for member in self.approved_members
            }
            working: dict[tuple[str, str], ConceptSetMember] = {
                (member.vocabulary.casefold(), member.concept_code): member.model_copy(deep=True)
                for member in self.decision.members
            }
            for act in self.review.actions:
                key = (act.vocabulary.casefold(), act.concept_code)
                if act.action == "exclude":
                    if key not in working:
                        raise ValueError(
                            f"review exclude action targets {act.vocabulary}:{act.concept_code}, "
                            "which is absent from decision members"
                        )
                    current = working[key]
                    working[key] = current.model_copy(
                        update={
                            "included": False,
                            "reason": act.reason or current.reason,
                        }
                    )
                elif act.action == "include":
                    if key not in working:
                        raise ValueError(
                            f"review include action targets {act.vocabulary}:{act.concept_code}, "
                            "which is absent from decision members"
                        )
                    current = working[key]
                    updates: dict[str, object] = {"included": True}
                    if act.relation is not None:
                        updates["relation"] = act.relation
                    if act.scope is not None:
                        updates["scope"] = act.scope
                    if act.reason is not None:
                        updates["reason"] = act.reason
                    working[key] = current.model_copy(update=updates)
                elif act.action == "add":
                    if act.relation is None or act.scope is None:
                        raise ValueError(
                            f"reviewer add action for {act.vocabulary}:{act.concept_code} requires "
                            "both relation and scope"
                        )
                    seed = approved_by_key.get(key)
                    vocab = (
                        seed.vocabulary
                        if seed is not None
                        and seed.vocabulary.casefold() == act.vocabulary.casefold()
                        else act.vocabulary
                    )
                    working[key] = ConceptSetMember(
                        vocabulary=vocab,
                        concept_code=act.concept_code,
                        label=seed.label if seed is not None else act.concept_code,
                        concept_class=seed.concept_class if seed is not None else None,
                        status=seed.status if seed is not None else ConceptStatus.UNKNOWN,
                        included=True,
                        relation=act.relation,
                        scope=act.scope,
                        reason=act.reason or f"Added by reviewer {self.review.reviewer}.",
                        anchor=None,
                        hops=0,
                        path=[],
                        via_operator="human_review",
                        added_by_reviewer=True,
                    )
            expected_approved = (
                [member for member in working.values() if member.included]
                if self.review.approved
                else []
            )
            if self.approved_members != expected_approved:
                raise ValueError(
                    "approved_members do not match decision.members after applying review.actions"
                )
            added_actions = {
                (act.vocabulary.casefold(), act.concept_code)
                for act in self.review.actions
                if act.action == "add"
            }
            for member in self.approved_members:
                if not member.included:
                    raise ValueError("approved_members cannot contain excluded members")
                key = (member.vocabulary.casefold(), member.concept_code)
                if key not in expansion_keys and (
                    not member.added_by_reviewer or key not in added_actions
                ):
                    raise ValueError(
                        f"approved member {member.vocabulary}:{member.concept_code} is outside "
                        "the expansion without a recorded reviewer add action"
                    )

        # G7: self-verifying content hash.
        expected_id = ConceptSetArtifact.compute_artifact_id(
            run_id=self.run_id,
            sau=self.sau,
            coding=self.coding,
            expansion=self.expansion,
            proposal=self.proposal,
            validation=self.validation,
            decision=self.decision,
        )
        if self.artifact_id != expected_id:
            raise ValueError("concept set artifact_id does not match its canonical payload")
        return self
