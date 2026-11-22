"""Phenotyping workflow and kernel finalizer (§5.4, §5.5 G2/G4/G5/G7/G8, WP6)."""

from __future__ import annotations

from collections.abc import Sequence

from biosemalign.enums import (
    ConceptStatus,
    RoutingStatus,
    Severity,
    ValidationStatus,
)
from biosemalign.exceptions import AdjudicationFailure, ConfigurationError, WorkflowViolation
from biosemalign.kernel.guards import isolated_view, mark_expansion_member_from_model
from biosemalign.kernel.tiers import operator_provenance
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.modules import OpContext, build
from biosemalign.modules.base import Expander, Selector
from biosemalign.modules.expand.hierarchy import HierarchyExpander
from biosemalign.modules.model import ModelService, capture_model_calls
from biosemalign.modules.select.hierarchy import HierarchySelector
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.coding import CodingArtifact
from biosemalign.schemas.decision import ValidationFinding, ValidationResult
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.phenotyping import (
    ConceptSet,
    ConceptSetArtifact,
    ConceptSetDecision,
    ConceptSetMember,
    ConceptSetProposal,
    ConceptSetReview,
    ConceptSetScope,
)
from biosemalign.schemas.provenance import (
    ModelCallRecord,
    ModelProvenance,
    Provenance,
    RunContext,
    TierProvenance,
    WorkflowProvenance,
)
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.settings import Settings
from biosemalign.targets.spec import TargetSpec
from biosemalign.util import stable_digest
from biosemalign.workflows.coding import CodingCascade, CodingWorkflow, coding_workflow_from_profile

__all__ = [
    "PhenotypingWorkflow",
    "apply_concept_set_review",
    "export_concept_set",
    "phenotyping_workflow_from_profile",
]


class PhenotypingWorkflow:
    """Turn one phenotype idea into an audited :class:`ConceptSetArtifact` awaiting approval."""

    def __init__(
        self,
        *,
        profile: TaskProfile,
        knowledge: KnowledgeRouter,
        coding_cascade: CodingCascade,
        expanders: Sequence[Expander],
        selector: Selector,
        model_services: Sequence[ModelService] = (),
    ) -> None:
        self.profile = profile
        self.knowledge = knowledge
        self.coding_cascade = coding_cascade
        self.expanders = tuple(expanders)
        self.selector = selector
        self._explicit_model_services = tuple(model_services)
        self.target_spec: TargetSpec = profile.target_spec()
        self.coding_runner = CodingWorkflow(
            profile=profile,
            cascade=coding_cascade,
            knowledge=knowledge,
            model_services=self.all_model_services,
        )
        self.workflow_provenance = self._build_workflow_provenance()

    @property
    def all_model_services(self) -> tuple[ModelService, ...]:
        """All distinct :class:`ModelService` instances referenced by phenotyping stages."""
        services: list[ModelService] = []
        seen: set[int] = set()

        def add(candidate: object) -> None:
            if isinstance(candidate, ModelService) and id(candidate) not in seen:
                seen.add(id(candidate))
                services.append(candidate)

        for svc in self._explicit_model_services:
            add(svc)
        for tier in self.coding_cascade.tiers:
            for op in [*tier.cleaners, *tier.matchers, *((tier.linker,) if tier.linker else ())]:
                add(getattr(op, "model_service", None))
        for expander in self.expanders:
            add(getattr(expander, "model_service", None))
        add(getattr(self.selector, "model_service", None))
        return tuple(services)

    def verify(self, settings: Settings) -> None:
        self.coding_cascade.verify(self.profile, settings)

    def _build_workflow_provenance(self) -> WorkflowProvenance:
        coding_tiers = self.coding_runner.workflow_provenance.tiers
        phenotype_ops = [
            *(operator_provenance("expand", op) for op in self.expanders),
            operator_provenance("select", self.selector),
        ]
        tiers: tuple[TierProvenance, ...] = (
            *coding_tiers,
            TierProvenance(name="phenotype_select", operators=tuple(phenotype_ops)),
        )
        payload: dict[str, object] = {
            "workflow": "phenotyping",
            "tiers": [tier.model_dump(mode="json") for tier in tiers],
            "target_spec": {
                "label": self.target_spec.label,
                "sha256": self.target_spec.digest,
            },
        }
        return WorkflowProvenance(
            workflow_id=stable_digest(payload, length=16),
            tiers=tiers,
            validators=(),
            target_spec=self.target_spec.label,
            target_spec_sha256=self.target_spec.digest,
        )

    def propose(self, sau: SemanticAlignmentUnit, run: RunContext) -> ConceptSetArtifact:
        """Run ``coding -> expand -> select``, validate G2/G4/G5/G7/G8, and route for review."""
        services = list(self.all_model_services)

        # 1. Code the idea to obtain anchor concepts.
        coding = self.coding_runner.code(sau.model_copy(deep=True), run)
        view = coding.sau
        anchors: list[CandidateConcept] = []
        for code in coding.codes:
            concept = self.knowledge.provider_for(code.vocabulary).lookup(code.concept_code)
            if concept is None:
                concept = CandidateConcept(
                    candidate_id=code.concept_code,
                    preferred_label=code.label,
                    vocabulary=code.vocabulary,
                    concept_class=code.concept_class,
                    status=code.status,
                    retrieval_evidence=list(code.retrieval_evidence),
                )
            else:
                evidence = (
                    list(code.retrieval_evidence)
                    if code.retrieval_evidence
                    else list(concept.retrieval_evidence)
                )
                concept = concept.model_copy(update={"retrieval_evidence": evidence})
            anchors.append(concept)

        extra_findings: list[ValidationFinding] = []
        model_prov: ModelProvenance | None = self.selector.provenance()

        with capture_model_calls(services) as stage_calls:
            # 2. Expand anchors while enforcing G2 (relationship policy) and G10 (model mark).
            expansion = self._expand(view, anchors)

            # 3. Select candidate members with isolated context (G3 / M5).
            select_sau = isolated_view(view, "select", self.profile)
            try:
                raw_proposal = self.selector.select(select_sau, expansion.model_copy(deep=True))
            except AdjudicationFailure:
                if bool(self.selector.uses_model):
                    extra_findings.append(
                        ValidationFinding(
                            code="MODEL_SELECTOR_UNAVAILABLE",
                            severity=Severity.WARNING,
                            message=(
                                "model-backed selector returned an unusable reply; empty proposal "
                                "routed for human review (M6)"
                            ),
                        )
                    )
                    raw_proposal = ConceptSetProposal(
                        sau_id=view.sau_id,
                        idea=view.mention.text,
                        target_terminology=view.task.target_terminology,
                        level=view.task.desired_granularity,
                        members=[],
                        rationale="model reply could not be used; reviewer must assemble the set",
                        requested_human_review=True,
                    )
                else:
                    raise
            if not isinstance(raw_proposal, ConceptSetProposal):
                raise WorkflowViolation(
                    f"selector {self.selector.op_id!r} returned {type(raw_proposal).__name__}, "
                    "not ConceptSetProposal"
                )

        model_coded_keys = {
            (code.vocabulary.casefold(), code.concept_code)
            for code in coding.codes
            if code.from_model_suggestion
        }
        if view.mention.retrieval_text_from_model or model_coded_keys:
            raw_proposal = raw_proposal.model_copy(
                update={
                    "members": [
                        (
                            member.model_copy(update={"from_model_suggestion": True})
                            if (
                                view.mention.retrieval_text_from_model
                                or (member.vocabulary.casefold(), member.concept_code)
                                in model_coded_keys
                            )
                            else member
                        )
                        for member in raw_proposal.members
                    ]
                }
            )

        combined_calls = [*coding.provenance.model_calls, *stage_calls]
        return self._finalize(
            sau=view,
            coding=coding,
            expansion=expansion,
            proposal=raw_proposal,
            extra_findings=extra_findings,
            run=run,
            model=model_prov,
            model_calls=combined_calls,
        )

    def _expand(self, sau: SemanticAlignmentUnit, anchors: list[CandidateConcept]) -> ExpansionSet:
        anchor_qcs = [
            QualifiedCode(
                vocabulary=c.vocabulary,
                concept_code=c.candidate_id,
                label=c.preferred_label,
            )
            for c in anchors
        ]
        if not self.expanders:
            return ExpansionSet(expander="none", anchors=anchor_qcs)

        limit = self.profile.limits.members
        merged: dict[tuple[str, str], ExpansionMember] = {}
        truncated = False
        names: list[str] = []
        anchors_from_model = any(
            ev.from_model_suggestion for anchor in anchors for ev in anchor.retrieval_evidence
        )

        for expander in self.expanders:
            names.append(expander.op_id)
            expander_uses_model = bool(expander.uses_model)
            expander_sau = (
                isolated_view(sau, "expand", self.profile)
                if expander_uses_model
                else sau.model_copy(deep=True)
            )
            try:
                exp = expander.expand(expander_sau, anchors)
            except AdjudicationFailure:
                if expander_uses_model:
                    exp = ExpansionSet(expander=expander.op_id, anchors=anchor_qcs)
                else:
                    raise
            if not isinstance(exp, ExpansionSet):
                raise WorkflowViolation(
                    f"expander {expander.op_id!r} returned {type(exp).__name__}, not ExpansionSet"
                )
            if exp.truncated:
                truncated = True
            for member in exp.members:
                # G2: remove any expanded member resting on a prohibited relationship.
                if not all(
                    (
                        step.predicate not in self.profile.knowledge.prohibited_relationships
                        if step.direction == "lexical"
                        else self.profile.knowledge.permits(step.predicate)
                    )
                    for step in member.path
                ):
                    continue
                clean_member = (
                    mark_expansion_member_from_model(member)
                    if (
                        expander_uses_model
                        or sau.mention.retrieval_text_from_model
                        or anchors_from_model
                    )
                    else member
                )
                key = (
                    clean_member.concept.vocabulary.casefold(),
                    clean_member.concept.candidate_id,
                )
                if key in merged:
                    continue
                if len(merged) >= limit:
                    truncated = True
                    break
                merged[key] = clean_member

        return ExpansionSet(
            expander="+".join(names),
            anchors=anchor_qcs,
            members=list(merged.values()),
            truncated=truncated,
        )

    def _finalize(
        self,
        *,
        sau: SemanticAlignmentUnit,
        coding: CodingArtifact,
        expansion: ExpansionSet,
        proposal: ConceptSetProposal,
        extra_findings: list[ValidationFinding],
        run: RunContext,
        model: ModelProvenance | None,
        model_calls: Sequence[ModelCallRecord],
    ) -> ConceptSetArtifact:
        findings = list(extra_findings)
        expansion_keys = {(a.vocabulary.casefold(), a.concept_code) for a in expansion.anchors} | {
            (m.concept.vocabulary.casefold(), m.concept.candidate_id) for m in expansion.members
        }

        seen_members: set[tuple[str, str]] = set()
        for member in proposal.members:
            key = (member.vocabulary.casefold(), member.concept_code)
            if key in seen_members:
                findings.append(
                    ValidationFinding(
                        code="DUPLICATE_CONCEPT_SET_MEMBER",
                        severity=Severity.ERROR,
                        concept_id=member.concept_code,
                        message=(
                            f"duplicate member {member.vocabulary}:{member.concept_code} in "
                            "concept set proposal"
                        ),
                    )
                )
                continue
            seen_members.add(key)

            # G4: every member in a set must be backed by the expansion.
            if key not in expansion_keys:
                findings.append(
                    ValidationFinding(
                        code="MEMBER_OUTSIDE_EXPANSION",
                        severity=Severity.ERROR,
                        concept_id=member.concept_code,
                        message=(
                            f"proposed member {member.vocabulary}:{member.concept_code} is "
                            "outside the expansion set (G4)"
                        ),
                    )
                )

            # G5: mandatory terminology existence & lifecycle checks.
            looked_up = self.knowledge.provider_for(member.vocabulary).lookup(member.concept_code)
            if looked_up is None:
                findings.append(
                    ValidationFinding(
                        code="CONCEPT_NOT_FOUND",
                        severity=Severity.ERROR,
                        concept_id=member.concept_code,
                        message=(
                            f"member {member.vocabulary}:{member.concept_code} was not found in "
                            "terminology"
                        ),
                    )
                )
            elif (
                self.profile.target.require_active_concept
                and looked_up.status is not ConceptStatus.ACTIVE
            ):
                findings.append(
                    ValidationFinding(
                        code="INACTIVE_CONCEPT_IN_SET",
                        severity=Severity.WARNING,
                        concept_id=member.concept_code,
                        message=(
                            f"member {member.vocabulary}:{member.concept_code} has status "
                            f"{looked_up.status.value}"
                        ),
                    )
                )

            # M2: surface model/hierarchy disagreement for reviewer.
            if member.model_disagreement:
                hier = (
                    member.hierarchy_relation.value
                    if member.hierarchy_relation is not None
                    else "unknown"
                )
                findings.append(
                    ValidationFinding(
                        code="MODEL_HIERARCHY_DISAGREEMENT",
                        severity=Severity.WARNING,
                        concept_id=member.concept_code,
                        message=(
                            f"model stated {member.relation.value} for "
                            f"{member.vocabulary}:{member.concept_code} while hierarchy stated "
                            f"{hier}"
                        ),
                    )
                )

            # G10 / M4: warn on model-suggested members.
            if member.from_model_suggestion:
                findings.append(
                    ValidationFinding(
                        code="MODEL_SUGGESTED_MEMBER",
                        severity=Severity.WARNING,
                        concept_id=member.concept_code,
                        message=(
                            f"member {member.vocabulary}:{member.concept_code} came from a "
                            "model suggestion"
                        ),
                    )
                )

        has_error = any(f.severity is Severity.ERROR for f in findings)
        if not has_error and not any(m.included for m in proposal.members):
            findings.append(
                ValidationFinding(
                    code="EMPTY_CONCEPT_SET",
                    severity=Severity.WARNING,
                    message="proposed concept set contains no included members",
                )
            )

        status = ValidationStatus.REJECTED if has_error else ValidationStatus.HUMAN_REVIEW_REQUIRED
        validation = ValidationResult(
            status=status,
            findings=findings,
            rules_version="phenotyping-0.1.0",
        )
        decision_members = [] if has_error else [m.model_copy(deep=True) for m in proposal.members]
        routing_reasons = ["concept set proposals always require human approval before export (G8)"]
        for finding in findings:
            if finding.severity in {Severity.ERROR, Severity.WARNING}:
                routing_reasons.append(f"{finding.code}: {finding.message}")

        provenance = self._provenance(run, model=model, model_calls=model_calls)
        decision = ConceptSetDecision(
            sau_id=sau.sau_id,
            idea=proposal.idea,
            target_terminology=proposal.target_terminology,
            level=proposal.level,
            members=decision_members,
            validation=validation,
            routing_status=RoutingStatus.HUMAN_REVIEW,
            routing_reasons=routing_reasons,
            provenance=provenance,
        )
        artifact_id = ConceptSetArtifact.compute_artifact_id(
            run_id=run.run_id,
            sau=sau,
            coding=coding,
            expansion=expansion,
            proposal=proposal,
            validation=validation,
            decision=decision,
        )
        return ConceptSetArtifact(
            artifact_id=artifact_id,
            run_id=run.run_id,
            mapping_key=sau.mapping_key,
            sau=sau,
            coding=coding,
            expansion=expansion,
            proposal=proposal,
            validation=validation,
            decision=decision,
        )

    def _provenance(
        self,
        run: RunContext,
        *,
        model: ModelProvenance | None,
        model_calls: Sequence[ModelCallRecord],
    ) -> Provenance:
        matcher_versions = "+".join(
            dict.fromkeys(op.version for tier in self.coding_cascade.tiers for op in tier.matchers)
        )
        run.retriever_version = matcher_versions or None
        run.validation_rules_version = "phenotyping-0.1.0"
        run.routing_policy_version = "phenotyping-0.1.0"
        run.model = model
        run.workflow = self.workflow_provenance
        frozen = run.freeze()
        frozen.resources = self.knowledge.resource_provenance()
        frozen.resolution = None
        frozen.memo = None
        frozen.model_calls = list(model_calls)
        return frozen


def phenotyping_workflow_from_profile(context: OpContext) -> PhenotypingWorkflow:
    """Assemble a :class:`PhenotypingWorkflow` from a Schema 2 profile."""
    if context.knowledge is None:
        raise ConfigurationError("phenotyping workflow needs the pipeline's knowledge router")
    coding_cascade = coding_workflow_from_profile(context)
    expanders: list[Expander] = []
    selectors: list[Selector] = []
    if context.profile.tiers:
        for tier in context.profile.tiers:
            for exp_cfg in tier.expand_ops:
                expanders.append(build("expand", exp_cfg.op, context, exp_cfg.options))
            if tier.select is not None:
                selectors.append(build("select", tier.select.op, context, tier.select.options))
    else:
        expanders.append(HierarchyExpander(context.knowledge, context.profile))
    selector: Selector = (
        selectors[-1] if selectors else HierarchySelector(context.knowledge, context.profile)
    )
    return PhenotypingWorkflow(
        profile=context.profile,
        knowledge=context.knowledge,
        coding_cascade=coding_cascade,
        expanders=tuple(expanders),
        selector=selector,
        model_services=(context.model_service,) if context.model_service is not None else (),
    )


def apply_concept_set_review(
    artifact: ConceptSetArtifact,
    review: ConceptSetReview,
    *,
    knowledge: KnowledgeRouter,
) -> ConceptSetArtifact:
    """Validate and apply human-reviewer actions to a proposed :class:`ConceptSetArtifact`."""
    verified = ConceptSetArtifact.model_validate(artifact.model_dump(mode="json"))
    if review.artifact_id != verified.artifact_id:
        raise ValueError(
            f"review targets artifact {review.artifact_id!r}, not {verified.artifact_id!r}"
        )

    working: dict[tuple[str, str], ConceptSetMember] = {
        (m.vocabulary.casefold(), m.concept_code): m.model_copy(deep=True)
        for m in verified.decision.members
    }

    for action in review.actions:
        key = (action.vocabulary.casefold(), action.concept_code)
        if action.action == "exclude":
            if key not in working:
                raise ValueError(
                    f"cannot exclude {action.vocabulary}:{action.concept_code}; it is absent from "
                    "the decision members"
                )
            member = working[key]
            working[key] = member.model_copy(
                update={
                    "included": False,
                    "reason": action.reason or member.reason,
                }
            )
        elif action.action == "include":
            if key not in working:
                raise ValueError(
                    f"cannot include {action.vocabulary}:{action.concept_code} via action='include'; "
                    "use action='add' for codes outside the proposed set"
                )
            member = working[key]
            updates: dict[str, object] = {"included": True}
            if action.relation is not None:
                updates["relation"] = action.relation
            if action.scope is not None:
                updates["scope"] = action.scope
            if action.reason is not None:
                updates["reason"] = action.reason
            working[key] = member.model_copy(update=updates)
        elif action.action == "add":
            concept = knowledge.provider_for(action.vocabulary).lookup(action.concept_code)
            if concept is None:
                raise ValueError(
                    f"reviewer-added concept {action.vocabulary}:{action.concept_code} does not "
                    "exist in the terminology"
                )
            if action.relation is None or action.scope is None:
                raise ValueError(
                    f"reviewer add action for {action.vocabulary}:{action.concept_code} requires "
                    "both relation and scope"
                )
            working[key] = ConceptSetMember(
                vocabulary=concept.vocabulary,
                concept_code=concept.candidate_id,
                label=concept.preferred_label,
                concept_class=concept.concept_class,
                status=concept.status,
                included=True,
                relation=action.relation,
                scope=action.scope,
                reason=action.reason or f"Added by reviewer {review.reviewer}.",
                anchor=None,
                hops=0,
                path=[],
                via_operator="human_review",
                added_by_reviewer=True,
            )

    approved_members = [m for m in working.values() if m.included] if review.approved else []
    payload = verified.model_dump(mode="json")
    payload["review"] = review.model_dump(mode="json")
    payload["approved_members"] = [m.model_dump(mode="json") for m in approved_members]
    return ConceptSetArtifact.model_validate(payload)


def export_concept_set(
    artifact: ConceptSetArtifact,
    *,
    name: str,
    version: str = "1.0.0",
    scope: ConceptSetScope | None = None,
) -> ConceptSet:
    """Export an approved :class:`ConceptSet` (G8: refuses unapproved sets)."""
    verified = ConceptSetArtifact.model_validate(artifact.model_dump(mode="json"))
    if verified.review is None or not verified.review.approved:
        raise WorkflowViolation(
            "concept set cannot be exported without a recorded human approval (G8)"
        )

    members = [
        member.model_copy(deep=True)
        for member in verified.approved_members
        if member.included and (scope != "narrow" or member.scope == "narrow")
    ]
    target_vocab = verified.sau.task.target_terminology
    terminology_release = next(
        (
            resource.terminology_release
            for resource in verified.decision.provenance.resources
            if resource.resource_name.casefold() == target_vocab.casefold()
        ),
        None,
    )
    artifact_digest = verified.compute_digest()
    concept_set_id = ConceptSet.compute_concept_set_id(
        name=name,
        version=version,
        terminology=target_vocab,
        terminology_release=terminology_release,
        level=verified.sau.task.desired_granularity,
        artifact_id=verified.artifact_id,
        artifact_digest=artifact_digest,
        review_id=verified.review.review_id,
        approved_by=verified.review.reviewer,
        approved_at=verified.review.reviewed_at,
        members=members,
    )
    return ConceptSet(
        concept_set_id=concept_set_id,
        name=name,
        version=version,
        terminology=target_vocab,
        terminology_release=terminology_release,
        level=verified.sau.task.desired_granularity,
        artifact_id=verified.artifact_id,
        artifact_digest=artifact_digest,
        review_id=verified.review.review_id,
        approved_by=verified.review.reviewer,
        approved_at=verified.review.reviewed_at,
        members=members,
    )
