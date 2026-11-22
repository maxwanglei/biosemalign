"""Public contract test kit for BioSemAlign operators, target specs, and domain packs (§6, WP7)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from biosemalign.enums import ConceptStatus, ReadinessStatus, RetrievalChannel, SourceType
from biosemalign.kernel.guards import isolated_view, require_non_destructive
from biosemalign.modules.base import (
    CandidateSet,
    Check,
    Cleaner,
    Expander,
    Linker,
    Matcher,
    Operator,
    PassToNext,
    Resolver,
    Selector,
)
from biosemalign.modules.link.targets import derive_link_set
from biosemalign.plugins import load_domain, loaded_domains, without_domain
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.decision import AlignmentProposal, ValidationFinding
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    CodingPolicy,
    KnowledgePackage,
    KnowledgeStatement,
    RetrievalEvidence,
)
from biosemalign.schemas.phenotyping import ConceptSetProposal
from biosemalign.schemas.provenance import ModelProvenance, Provenance
from biosemalign.schemas.request import MappingTask
from biosemalign.schemas.sau import (
    ContextEnvelope,
    ContextualAttributes,
    Mention,
    ReadinessAssessment,
    SemanticAlignmentUnit,
)
from biosemalign.schemas.source import SourceRecord
from biosemalign.targets.spec import TargetSpec
from biosemalign.util import normalize_text, stable_id

__all__ = [
    "check_check_contract",
    "check_cleaner_contract",
    "check_domain_pack_contract",
    "check_expander_contract",
    "check_linker_contract",
    "check_matcher_contract",
    "check_operator_metadata",
    "check_resolver_contract",
    "check_selector_contract",
    "check_target_spec_contract",
    "make_contract_candidate",
    "make_contract_package",
    "make_contract_sau",
]


def make_contract_sau(
    text: str = "toyprolol 50 mg",
    *,
    target_terminology: str = "RxNorm",
    desired_granularity: str | None = "ingredient",
    context: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> SemanticAlignmentUnit:
    """Build a synthetic :class:`SemanticAlignmentUnit` for contract testing."""
    norm = normalize_text(text)
    payload = {"text": text, "target": target_terminology, "level": desired_granularity}
    return SemanticAlignmentUnit(
        sau_id=stable_id("SAU", payload, length=12),
        mapping_key=stable_id("KEY", payload, length=16),
        source=SourceRecord(source_type=SourceType.STRUCTURED_FIELD, raw_value=text),
        mention=Mention(text=text, normalized_text=norm),
        context=ContextEnvelope.model_validate(dict(context or {})),
        attributes=ContextualAttributes.model_validate(dict(attributes or {})),
        task=MappingTask(
            operation="contract_test",
            mapping_intent="contract_verification",
            target_terminology=target_terminology,
            desired_granularity=desired_granularity,
        ),
        provenance=Provenance(
            run_id="contract-test-run",
            package_version="contract-test",
            profile_name="contract_test",
            profile_version="0.1.0",
        ),
        readiness=ReadinessAssessment(status=ReadinessStatus.READY_FOR_MAPPING),
    )


def make_contract_candidate(
    code: str = "T100",
    label: str = "toyprolol",
    *,
    vocabulary: str = "RxNorm",
    concept_class: str | None = "IN",
    status: ConceptStatus = ConceptStatus.ACTIVE,
    relationships: Sequence[KnowledgeStatement] = (),
) -> CandidateConcept:
    """Build a synthetic :class:`CandidateConcept` with retrieval evidence for contract tests."""
    return CandidateConcept(
        candidate_id=code,
        preferred_label=label,
        vocabulary=vocabulary,
        concept_class=concept_class,
        status=status,
        relationships=list(relationships),
        retrieval_evidence=[
            RetrievalEvidence(
                channel=RetrievalChannel.NORMALIZED_LOOKUP,
                query=label,
                provider="contract_test",
                score=100.0,
                rank=1,
            )
        ],
    )


def make_contract_package(
    sau: SemanticAlignmentUnit,
    candidates: Sequence[CandidateConcept] | None = None,
    *,
    target_spec: TargetSpec | None = None,
) -> KnowledgePackage:
    """Build a synthetic :class:`KnowledgePackage` (with typed links when ``target_spec`` is given)."""
    resolved = (
        list(candidates)
        if candidates is not None
        else [make_contract_candidate(vocabulary=sau.task.target_terminology)]
    )
    package = KnowledgePackage(
        sau_id=sau.sau_id,
        candidates=resolved,
        coding_policies=[
            CodingPolicy(
                policy_id="candidate_constrained",
                description="Select only from retrieved candidates",
                rule="Only select concepts present in candidates.",
                applies_to=[sau.task.operation],
            )
        ],
    )
    if target_spec is not None:
        links = derive_link_set(package, target_spec, sau.task.desired_granularity)
        package = package.model_copy(update={"links": links})
    return package


def check_operator_metadata(op: object, expected_protocol: type) -> None:
    """Verify protocol conformance and required operator metadata."""
    assert isinstance(op, Operator), f"{op!r} does not satisfy the Operator protocol"
    assert isinstance(op, expected_protocol), (
        f"{op!r} does not satisfy the {expected_protocol.__name__} protocol"
    )
    assert isinstance(op.op_id, str) and op.op_id.strip(), "op_id must be a non-empty string"
    assert isinstance(op.version, str) and op.version.strip(), "version must be a non-empty string"
    assert isinstance(op.uses_model, bool), "uses_model must be a bool"


def check_cleaner_contract(
    cleaner: Cleaner,
    sau: SemanticAlignmentUnit | None = None,
    *,
    prefilled_attribute: tuple[str, Any] = ("route", "oral"),
) -> SemanticAlignmentUnit:
    """Verify :class:`Cleaner` contract: non-destructive (G1), pure, and deterministic."""
    check_operator_metadata(cleaner, Cleaner)
    base = (sau or make_contract_sau()).model_copy(deep=True)
    snapshot = base.model_copy(deep=True)

    first = cleaner.clean(base)
    assert base == snapshot, f"cleaner {cleaner.op_id!r} mutated its input SAU in place"
    require_non_destructive(base, first, cleaner.op_id)

    slot_name, slot_value = prefilled_attribute
    if slot_name in type(base.attributes).model_fields and slot_name != "extensions":
        prefilled = base.model_copy(
            update={"attributes": base.attributes.model_copy(update={slot_name: slot_value})}
        )
    else:
        ext = {**base.attributes.extensions, slot_name: slot_value}
        prefilled = base.model_copy(
            update={"attributes": base.attributes.model_copy(update={"extensions": ext})}
        )
    prefilled_before = prefilled.model_copy(deep=True)
    prefilled_after = cleaner.clean(prefilled)
    require_non_destructive(prefilled_before, prefilled_after, cleaner.op_id)

    if not cleaner.uses_model:
        second = cleaner.clean(base.model_copy(deep=True))
        assert first == second, f"deterministic cleaner {cleaner.op_id!r} is not reproducible"
    return first


def check_matcher_contract(
    matcher: Matcher,
    sau: SemanticAlignmentUnit | None = None,
) -> CandidateSet:
    """Verify :class:`Matcher` contract: ``CandidateSet`` shape, evidence, and determinism."""
    check_operator_metadata(matcher, Matcher)
    assert isinstance(matcher.requires_network, bool), "Matcher.requires_network must be a bool"
    base = (sau or make_contract_sau()).model_copy(deep=True)
    snapshot = base.model_copy(deep=True)

    first = matcher.match(base)
    assert base == snapshot, f"matcher {matcher.op_id!r} mutated its input SAU in place"
    assert isinstance(first, CandidateSet), (
        f"matcher {matcher.op_id!r} returned {type(first).__name__}, not CandidateSet"
    )
    assert first.matcher == matcher.op_id, (
        f"CandidateSet.matcher {first.matcher!r} does not match op_id {matcher.op_id!r}"
    )
    for candidate in first.candidates:
        assert isinstance(candidate, CandidateConcept)
        assert candidate.retrieval_evidence, (
            f"candidate {candidate.candidate_id!r} from {matcher.op_id!r} lacks retrieval_evidence"
        )

    if not matcher.uses_model:
        second = matcher.match(base.model_copy(deep=True))
        assert first == second, f"deterministic matcher {matcher.op_id!r} is not reproducible"
    return first


def check_expander_contract(
    expander: Expander,
    sau: SemanticAlignmentUnit,
    anchors: Sequence[CandidateConcept],
    *,
    profile: TaskProfile | None = None,
) -> ExpansionSet:
    """Verify :class:`Expander` contract: ``ExpansionSet`` paths, G2 compliance, determinism."""
    check_operator_metadata(expander, Expander)
    base_sau = sau.model_copy(deep=True)
    base_anchors = [c.model_copy(deep=True) for c in anchors]
    sau_snap = base_sau.model_copy(deep=True)
    anchors_snap = [c.model_copy(deep=True) for c in base_anchors]

    first = expander.expand(base_sau, base_anchors)
    assert base_sau == sau_snap, f"expander {expander.op_id!r} mutated input SAU in place"
    assert base_anchors == anchors_snap, (
        f"expander {expander.op_id!r} mutated input anchors in place"
    )
    assert isinstance(first, ExpansionSet), (
        f"expander {expander.op_id!r} returned {type(first).__name__}, not ExpansionSet"
    )
    assert first.expander == expander.op_id
    for member in first.members:
        assert isinstance(member, ExpansionMember)
        assert member.hops >= 1 and member.path, (
            f"expanded member {member.concept.candidate_id!r} must record a non-empty path (hops >= 1)"
        )
        if profile is not None:
            for step in member.path:
                permitted = (
                    step.predicate not in profile.knowledge.prohibited_relationships
                    if step.direction == "lexical"
                    else profile.knowledge.permits(step.predicate)
                )
                assert permitted, (
                    f"expander {expander.op_id!r} emitted prohibited predicate {step.predicate!r} (G2)"
                )

    if not expander.uses_model:
        second = expander.expand(base_sau.model_copy(deep=True), base_anchors)
        assert first == second, f"deterministic expander {expander.op_id!r} is not reproducible"
    return first


def check_linker_contract(
    linker: Linker,
    sau: SemanticAlignmentUnit,
    candidate_sets: Sequence[CandidateSet],
    *,
    profile: TaskProfile | None = None,
) -> KnowledgePackage:
    """Verify :class:`Linker` contract: package identity, G2 policy filtering, determinism."""
    check_operator_metadata(linker, Linker)
    base_sau = sau.model_copy(deep=True)
    sau_snap = base_sau.model_copy(deep=True)
    first = linker.link(base_sau, candidate_sets)
    assert base_sau == sau_snap, f"linker {linker.op_id!r} mutated input SAU in place"
    assert isinstance(first, KnowledgePackage), (
        f"linker {linker.op_id!r} returned {type(first).__name__}, not KnowledgePackage"
    )
    assert first.sau_id == base_sau.sau_id, (
        f"package sau_id {first.sau_id!r} does not match SAU {base_sau.sau_id!r}"
    )

    if profile is not None:
        statements = [
            *(s for candidate in first.candidates for s in candidate.relationships),
            *first.ontology_guidance,
            *first.umls_guidance,
            *first.kg_guidance,
            *(s for conflict in first.conflicts for s in conflict.statements),
        ]
        leaked = sorted(
            {s.predicate for s in statements if not profile.knowledge.permits(s.predicate)}
        )
        assert not leaked, f"linker {linker.op_id!r} leaked prohibited predicates (G2): {leaked}"

    if not linker.uses_model:
        second = linker.link(base_sau.model_copy(deep=True), candidate_sets)
        assert [
            (r.resource_name, r.provider_name, r.terminology_release)
            for r in first.resource_provenance
        ] == [
            (r.resource_name, r.provider_name, r.terminology_release)
            for r in second.resource_provenance
        ], f"deterministic linker {linker.op_id!r} changed resources across calls"
        assert first.model_copy(update={"resource_provenance": []}) == second.model_copy(
            update={"resource_provenance": []}
        ), f"deterministic linker {linker.op_id!r} is not reproducible"
    return first


def check_resolver_contract(
    resolver: Resolver,
    sau: SemanticAlignmentUnit,
    package: KnowledgePackage,
    *,
    profile: TaskProfile | None = None,
) -> AlignmentProposal | PassToNext:
    """Verify :class:`Resolver` contract: G3 isolated view, G4 package evidence, determinism."""
    check_operator_metadata(resolver, Resolver)
    assert isinstance(resolver.requires_network, bool), "Resolver.requires_network must be a bool"
    prov = resolver.provenance()
    assert isinstance(prov, ModelProvenance), "Resolver.provenance() must return ModelProvenance"

    view = (
        isolated_view(sau, "resolve", profile) if profile is not None else sau.model_copy(deep=True)
    )
    pkg_copy = package.model_copy(deep=True)
    view_snap = view.model_copy(deep=True)
    pkg_snap = pkg_copy.model_copy(deep=True)

    first = resolver.resolve(view, pkg_copy)
    assert view == view_snap, f"resolver {resolver.op_id!r} mutated input SAU in place"
    assert pkg_copy == pkg_snap, f"resolver {resolver.op_id!r} mutated input package in place"
    assert isinstance(first, AlignmentProposal | PassToNext), (
        f"resolver {resolver.op_id!r} returned {type(first).__name__}, "
        "expected AlignmentProposal or PassToNext"
    )

    if isinstance(first, AlignmentProposal):
        assert first.sau_id == view.sau_id
        for concept in first.selected_concepts:
            evidence = package.evidence_for(sau.task.target_terminology, concept.concept_id)
            assert evidence is not None, (
                f"resolver {resolver.op_id!r} selected {concept.concept_id!r} outside package (G4)"
            )
        if first.analysis_level_concept is not None:
            evidence = package.evidence_for(
                sau.task.target_terminology, first.analysis_level_concept.concept_id
            )
            assert evidence is not None, (
                f"resolver {resolver.op_id!r} named analysis concept outside package (G4)"
            )
        if first.source_level_concept is not None:
            src_vocab = sau.task.source_vocabulary(first.source_level_concept.vocabulary)
            evidence = package.evidence_for(src_vocab, first.source_level_concept.concept_id)
            assert evidence is not None, (
                f"resolver {resolver.op_id!r} named source concept outside package (G4)"
            )

    if not resolver.uses_model:
        second = resolver.resolve(view.model_copy(deep=True), package.model_copy(deep=True))
        assert first == second, f"deterministic resolver {resolver.op_id!r} is not reproducible"
    return first


def check_selector_contract(
    selector: Selector,
    sau: SemanticAlignmentUnit,
    expansion: ExpansionSet,
    *,
    profile: TaskProfile | None = None,
) -> ConceptSetProposal | PassToNext:
    """Verify :class:`Selector` contract: G3 isolated view, G4 expansion evidence, determinism."""
    check_operator_metadata(selector, Selector)
    assert isinstance(selector.requires_network, bool), "Selector.requires_network must be a bool"
    prov = selector.provenance()
    assert isinstance(prov, ModelProvenance), "Selector.provenance() must return ModelProvenance"

    view = (
        isolated_view(sau, "select", profile) if profile is not None else sau.model_copy(deep=True)
    )
    exp_copy = expansion.model_copy(deep=True)
    view_snap = view.model_copy(deep=True)
    exp_snap = exp_copy.model_copy(deep=True)

    first = selector.select(view, exp_copy)
    assert view == view_snap, f"selector {selector.op_id!r} mutated input SAU in place"
    assert exp_copy == exp_snap, f"selector {selector.op_id!r} mutated input expansion in place"
    assert isinstance(first, ConceptSetProposal | PassToNext), (
        f"selector {selector.op_id!r} returned {type(first).__name__}, "
        "expected ConceptSetProposal or PassToNext"
    )

    if isinstance(first, ConceptSetProposal):
        assert first.sau_id == view.sau_id
        allowed = {(a.vocabulary.casefold(), a.concept_code) for a in expansion.anchors} | {
            (m.concept.vocabulary.casefold(), m.concept.candidate_id) for m in expansion.members
        }
        for member in first.members:
            key = (member.vocabulary.casefold(), member.concept_code)
            assert key in allowed, (
                f"selector {selector.op_id!r} emitted {member.vocabulary}:{member.concept_code} "
                "outside the expansion (G4)"
            )

    if not selector.uses_model:
        second = selector.select(view.model_copy(deep=True), expansion.model_copy(deep=True))
        assert first == second, f"deterministic selector {selector.op_id!r} is not reproducible"
    return first


def check_check_contract(
    check_op: Check,
    proposal: AlignmentProposal,
    package: KnowledgePackage,
    sau: SemanticAlignmentUnit | None = None,
) -> list[ValidationFinding]:
    """Verify :class:`Check` contract: findings type, non-mutation, determinism."""
    check_operator_metadata(check_op, Check)
    prop_copy = proposal.model_copy(deep=True)
    pkg_copy = package.model_copy(deep=True)
    sau_copy = sau.model_copy(deep=True) if sau is not None else None
    prop_snap = prop_copy.model_copy(deep=True)
    pkg_snap = pkg_copy.model_copy(deep=True)
    sau_snap = sau_copy.model_copy(deep=True) if sau_copy is not None else None

    first = check_op.check(prop_copy, pkg_copy, sau_copy)
    assert prop_copy == prop_snap, f"check {check_op.op_id!r} mutated proposal in place"
    assert pkg_copy == pkg_snap, f"check {check_op.op_id!r} mutated package in place"
    assert sau_copy == sau_snap, f"check {check_op.op_id!r} mutated SAU in place"
    assert isinstance(first, list)
    assert all(isinstance(item, ValidationFinding) for item in first)

    if not check_op.uses_model:
        second = check_op.check(
            proposal.model_copy(deep=True),
            package.model_copy(deep=True),
            sau.model_copy(deep=True) if sau is not None else None,
        )
        assert first == second, f"deterministic check {check_op.op_id!r} is not reproducible"
    return first


def check_target_spec_contract(spec: TargetSpec) -> None:
    """Verify :class:`TargetSpec` round-trip, digest, and target family invariants."""
    assert isinstance(spec, TargetSpec)
    assert spec.label
    assert spec.in_family(spec.target), "TargetSpec.family must contain TargetSpec.target"
    assert len(spec.digest) == 64
    round_trip = TargetSpec.model_validate(spec.model_dump(mode="json", by_alias=True))
    assert round_trip.digest == spec.digest


def check_domain_pack_contract(domain: str) -> None:
    """Verify a registered domain pack loads and unloads cleanly."""
    assert load_domain(domain), f"domain pack {domain!r} failed to load"
    key = domain.casefold()
    assert key in loaded_domains(), f"domain pack {domain!r} not in loaded_domains()"
    with without_domain(domain):
        assert key not in loaded_domains(), f"without_domain({domain!r}) did not unload cleanly"
    assert key in loaded_domains(), f"without_domain({domain!r}) did not restore domain on exit"
