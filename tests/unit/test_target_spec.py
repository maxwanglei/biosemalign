"""Target terminology specs: the RxNorm spec reproduces the v0.2 drug rules exactly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from tests.support import PRODUCT, make_pipeline

from biosemalign import verify_artifact
from biosemalign.domains.drug.tools.links import common as link_classes
from biosemalign.domains.drug.tools.omop import RXNORM_CLASS_TO_TTY
from biosemalign.enums import (
    Authority,
    ConceptStatus,
    DecisionType,
    RelationshipType,
    SemanticLossKind,
)
from biosemalign.exceptions import ProfileError
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage, KnowledgeStatement
from biosemalign.targets import builtin_specs, load_target_spec, spec_for_vocabulary
from biosemalign.targets.spec import LossRule, TargetSpec

RXNORM = builtin_specs()["rxnorm"]


# -- the RxNorm spec is the v0.2 tables -------------------------------------------


def test_rxnorm_spec_reproduces_the_hard_coded_granularity_tables() -> None:
    # The adjudicator's and the validator's former tables, side by side.
    assert {g: RXNORM.predicate(g) for g in RXNORM.granularities} == {
        "ingredient": "has_ingredient",
        "precise_ingredient": "has_precise_ingredient",
        "active_moiety": "has_boss",
    }
    assert RXNORM.terminal_classes("ingredient") == {"IN"}  # MIN is never terminal
    assert RXNORM.terminal_classes("precise_ingredient") == {"PIN"}
    assert RXNORM.terminal_classes("active_moiety") == {"IN"}
    assert RXNORM.accepted_classes("active_moiety") == {"IN", "PIN"}
    assert RXNORM.accepted_classes("ingredient") == {"IN"}
    assert RXNORM.release_bound_predicates == [
        "has_ingredient",
        "has_precise_ingredient",
        "has_boss",
    ]
    assert RXNORM.inferred_class("has_ingredient") == "IN"
    assert RXNORM.inferred_class("has_precise_ingredient") == "PIN"
    assert RXNORM.inferred_class("has_boss") is None
    assert RXNORM.inferred_class("has_tradename") is None
    assert RXNORM.family == ["RxNorm", "RxNorm Extension"]
    assert RXNORM.predicate("nonsense") is None
    assert RXNORM.terminal_classes(None) == frozenset()
    assert RXNORM.accepted_classes("nonsense") == frozenset()


def test_rxnorm_spec_reproduces_the_hard_coded_loss_tables() -> None:
    by_kind: dict[SemanticLossKind, list[LossRule]] = {}
    for rule in RXNORM.losses:
        by_kind.setdefault(rule.kind, []).append(rule)
    assert set(by_kind[SemanticLossKind.STRENGTH][0].classes) == {
        "SCD",
        "SBD",
        "SCDC",
        "SBDC",
        "GPCK",
        "BPCK",
    }
    assert set(by_kind[SemanticLossKind.DOSE_FORM][0].classes) == {
        "SCD",
        "SBD",
        "SCDF",
        "SBDF",
        "SCDG",
        "SBDG",
        "GPCK",
        "BPCK",
    }
    assert set(by_kind[SemanticLossKind.BRAND_IDENTITY][0].classes) == {
        "SBD",
        "SBDC",
        "SBDF",
        "SBDG",
        "BN",
        "BPCK",
    }
    assert [r.rule for r in by_kind[SemanticLossKind.SALT_FORM]] == [
        "source_not_selected",
        "unselected_objects",
    ]
    assert [r.kind for r in RXNORM.losses][:2] == [
        SemanticLossKind.STRENGTH,
        SemanticLossKind.DOSE_FORM,
    ]


def test_class_aliases_replace_every_copy_of_the_omop_class_map() -> None:
    assert RXNORM_CLASS_TO_TTY["Quant Clinical Drug"] == "SCD"
    assert RXNORM.class_aliases == RXNORM_CLASS_TO_TTY
    assert set(link_classes.INGREDIENT_CLASSES) == {"IN", "Ingredient"}
    assert set(link_classes.BRAND_CLASSES) == {"BN", "Brand Name"}
    assert set(link_classes.BRANDED_PRODUCT_CLASSES) == {
        "SBD",
        "SBDC",
        "SBDF",
        "SBDG",
        "Branded Drug",
        "Quant Branded Drug",
        "Branded Drug Comp",
        "Branded Drug Form",
        "Branded Dose Group",
    }
    assert RXNORM.canonical_class("Quant Branded Drug") == "SBD"
    assert RXNORM.canonical_class("IN") == "IN"
    assert RXNORM.canonical_class(None) is None
    assert RXNORM.is_terminal("Ingredient", "ingredient")


# -- loss rules ---------------------------------------------------------------------


def _concept(
    code: str, label: str, klass: str, **edges: list[tuple[str, str | None]]
) -> CandidateConcept:
    return CandidateConcept(
        candidate_id=code,
        preferred_label=label,
        vocabulary="RxNorm",
        concept_class=klass,
        status=ConceptStatus.ACTIVE,
        relationships=[
            KnowledgeStatement(
                subject_id=code,
                predicate=predicate,
                object_id=obj,
                object_label=obj_label,
                source_resource="RxNorm",
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
            )
            for predicate, objects in edges.items()
            for obj, obj_label in objects
        ],
    )


def _losses(
    source: CandidateConcept,
    *selected: str,
    target_label: str | None = "base",
    mention: str | None = None,
    route: str | None = None,
) -> list[tuple[str, str | None]]:
    return [
        (loss.kind.value, loss.source_value)
        for loss in RXNORM.expected_losses(
            source,
            selected_ids=list(selected),
            target_label=target_label,
            mention_text=mention,
            context={"route": route},
        )
    ]


def test_branded_product_losses_follow_rule_order() -> None:
    product = _concept(
        "2001",
        "base 10 MG Oral Tablet [Brand]",
        "Quant Branded Drug",  # an OMOP spelling: aliases apply
        has_dose_form=[("DF1", "Oral Tablet")],
        has_precise_ingredient=[("PIN1", "base sodium")],
        has_tradename=[],
    )
    assert _losses(product, "IN1", route="oral") == [
        ("strength", "10 MG"),
        ("dose_form", "Oral Tablet"),
        ("salt_form", "base sodium"),
        ("brand_identity", "base 10 MG Oral Tablet [Brand]"),  # no tradename edge: the label
        ("route", "oral"),
    ]
    # A target that carries the strength loses none; nor does an unknown target label.
    assert ("strength", "10 MG") not in _losses(product, "IN1", target_label="base 10 MG")
    assert ("strength", "10 MG") in _losses(product, "IN1", target_label=None)


def test_precise_ingredient_losses_and_first_rule_of_a_kind_wins() -> None:
    salt = _concept("1101", "base sodium", "PIN", has_precise_ingredient=[("X", None)])
    assert _losses(salt, "1001") == [("salt_form", "base sodium")]
    assert _losses(salt, "1101") == []  # mapped to itself: nothing lost
    clinical = _concept(
        "5001", "base sodium 10 MG", "SCDC", has_precise_ingredient=[("1101", None)]
    )
    assert _losses(clinical, "1001", mention="BASE 10mg") == [
        ("strength", "10 MG"),
        ("salt_form", "1101"),
    ]
    brand = _concept("3001", "Brand", "BN", has_tradename=[("TN", "Brandname"), ("T2", None)])
    assert _losses(brand, "1001") == [("brand_identity", "Brandname, T2")]


def test_combination_loss_comes_from_the_granularity() -> None:
    loss = RXNORM.combination_loss("ingredient", count=2, source_label="a / b")
    assert loss is not None and loss.kind is SemanticLossKind.COMBINATION_COMPONENT
    assert "represented by 2 separate ingredient concepts" in loss.description
    assert RXNORM.combination_loss("nonsense", count=2, source_label="x") is None


# -- spec validation and loading --------------------------------------------------------


def _spec_payload(**overrides: Any) -> dict[str, Any]:
    payload = RXNORM.model_dump(mode="json")
    payload.update(overrides)
    return payload


def test_specs_are_validated() -> None:
    with pytest.raises(ValidationError, match="needs \\['pattern'\\]"):
        LossRule.model_validate({"kind": "strength", "rule": "pattern", "description": "x"})
    with pytest.raises(ValidationError, match="needs \\['field'\\]"):
        LossRule.model_validate({"kind": "route", "rule": "context", "description": "x"})
    with pytest.raises(ValidationError):
        LossRule.model_validate(
            {"kind": "strength", "rule": "pattern", "description": "x", "pattern": "("}
        )
    with pytest.raises(ValidationError, match="twice"):
        TargetSpec.model_validate(_spec_payload(family=["RxNorm", "rxnorm"]))
    spec = TargetSpec.model_validate({"spec_id": "x", "version": "1", "target": "ATC"})
    assert spec.family == ["ATC"] and spec.in_family("atc") and not spec.in_family(None)
    assert spec.label == "x@1" and len(spec.digest) == 64


def test_specs_load_by_id_by_path_or_by_terminology(tmp_path: Path) -> None:
    current_rxnorm = builtin_specs()["rxnorm"]
    assert current_rxnorm == RXNORM
    assert load_target_spec("rxnorm", terminology="RxNorm") is current_rxnorm
    assert load_target_spec(None, terminology="rxnorm") is current_rxnorm
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(_spec_payload(spec_id="custom", version="9")))
    assert load_target_spec(str(custom), terminology="RxNorm").label == "custom@9"
    with pytest.raises(ProfileError, match="describes 'RxNorm'"):
        load_target_spec("rxnorm", terminology="ATC")
    with pytest.raises(ProfileError, match="neither a built-in"):
        load_target_spec("nowhere.yaml", terminology="RxNorm")
    broken = tmp_path / "broken.yaml"
    broken.write_text("spec_id: [unclosed")
    with pytest.raises(ProfileError, match="invalid target spec"):
        load_target_spec(str(broken), terminology="RxNorm")
    # A terminology without a spec gets an empty one: nothing is reachable.
    empty = load_target_spec(None, terminology="SomethingElse")
    assert empty.granularities == {} and empty.predicate("ingredient") is None
    assert spec_for_vocabulary("RxNorm Extension") is current_rxnorm
    assert spec_for_vocabulary("MeSH") is None and spec_for_vocabulary(None) is None


# -- profiles -------------------------------------------------------------------------


def _profile_payload() -> dict[str, Any]:
    return load_profile("faers_drug_normalization").model_dump(mode="json")


def test_profiles_accept_a_target_section() -> None:
    payload = _profile_payload()
    payload["task"] = {"operation": "drug_normalization", "mapping_intent": "x"}
    payload["target"] = {
        "spec": "rxnorm",
        "level": "precise_ingredient",
        "source_terminologies": ["RxNorm Extension"],
    }
    profile = TaskProfile.model_validate(payload)
    assert profile.target.terminology == "RxNorm"
    assert profile.target.level == "precise_ingredient"
    assert profile.target.spec == "rxnorm"
    assert profile.target.source_terminologies == ["RxNorm Extension"]
    assert profile.target_spec() is builtin_specs()["rxnorm"]
    assert profile.target_spec() == RXNORM

    agreeing = _profile_payload()
    agreeing["target"] = {"terminology": "RxNorm", "level": "ingredient"}
    assert TaskProfile.model_validate(agreeing).target.level == "ingredient"
    for target, message in [
        ({"granularity": "x", "terminology": "RxNorm"}, "Extra inputs"),
        ({"level": "x"}, "Field required"),
        ("rxnorm", "valid dictionary"),
    ]:
        payload = _profile_payload()
        payload["target"] = target
        with pytest.raises(ValidationError, match=message):
            TaskProfile.model_validate(payload)


def test_a_target_without_a_spec_cannot_be_projected(tmp_path: Path) -> None:
    payload = _profile_payload()
    payload["target"]["spec"] = None
    profile = TaskProfile.model_validate(payload)
    assert profile.target_spec() is builtin_specs()["rxnorm"]
    assert profile.target_spec() == RXNORM


# -- one source for the adjudicator and the validator ---------------------------------


def test_a_custom_spec_changes_what_is_reported_and_what_is_checked(tmp_path: Path) -> None:
    payload = _spec_payload(spec_id="custom", version="2")
    for rule in payload["losses"]:
        if rule["kind"] == "dose_form":
            rule["description"] = "the custom wording for a lost dose form"
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(payload))
    profile_payload = _profile_payload()
    profile_payload["target"]["spec"] = str(custom)
    profile = TaskProfile.model_validate(profile_payload)
    with make_pipeline(tmp_path / "cache", profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(PRODUCT))
    losses = {loss.kind: loss.description for loss in artifact.decision.semantic_loss}
    assert losses[SemanticLossKind.DOSE_FORM] == "the custom wording for a lost dose form"
    # The validator derived the same payload, so nothing was flagged.
    assert not [f for f in artifact.validation.findings if "SEMANTIC_LOSS" in f.code]
    assert artifact.decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE


def test_an_unknown_granularity_still_abstains(tmp_path: Path) -> None:
    payload = _profile_payload()
    payload["target"]["level"] = "clinical_drug"
    profile = TaskProfile.model_validate(payload)
    with make_pipeline(tmp_path / "cache", profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(PRODUCT))
    assert artifact.decision.decision is not DecisionType.MAPPED
    assert verify_artifact(artifact).valid


# -- release-bound evidence across the family ------------------------------------------


def test_rxnorm_extension_edges_are_release_bound_evidence() -> None:
    product = CandidateConcept(
        candidate_id="OMOP1",
        preferred_label="Extbrand tablet",
        vocabulary="RxNorm Extension",
        concept_class="SBD",
        status=ConceptStatus.ACTIVE,
        relationships=[
            KnowledgeStatement(
                subject_id="OMOP1",
                predicate="has_ingredient",
                object_id="10",
                object_vocabulary="RxNorm",
                source_resource="RxNorm Extension",
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
            ),
            KnowledgeStatement(
                subject_id="OMOP1",
                predicate="has_ingredient",
                object_id="M1",
                object_vocabulary="MeSH",
                source_resource="RxNorm Extension",
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
            ),
        ],
    )
    index = KnowledgePackage(
        sau_id="SAU-1", candidates=[product]
    ).qualified_concept_evidence_index()
    target = index[("rxnorm", "10")]
    assert (target.concept_class, target.status) == ("IN", ConceptStatus.ACTIVE)
    # An edge leaving the family is not the family's release evidence.
    outside = index[("mesh", "M1")]
    assert (outside.concept_class, outside.status) == (None, ConceptStatus.UNKNOWN)


# -- retargeted drug rules still fire ------------------------------------------------------


def test_spec_driven_drug_rules_catch_dropped_components_and_salt_conflation(
    tmp_path: Path,
) -> None:
    from tests.support import COMBINATION, SALT

    from biosemalign.domains.drug.checks import validate_drug_semantics

    with make_pipeline(tmp_path / "cache") as pipeline:
        combination = pipeline.align_artifact(pipeline.make_request(COMBINATION))
        salt = pipeline.align_artifact(pipeline.make_request(SALT))
        profile = pipeline.profile

    assert combination.decision.decision is DecisionType.MULTI_CONCEPT_MAPPED
    dropped = combination.proposal.model_copy(
        update={"selected_concepts": combination.proposal.selected_concepts[:1]}
    )
    codes = {
        f.code
        for f in validate_drug_semantics(
            dropped, combination.knowledge_package, profile, sau=combination.sau
        )
    }
    assert "COMBINATION_INGREDIENT_DROPPED" in codes

    # At precise-ingredient level the combination check walks has_precise_ingredient,
    # which the combination concept lacks: nothing to drop.
    precise = profile.model_copy(
        update={"target": profile.target.model_copy(update={"level": "precise_ingredient"})}
    )
    codes = {
        f.code
        for f in validate_drug_semantics(
            dropped, combination.knowledge_package, precise, sau=combination.sau
        )
    }
    assert "COMBINATION_INGREDIENT_DROPPED" not in codes

    assert salt.decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    exact = salt.proposal.model_copy(
        update={"relationship_to_source": RelationshipType.EXACT, "semantic_loss": []}
    )
    findings = validate_drug_semantics(exact, salt.knowledge_package, profile, sau=salt.sau)
    assert "SALT_ACTIVE_MOIETY_CONFLATION" in {f.code for f in findings}
    # The check reads class spellings through the spec's aliases.
    aliased = exact.model_copy(
        update={
            "source_level_concept": exact.source_level_concept.model_copy(
                update={"concept_class": "Precise Ingredient"}
            )
        }
    )
    findings = validate_drug_semantics(aliased, salt.knowledge_package, profile, sau=salt.sau)
    assert "SALT_ACTIVE_MOIETY_CONFLATION" in {f.code for f in findings}


def test_identity_levels_multihop_ancestor_loss_and_object_class() -> None:
    from biosemalign.modules.link.targets import targets_at

    spec = builtin_specs()["toydisease"]
    assert spec.has_granularity(None)
    assert spec.has_granularity("Disease")
    assert spec.max_hops_for("Group") == 2

    d2 = CandidateConcept(
        candidate_id="D2",
        preferred_label="severe toy fever",
        vocabulary="ToyDisease",
        concept_class="Disease",
        status=ConceptStatus.ACTIVE,
        relationships=[
            KnowledgeStatement(
                subject_id="D2",
                predicate="is_a",
                object_id="D1",
                object_label="toy fever",
                object_class="Disease",
                source_resource="ToyDisease",
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
            ),
            KnowledgeStatement(
                subject_id="D2",
                predicate="in_group",
                object_id="G1",
                object_label="toy fever disorders",
                object_class="Group",
                source_resource="ToyDisease",
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
                direct=False,
                extensions={"step_subject_id": "D1"},
            ),
        ],
    )
    identity_targets = targets_at(d2, spec, None)
    assert [(r.concept_code, r.statement) for r in identity_targets] == [("D2", None)]
    disease_targets = targets_at(d2, spec, "Disease")
    assert [(r.concept_code, r.statement) for r in disease_targets] == [("D2", None)]
    group_targets = targets_at(d2, spec, "Group")
    assert [(r.concept_code, r.predicates) for r in group_targets] == [("G1", ("is_a", "in_group"))]

    losses = spec.expected_losses(
        d2,
        selected_ids=["G1"],
        target_label="toy fever disorders",
        mention_text="severe toy fever",
        context={},
    )
    assert [(loss.kind, loss.source_value) for loss in losses] == [
        (SemanticLossKind.SPECIFICITY, "severe toy fever")
    ]

    # Object class wins when present, with legacy extensions["target_tty"] fallback.
    pkg = KnowledgePackage(
        sau_id="s",
        candidates=[
            CandidateConcept(
                candidate_id="C1",
                preferred_label="Compound",
                vocabulary="Test",
                status=ConceptStatus.ACTIVE,
                relationships=[
                    KnowledgeStatement(
                        subject_id="C1",
                        predicate="rel",
                        object_id="O_NEW",
                        object_class="NewClass",
                        source_resource="Test",
                        authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
                    ),
                    KnowledgeStatement(
                        subject_id="C1",
                        predicate="rel",
                        object_id="O_LEGACY",
                        extensions={"target_tty": "LegacyClass"},
                        source_resource="Test",
                        authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
                    ),
                ],
            )
        ],
    )
    assert pkg.evidence_for("Test", "O_NEW").concept_class == "NewClass"
    assert pkg.evidence_for("Test", "O_LEGACY").concept_class == "LegacyClass"
