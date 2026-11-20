"""Benchmark denominators and joins must account for every source case."""

import pytest
from pydantic import ValidationError

from biosemalign.enums import (
    Authority,
    ConceptStatus,
    DecisionType,
    RelationshipType,
    RoutingStatus,
    ValidationStatus,
)
from biosemalign.evaluation.metrics import (
    EvaluationInputError,
    EvaluationPrediction,
    decision_case_id,
    evaluate_predictions,
    recall_at_k,
    selective_accuracy,
)
from biosemalign.schemas.decision import AlignmentDecision, SelectedConcept, ValidationResult
from biosemalign.schemas.evaluation import GoldMapping
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgePackage, KnowledgeStatement
from biosemalign.schemas.provenance import Provenance


def gold(case_id="case-1", ids=None, **kwargs):
    return GoldMapping(
        case_id=case_id,
        source_value="same source",
        expected_concept_ids=["one"] if ids is None else ids,
        **kwargs,
    )


def decision(case_id="case-1", ids=None, route=RoutingStatus.AUTO_ACCEPT, **updates):
    values = {
        "sau_id": "sau-" + case_id,
        "record_id": case_id,
        "source_value": "same source",
        "decision": DecisionType.MAPPED,
        "selected_concepts": [
            SelectedConcept(concept_id=value, label=value, vocabulary="RxNorm")
            for value in (["one"] if ids is None else ids)
        ],
        "relationship_to_source": RelationshipType.EXACT,
        "validation": ValidationResult(status=ValidationStatus.VALID, rules_version="test"),
        "routing_status": route,
        "provenance": Provenance(
            run_id="run", package_version="test", profile_name="test", profile_version="1"
        ),
    }
    values.update(updates)
    return AlignmentDecision(**values)


def candidate(identifier, relationships=None):
    return CandidateConcept(
        candidate_id=identifier,
        preferred_label=identifier,
        vocabulary="RxNorm",
        concept_class="IN",
        status=ConceptStatus.ACTIVE,
        relationships=relationships or [],
    )


def test_case_id_precedence_and_missing_identity():
    assert decision_case_id(decision(extensions={"case_id": "explicit"})) == "explicit"
    assert decision_case_id(decision()) == "case-1"
    assert decision_case_id(decision(record_id=None)) == "sau-case-1"
    with pytest.raises(EvaluationInputError, match="no usable case ID"):
        decision_case_id(decision(record_id=None, sau_id=" "))
    for bad in ["", " ", 1]:
        with pytest.raises(EvaluationInputError, match="non-blank string"):
            decision_case_id(decision(extensions={"case_id": bad}))


def test_identical_source_text_is_not_a_join_key_and_missing_cases_remain_denominator():
    cases = [
        gold("a", category="exact"),
        gold("b", category="exact"),
        gold("c", ids=[], category="negative"),
        gold("missing"),
    ]
    predictions = [
        decision("a"),
        decision("b", ids=["wrong"], route=RoutingStatus.HUMAN_REVIEW),
        decision("c", ids=[], route=RoutingStatus.ABSTAIN),
        decision("extra"),
    ]
    report = evaluate_predictions(cases, predictions)
    assert not report.complete
    assert report.overall.gold_cases == 4 and report.overall.evaluated == 3
    assert report.overall.correct_predictions == 2
    assert report.overall.coverage == 0.25
    assert report.overall.accuracy == 0.5
    assert report.overall.auto_accept_precision == 1
    assert report.missing_prediction_ids == ["missing"]
    assert report.unexpected_prediction_ids == ["extra"]
    assert report.categories["exact"].reviewed == 1
    assert report.categories["negative"].abstained == 1
    assert report.as_dict()["unexpected_predictions"] == 1
    assert report.as_dict()["categories"]["exact"]["accuracy"] == 0.5


def test_empty_evaluation_is_defined_and_complete():
    report = evaluate_predictions([], [])
    assert report.complete
    assert report.overall.coverage == 0
    assert report.overall.accuracy == 0
    assert report.overall.auto_accept_precision == 0
    assert report.as_dict()["complete"] is True


@pytest.mark.parametrize(
    "gold_rows,predictions,match",
    [
        ([gold(), gold()], [decision()], "duplicate gold"),
        ([gold()], [decision(), decision()], "duplicate prediction"),
        ([gold()], [decision(source_value="different")], "source value differs"),
    ],
)
def test_invalid_evaluation_join_is_rejected(gold_rows, predictions, match):
    with pytest.raises(EvaluationInputError, match=match):
        evaluate_predictions(gold_rows, predictions)


def test_context_requires_full_evidence_and_exact_declared_fields():
    case = gold(context={"route": "oral", "role_cod": "PS"})
    with pytest.raises(EvaluationInputError, match="cannot be verified"):
        evaluate_predictions([case], [decision()])
    for actual in [{"route": "IV", "role_cod": "PS"}, {"route": "oral"}]:
        prediction = EvaluationPrediction(decision=decision(), context=actual, mapping_key="MAP-1")
        with pytest.raises(EvaluationInputError, match="context differs"):
            evaluate_predictions([case], [prediction])
    prediction = EvaluationPrediction(
        decision=decision(),
        context={"route": "oral", "role_cod": "PS", "extra": "allowed"},
        mapping_key="MAP-1",
    )
    report = evaluate_predictions([case], [prediction])
    assert report.complete and report.overall.context_verified == 1
    assert report.as_dict()["prediction_mapping_keys"] == {"case-1": "MAP-1"}


def test_artifact_prediction_keeps_source_native_and_canonical_context(artifact):
    prediction = EvaluationPrediction.from_artifact(artifact)
    assert prediction.mapping_key == artifact.mapping_key
    assert prediction.context["route"] == "oral"
    assert prediction.decision == artifact.decision
    case = GoldMapping(
        case_id=decision_case_id(artifact.decision),
        source_value=artifact.sau.source.raw_value,
        context={"route": "oral"},
        expected_concept_ids=[value.concept_id for value in artifact.decision.selected_concepts],
    )
    assert evaluate_predictions([case], [prediction]).overall.accuracy == 1


def test_retrieval_recall_uses_only_positive_cases_and_counts_relationship_targets():
    relationship = KnowledgeStatement(
        subject_id="product",
        predicate="has_ingredient",
        object_id="ingredient",
        source_resource="RxNorm",
        authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
    )
    packages = [
        (gold("direct", ["one"]), KnowledgePackage(sau_id="s", candidates=[candidate("one")])),
        (
            gold("related", ["ingredient"]),
            KnowledgePackage(
                sau_id="s", candidates=[candidate("other"), candidate("product", [relationship])]
            ),
        ),
        (gold("miss", ["missing"]), KnowledgePackage(sau_id="s", candidates=[])),
        (gold("negative", []), KnowledgePackage(sau_id="s", candidates=[])),
        (gold("false-positive", []), KnowledgePackage(sau_id="s", candidates=[candidate("wrong")])),
    ]
    metrics = recall_at_k(packages)
    assert metrics.total == 5 and metrics.positive_cases == 3 and metrics.negative_cases == 2
    assert metrics.recall_at_1 == pytest.approx(1 / 3)
    assert metrics.recall_at_5 == pytest.approx(2 / 3)
    assert metrics.retrieval_ceiling == metrics.recall_at_10 == pytest.approx(2 / 3)
    assert metrics.negative_rejections == 1 and metrics.candidate_rejection_specificity == 0.5
    assert metrics.missed == ["same source"]
    empty = recall_at_k([])
    assert empty.recall_at_1 == empty.candidate_rejection_specificity == 0


def test_selective_metrics_count_review_and_abstention_separately():
    results = [
        (gold("a"), decision("a")),
        (gold("b"), decision("b", ids=["wrong"])),
        (gold("c"), decision("c", route=RoutingStatus.HUMAN_REVIEW)),
        (gold("d", []), decision("d", ids=[], route=RoutingStatus.ABSTAIN)),
    ]
    metrics = selective_accuracy(results)
    assert metrics.total == 4 and metrics.auto_accepted == 2
    assert metrics.reviewed == metrics.abstained == 1
    assert metrics.coverage == metrics.auto_accept_precision == 0.5
    empty = selective_accuracy([])
    assert empty.coverage == empty.auto_accept_precision == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"case_id": " "},
        {"case_id": " leading"},
        {"source_value": " "},
        {"expected_concept_ids": [""]},
        {"expected_concept_ids": ["one", "one"]},
    ],
)
def test_gold_validation_rejects_ambiguous_reference_records(updates):
    values = {"case_id": "one", "source_value": "source", "expected_concept_ids": ["one"]}
    values.update(updates)
    with pytest.raises(ValidationError):
        GoldMapping(**values)
