"""Component-level evaluation metrics (§22.1).

The framework's central evaluation claim is that retrieval failure and
adjudication failure are separable (§6.3):

    correct concept absent from the candidate list  -> retrieval failure
    correct concept present but not selected        -> adjudication failure

Both are measurable here without an LLM, which is why the evaluation skeleton
ships with the deterministic slice rather than waiting for v0.2.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from biosemalign.enums import RoutingStatus
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentDecision
from biosemalign.schemas.evaluation import GoldMapping
from biosemalign.schemas.knowledge import KnowledgePackage

__all__ = [
    "CategoryMetrics",
    "EvaluationInputError",
    "EvaluationPrediction",
    "EvaluationReport",
    "GoldMapping",
    "RetrievalMetrics",
    "SelectiveMetrics",
    "decision_case_id",
    "evaluate_predictions",
    "recall_at_k",
    "selective_accuracy",
]


class EvaluationInputError(ValueError):
    """The gold and prediction files cannot be joined one-to-one."""


class EvaluationPrediction(BioSemAlignModel):
    """A decision plus artifact evidence needed to verify benchmark context."""

    decision: AlignmentDecision
    context: dict[str, object] | None = None
    mapping_key: str | None = None

    @classmethod
    def from_artifact(cls, artifact: AlignmentArtifact) -> EvaluationPrediction:
        """Retain a complete artifact's canonical and source-native context."""
        # Keep source-native aliases so existing expert datasets can state
        # ``role_cod``/``indi_pt`` while also exposing canonical SAU fields.
        context: dict[str, object] = dict(artifact.sau.source.row)
        envelope = artifact.sau.context.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"extensions", "neighboring_fields"},
        )
        context.update(artifact.sau.context.neighboring_fields)
        context.update(envelope)
        context.update(
            artifact.sau.attributes.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"extensions"},
            )
        )
        return cls(
            decision=artifact.decision,
            context=context,
            mapping_key=artifact.mapping_key,
        )


class RetrievalMetrics(BioSemAlignModel):
    """Positive recall and negative rejection quality, per §22.1."""

    # ``total`` includes both mapping-positive and mapping-negative cases.
    # Recall denominators use only ``positive_cases``: an empty gold concept
    # set asks the retriever to reject candidates and can never be a recall hit.
    total: int
    positive_cases: int
    negative_cases: int
    negative_rejections: int
    candidate_rejection_specificity: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    missed: list[str]

    @property
    def retrieval_ceiling(self) -> float:
        """The best any adjudicator could do given these candidate sets.

        Reporting this next to end-to-end accuracy is what stops a retrieval
        gap from being misread as a reasoning failure.
        """
        return self.recall_at_10


class SelectiveMetrics(BioSemAlignModel):
    """Accuracy and coverage under abstention (§22.1, routing row)."""

    total: int
    auto_accepted: int
    reviewed: int
    abstained: int
    auto_accept_correct: int

    @property
    def coverage(self) -> float:
        """Fraction handled without a human."""
        return self.auto_accepted / self.total if self.total else 0.0

    @property
    def auto_accept_precision(self) -> float:
        """Precision among automatically accepted mappings.

        The quantity the first release optimizes (§9.2): a wrong auto-accept
        silently corrupts an analysis, while an unnecessary review costs a
        reviewer a minute.
        """
        return self.auto_accept_correct / self.auto_accepted if self.auto_accepted else 0.0


class CategoryMetrics(BioSemAlignModel):
    """Accounting and quality metrics for one benchmark slice."""

    gold_cases: int
    evaluated: int
    missing_predictions: int
    auto_accepted: int
    reviewed: int
    abstained: int
    correct_predictions: int
    auto_accept_correct: int
    context_verified: int

    @property
    def coverage(self) -> float:
        """Fraction of all gold cases automatically accepted."""
        return self.auto_accepted / self.gold_cases if self.gold_cases else 0.0

    @property
    def accuracy(self) -> float:
        """Correct predictions divided by all gold cases, including missing ones."""
        return self.correct_predictions / self.gold_cases if self.gold_cases else 0.0

    @property
    def auto_accept_precision(self) -> float:
        return self.auto_accept_correct / self.auto_accepted if self.auto_accepted else 0.0

    def as_dict(self) -> dict[str, int | float]:
        """Return JSON-ready counts plus computed rates."""
        return {
            **self.model_dump(),
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "auto_accept_precision": round(self.auto_accept_precision, 4),
        }


class EvaluationReport(BioSemAlignModel):
    """One-to-one accounting plus aggregate and category-level metrics."""

    prediction_cases: int
    overall: CategoryMetrics
    categories: dict[str, CategoryMetrics]
    missing_prediction_ids: list[str]
    unexpected_prediction_ids: list[str]
    prediction_mapping_keys: dict[str, str]

    @property
    def complete(self) -> bool:
        return not self.missing_prediction_ids and not self.unexpected_prediction_ids

    def as_dict(self) -> dict[str, object]:
        """Return the stable CLI/report representation."""
        return {
            **self.overall.as_dict(),
            "prediction_cases": self.prediction_cases,
            "unexpected_predictions": len(self.unexpected_prediction_ids),
            "missing_prediction_ids": self.missing_prediction_ids,
            "unexpected_prediction_ids": self.unexpected_prediction_ids,
            "prediction_mapping_keys": dict(sorted(self.prediction_mapping_keys.items())),
            "complete": self.complete,
            "categories": {
                name: metrics.as_dict() for name, metrics in sorted(self.categories.items())
            },
        }


def decision_case_id(decision: AlignmentDecision) -> str:
    """Resolve a prediction's stable join key.

    Callers can explicitly attach ``extensions.case_id``. Batch decisions fall
    back to their source ``record_id`` and scalar decisions to the deterministic
    ``sau_id``. Source text is deliberately never used as an identity key.
    """
    explicit = decision.extensions.get("case_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise EvaluationInputError("prediction extensions.case_id must be a non-blank string")
        return explicit
    if decision.record_id and decision.record_id.strip():
        return decision.record_id
    if decision.sau_id.strip():
        return decision.sau_id
    raise EvaluationInputError("prediction has no usable case ID")


def _unique_index[T](
    entries: Iterable[T],
    *,
    key: Callable[[T], str],
    label: str,
) -> dict[str, T]:
    indexed: dict[str, T] = {}
    duplicates: set[str] = set()
    for entry in entries:
        case_id = key(entry)
        if case_id in indexed:
            duplicates.add(case_id)
        else:
            indexed[case_id] = entry
    if duplicates:
        rendered = ", ".join(sorted(duplicates))
        raise EvaluationInputError(f"duplicate {label} case ID(s): {rendered}")
    return indexed


def _category_metrics(
    gold_cases: Sequence[GoldMapping],
    predictions: dict[str, EvaluationPrediction],
) -> CategoryMetrics:
    evaluated = auto = reviewed = abstained = correct = auto_correct = context_verified = 0

    for gold in gold_cases:
        prediction = predictions.get(gold.case_id)
        if prediction is None:
            continue
        decision = prediction.decision
        evaluated += 1
        context_verified += int(bool(gold.context))
        is_correct = {c.concept_id for c in decision.selected_concepts} == set(
            gold.expected_concept_ids
        )
        correct += int(is_correct)
        if decision.routing_status is RoutingStatus.AUTO_ACCEPT:
            auto += 1
            auto_correct += int(is_correct)
        elif decision.routing_status is RoutingStatus.HUMAN_REVIEW:
            reviewed += 1
        else:
            abstained += 1

    return CategoryMetrics(
        gold_cases=len(gold_cases),
        evaluated=evaluated,
        missing_predictions=len(gold_cases) - evaluated,
        auto_accepted=auto,
        reviewed=reviewed,
        abstained=abstained,
        correct_predictions=correct,
        auto_accept_correct=auto_correct,
        context_verified=context_verified,
    )


def evaluate_predictions(
    gold_cases: Iterable[GoldMapping],
    predictions: Iterable[AlignmentDecision | EvaluationPrediction],
) -> EvaluationReport:
    """Evaluate predictions with exact one-to-one case accounting.

    Duplicate identities and source-value disagreement are invalid inputs.
    Missing and unexpected cases remain explicit in the returned report so a
    partial output cannot look like a successful benchmark run.
    """
    gold_list = list(gold_cases)
    prediction_list = [
        item if isinstance(item, EvaluationPrediction) else EvaluationPrediction(decision=item)
        for item in predictions
    ]
    gold_by_id = _unique_index(gold_list, key=lambda item: item.case_id, label="gold")
    prediction_by_id = _unique_index(
        prediction_list,
        key=lambda item: decision_case_id(item.decision),
        label="prediction",
    )

    shared_ids = gold_by_id.keys() & prediction_by_id.keys()
    mismatches = [
        case_id
        for case_id in shared_ids
        if prediction_by_id[case_id].decision.source_value is not None
        and prediction_by_id[case_id].decision.source_value != gold_by_id[case_id].source_value
    ]
    if mismatches:
        rendered = ", ".join(sorted(mismatches))
        raise EvaluationInputError(f"source value differs for case ID(s): {rendered}")

    unavailable_context = [
        case_id
        for case_id in shared_ids
        if gold_by_id[case_id].context
        and (prediction_by_id[case_id].context is None or not prediction_by_id[case_id].mapping_key)
    ]
    if unavailable_context:
        rendered = ", ".join(sorted(unavailable_context))
        raise EvaluationInputError(
            "gold context cannot be verified for case ID(s): "
            f"{rendered}; use complete AlignmentArtifact predictions"
        )

    context_mismatches: dict[str, list[str]] = {}
    for case_id in shared_ids:
        expected_context = gold_by_id[case_id].context
        actual_context = prediction_by_id[case_id].context
        if not expected_context or actual_context is None:
            continue
        differing_fields = sorted(
            key
            for key, expected_value in expected_context.items()
            if key not in actual_context or actual_context[key] != expected_value
        )
        if differing_fields:
            context_mismatches[case_id] = differing_fields
    if context_mismatches:
        rendered = "; ".join(
            f"{case_id}: {', '.join(fields)}"
            for case_id, fields in sorted(context_mismatches.items())
        )
        raise EvaluationInputError(f"gold context differs from artifact context ({rendered})")

    missing_ids = sorted(gold_by_id.keys() - prediction_by_id.keys())
    unexpected_ids = sorted(prediction_by_id.keys() - gold_by_id.keys())
    grouped: dict[str, list[GoldMapping]] = {}
    for gold in gold_list:
        grouped.setdefault(gold.category or "uncategorized", []).append(gold)

    return EvaluationReport(
        prediction_cases=len(prediction_list),
        overall=_category_metrics(gold_list, prediction_by_id),
        categories={
            category: _category_metrics(cases, prediction_by_id)
            for category, cases in grouped.items()
        },
        missing_prediction_ids=missing_ids,
        unexpected_prediction_ids=unexpected_ids,
        prediction_mapping_keys={
            case_id: prediction.mapping_key
            for case_id, prediction in prediction_by_id.items()
            if prediction.mapping_key is not None
        },
    )


def recall_at_k(
    packages: Iterable[tuple[GoldMapping, KnowledgePackage]],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Measure positive recall and rejection specificity over candidate sets.

    A concept reachable from a candidate's relationships counts as retrieved:
    ingredient-level mapping legitimately walks from a retrieved product to its
    ingredient, and scoring that as a retrieval miss would understate the
    retriever. Cases with no expected concept are negative controls: they are
    excluded from recall and count as correctly rejected only when the retriever
    returns no candidates.
    """
    hits = Counter[int]()
    missed: list[str] = []
    total = 0
    positive_cases = 0
    negative_cases = 0
    negative_rejections = 0

    for gold, package in packages:
        total += 1
        expected = set(gold.expected_concept_ids)
        if not expected:
            negative_cases += 1
            negative_rejections += int(not package.candidates)
            continue

        positive_cases += 1
        found_at: int | None = None

        for position, candidate in enumerate(package.candidates, start=1):
            reachable = {candidate.candidate_id} | {s.object_id for s in candidate.relationships}
            if expected & reachable:
                found_at = position
                break

        if found_at is None:
            missed.append(gold.source_value)
            continue
        for k in ks:
            if found_at <= k:
                hits[k] += 1

    def rate(k: int) -> float:
        return hits[k] / positive_cases if positive_cases else 0.0

    return RetrievalMetrics(
        total=total,
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        negative_rejections=negative_rejections,
        candidate_rejection_specificity=(
            negative_rejections / negative_cases if negative_cases else 0.0
        ),
        recall_at_1=rate(1),
        recall_at_5=rate(5),
        recall_at_10=rate(10),
        missed=missed,
    )


def selective_accuracy(
    results: Iterable[tuple[GoldMapping, AlignmentDecision]],
) -> SelectiveMetrics:
    """Coverage and precision under the routing policy."""
    total = auto = reviewed = abstained = correct = 0

    for gold, decision in results:
        total += 1
        selected = {c.concept_id for c in decision.selected_concepts}
        is_correct = selected == set(gold.expected_concept_ids)

        if decision.routing_status is RoutingStatus.AUTO_ACCEPT:
            auto += 1
            correct += int(is_correct)
        elif decision.routing_status is RoutingStatus.HUMAN_REVIEW:
            reviewed += 1
        else:
            abstained += 1

    return SelectiveMetrics(
        total=total,
        auto_accepted=auto,
        reviewed=reviewed,
        abstained=abstained,
        auto_accept_correct=correct,
    )
