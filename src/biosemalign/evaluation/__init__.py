"""Component-level evaluation (§22)."""

from biosemalign.evaluation.metrics import (
    CategoryMetrics,
    EvaluationInputError,
    EvaluationPrediction,
    EvaluationReport,
    GoldMapping,
    RetrievalMetrics,
    SelectiveMetrics,
    decision_case_id,
    evaluate_predictions,
    recall_at_k,
    selective_accuracy,
)

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
