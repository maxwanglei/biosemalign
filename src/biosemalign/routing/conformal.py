"""Calibrated automatic acceptance with a finite-sample error guarantee (v0.2).

The prespecified routing rules decide what is *eligible* for automatic
acceptance. Calibration adds one gate on top: an eligible decision is accepted
only if its acceptance score clears a threshold chosen on held-out,
expert-labelled cases so that, with probability at least ``1 - delta``, the
error rate among accepted decisions is at most ``alpha``.

The procedure is Learn-then-Test (Angelopoulos, Bates et al., 2021) with a
Bonferroni correction: every threshold on a grid fixed in advance is tested
for the null hypothesis "error rate above alpha" with an exact binomial
p-value at level ``delta / len(grid)``, and the most permissive rejected
threshold is used. All rejections hold simultaneously with probability at
least ``1 - delta``, so the chosen one does too. (Fixed-sequence testing,
the other common choice, must stop at its first non-rejection — which, on the
strictest thresholds where only a handful of cases qualify, is almost always
immediately.) Nothing here is asymptotic: a small calibration set yields an
honest answer, often "nothing can be certified", rather than an optimistic one.

The guarantee assumes calibration and deployment cases are exchangeable.
Thresholds are therefore fitted per tier when a tier has enough cases, and
the bound actually achieved is reported with the counts behind it.

The score is deliberately simple and fixed in advance; calibration, not the
score, supplies the guarantee.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from biosemalign.enums import ConcordanceState, RoutingStatus
from biosemalign.exceptions import ConfigurationError
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentDecision, RoutingFeatures
from biosemalign.schemas.evaluation import GoldMapping
from biosemalign.util import stable_digest

__all__ = [
    "CALIBRATION_VERSION",
    "POOLED",
    "SCORE_VERSION",
    "Calibration",
    "TierThreshold",
    "acceptance_score",
    "binomial_cdf",
    "calibrate",
    "case_digest",
    "clopper_pearson_upper",
]

CALIBRATION_VERSION = "1"
SCORE_VERSION = "1"
POOLED = "*"

#: Candidate thresholds, fixed in advance; coarse, because the Bonferroni
#: correction divides the error budget among them.
_GRID = [round(1.0 - step / 20, 2) for step in range(21)]


def acceptance_score(
    features: RoutingFeatures, concordance_state: ConcordanceState | None = None
) -> float:
    """How strongly the evidence supports a decision, in [0, 1]."""
    lexical = features.lexical_similarity or 0.0
    exact = 1.0 if features.exact_terminology_match or features.official_synonym_match else 0.0
    agreement = min(features.channel_agreement, 2) / 2
    concordant = 1.0 if concordance_state is ConcordanceState.MULTI_SOURCE_CONCORDANT else 0.0
    return round(0.5 * lexical + 0.3 * exact + 0.1 * agreement + 0.1 * concordant, 6)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, computed in log space."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(term - peak) for term in terms))


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """Exact one-sided ``1 - delta`` upper confidence bound on an error rate."""
    if n == 0 or errors >= n:
        return 1.0
    low, high = errors / n, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if binomial_cdf(errors, n, middle) > delta:
            low = middle
        else:
            high = middle
    return round(high, 6)


def case_digest(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


class TierThreshold(BioSemAlignModel):
    """The certified threshold for one tier, with the evidence behind it."""

    threshold: float | None = Field(
        description="Minimum acceptance score; None means nothing could be certified."
    )
    calibration_cases: int = Field(ge=0, description="Eligible calibration decisions.")
    accepted: int = Field(ge=0, description="Calibration decisions at or above the threshold.")
    errors: int = Field(ge=0, description="Incorrect decisions among those accepted.")
    error_upper_bound: float | None = Field(
        default=None,
        description=(
            "Clopper-Pearson upper bound on the accepted error rate, at the same "
            "simultaneous confidence the threshold was certified with."
        ),
    )


class Calibration(BioSemAlignModel):
    """A fitted, versioned acceptance calibration."""

    calibration_version: str = CALIBRATION_VERSION
    method: str = "learn_then_test_bonferroni"
    score_version: str = SCORE_VERSION
    alpha: float = Field(gt=0.0, lt=1.0)
    delta: float = Field(gt=0.0, lt=1.0)
    min_cases: int = Field(ge=1)
    gold_sha256: str
    case_digests: list[str] = Field(
        description="SHA-256 of every calibration case ID, to refuse overlapping evaluation."
    )
    thresholds: dict[str, TierThreshold]
    created_at: datetime

    @property
    def digest(self) -> str:
        return stable_digest(self.model_dump(mode="json", exclude={"created_at"}), length=12)

    def threshold_for(self, tier: str | None) -> tuple[str, TierThreshold]:
        """The tier's own threshold when it had enough cases, else the pooled one."""
        if tier is not None and tier in self.thresholds:
            return tier, self.thresholds[tier]
        return POOLED, self.thresholds[POOLED]

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        try:
            calibration = cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot read calibration {path}: {exc}") from exc
        if calibration.score_version != SCORE_VERSION:
            raise ConfigurationError(
                f"calibration {path} was fitted for acceptance score {calibration.score_version}, "
                f"but this release computes score {SCORE_VERSION}; refit it"
            )
        return calibration


def _fit(examples: list[tuple[float, bool]], alpha: float, delta: float) -> TierThreshold:
    level = delta / len(_GRID)
    certified: list[tuple[float, int, int]] = []
    for threshold in _GRID:
        accepted = [correct for score, correct in examples if score >= threshold]
        if not accepted:
            continue
        errors = len(accepted) - sum(accepted)
        # Reject "error rate > alpha" only when this few errors would be unlikely under it.
        if binomial_cdf(errors, len(accepted), alpha) <= level:
            certified.append((threshold, len(accepted), errors))
    if not certified:
        return TierThreshold(threshold=None, calibration_cases=len(examples), accepted=0, errors=0)
    threshold, accepted_count, errors = min(certified)
    return TierThreshold(
        threshold=threshold,
        calibration_cases=len(examples),
        accepted=accepted_count,
        errors=errors,
        error_upper_bound=clopper_pearson_upper(errors, accepted_count, level),
    )


def calibrate(
    decisions: Iterable[AlignmentDecision],
    gold: Iterable[GoldMapping],
    *,
    alpha: float,
    delta: float,
    min_cases: int = 30,
) -> Calibration:
    """Fit thresholds from decisions of an uncalibrated run on labelled cases."""
    from biosemalign.evaluation.metrics import decision_case_id, is_correct, resolving_tier

    gold_list = list(gold)
    gold_by_id = {case.case_id: case for case in gold_list}
    if len(gold_by_id) != len(gold_list):
        raise ConfigurationError("calibration gold has duplicate case IDs")
    by_tier: dict[str, list[tuple[float, bool]]] = {}
    pooled: list[tuple[float, bool]] = []
    seen: set[str] = set()
    for decision in decisions:
        if "+calibration:" in (decision.provenance.routing_policy_version or ""):
            raise ConfigurationError(
                "calibrate from an uncalibrated run: these decisions were already gated by a "
                "calibration, which would bias the fit"
            )
        identifier = decision_case_id(decision)
        if identifier in seen:
            raise ConfigurationError(f"duplicate prediction for calibration case {identifier}")
        seen.add(identifier)
        case = gold_by_id.get(identifier)
        if (
            case is None
            or decision.routing_status is not RoutingStatus.AUTO_ACCEPT
            or decision.routing_features is None
        ):
            continue
        example = (
            acceptance_score(decision.routing_features, decision.concordance_state),
            is_correct(decision, case),
        )
        pooled.append(example)
        by_tier.setdefault(resolving_tier(decision), []).append(example)

    thresholds = {POOLED: _fit(pooled, alpha, delta)}
    for tier, examples in sorted(by_tier.items()):
        if len(examples) >= min_cases:
            thresholds[tier] = _fit(examples, alpha, delta)
    return Calibration(
        alpha=alpha,
        delta=delta,
        min_cases=min_cases,
        gold_sha256=stable_digest([case.model_dump(mode="json") for case in gold_list]),
        case_digests=sorted(case_digest(case.case_id) for case in gold_list),
        thresholds=thresholds,
        created_at=datetime.now(UTC),
    )
