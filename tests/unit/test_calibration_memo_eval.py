"""M4: per-tier evaluation, calibrated acceptance, and the persistent crosswalk memo."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.support import ALPHA, MISSPELLING, NO_MATCH, PRODUCT, canonicalize, make_pipeline
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.enums import RoutingStatus
from biosemalign.evaluation.metrics import (
    EvaluationPrediction,
    evaluate_predictions,
    resolving_tier,
)
from biosemalign.exceptions import ConfigurationError
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.routing.conformal import (
    POOLED,
    Calibration,
    TierThreshold,
    _fit,
    acceptance_score,
    binomial_cdf,
    calibrate,
    case_digest,
    clopper_pearson_upper,
)
from biosemalign.schemas.evaluation import GoldMapping
from biosemalign.storage.crosswalk_memo import CrosswalkMemo
from biosemalign.storage.jsonl import write_models


def two_tier_profile(**routing: Any) -> TaskProfile:
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["tiers"] = [
        {
            "name": "exact",
            "match": [{"op": "drug.rxnav", "options": {"channels": ["normalized_lookup"]}}],
            "resolve": "rules",
            "pass_on": ["NO_VALID_MAPPING"],
        },
        {"name": "full", "match": ["drug.rxnav"]},
    ]
    payload["routing"].update(routing)
    return TaskProfile.model_validate(payload)


def _rows(values: list[str]) -> list[dict[str, Any]]:
    return [{"drug": value, "case": f"case-{index}"} for index, value in enumerate(values)]


def _records(pipeline: BioSemAlign, values: list[str]) -> list[Any]:
    return list(
        pipeline.align_batch_records(_rows(values), value_column="drug", id_columns=["case"])
    )


def _gold(values: list[str], expected: list[list[str]], **extra: Any) -> list[GoldMapping]:
    return [
        GoldMapping(
            case_id=f"case-{index}",
            source_value=value,
            expected_concept_ids=ids,
            **{key: column[index] for key, column in extra.items()},
        )
        for index, (value, ids) in enumerate(zip(values, expected, strict=True))
    ]


# -- evaluation by tier and stratum -----------------------------------------


def test_evaluation_attributes_results_to_tiers(tmp_path: Path) -> None:
    values = [ALPHA, MISSPELLING, NO_MATCH]
    with make_pipeline(tmp_path, profile=two_tier_profile()) as pipeline:
        records = _records(pipeline, values)
    predictions = [EvaluationPrediction.from_artifact(r.artifact) for r in records]
    report = evaluate_predictions(
        _gold(values, [["1001"], ["1001"], []], stratum=["outcome", "no_outcome", "no_outcome"]),
        predictions,
    )
    assert set(report.tiers) == {"exact", "full"}
    assert report.tiers["exact"].auto_accepted == 1
    assert [(t.tier, t.auto_accepted, t.auto_accept_correct) for t in report.cumulative_tiers] == [
        ("exact", 1, 1),
        ("full", 2, 2),
    ]
    assert report.cumulative_tiers[0].coverage == pytest.approx(1 / 3)
    payload = report.as_dict()
    assert payload["cumulative_tiers"][1]["auto_accept_precision"] == 1.0
    assert set(report.strata) == {"outcome", "no_outcome"}
    assert payload["stratum_disparity"]["coverage"] == pytest.approx(0.5)


def test_decisions_without_a_trace_are_unattributed(synthetic_pipeline: BioSemAlign) -> None:
    run = synthetic_pipeline.new_run()
    sau = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(ALPHA), run)[0]
    package = synthetic_pipeline.build_package(sau)
    artifact = synthetic_pipeline._align_with_package(sau, package, run)
    assert resolving_tier(artifact.decision) == "unattributed"
    report = evaluate_predictions(
        [
            GoldMapping(
                case_id=artifact.decision.sau_id, source_value=ALPHA, expected_concept_ids=["1001"]
            )
        ],
        [artifact.decision],
    )
    # Without a trace for every decision, cumulative attribution is not reported.
    assert report.cumulative_tiers == [] and report.stratum_disparity is None


# -- the statistics ----------------------------------------------------------


def test_binomial_tail_and_exact_bounds() -> None:
    assert binomial_cdf(0, 10, 0.1) == pytest.approx(0.9**10)
    assert binomial_cdf(10, 10, 0.3) == 1.0 and binomial_cdf(-1, 10, 0.3) == 0.0
    assert binomial_cdf(2, 10, 0.0) == 1.0 and binomial_cdf(2, 10, 1.0) == 0.0
    # With zero errors the exact bound has a closed form: 1 - delta ** (1 / n).
    assert clopper_pearson_upper(0, 45, 0.1) == pytest.approx(1 - 0.1 ** (1 / 45), abs=1e-5)
    assert clopper_pearson_upper(3, 3, 0.1) == 1.0 and clopper_pearson_upper(0, 0, 0.1) == 1.0


def test_fitting_certifies_only_what_the_evidence_supports() -> None:
    clean = [(0.9, True)] * 200 + [(0.4, False)] * 20
    certified = _fit(clean, alpha=0.05, delta=0.1)
    assert certified.threshold == 0.45  # every threshold above the errors is certified
    assert (certified.accepted, certified.errors) == (200, 0)
    assert certified.error_upper_bound is not None and certified.error_upper_bound < 0.05
    assert _fit([(0.9, True)] * 10, alpha=0.05, delta=0.1).threshold is None
    noisy = [(0.9, index % 5 != 0) for index in range(500)]
    assert _fit(noisy, alpha=0.05, delta=0.1).threshold is None
    assert _fit([], alpha=0.05, delta=0.1).threshold is None


def _calibration(threshold: float | None, **tiers: TierThreshold) -> Calibration:
    thresholds = {
        POOLED: TierThreshold(threshold=threshold, calibration_cases=100, accepted=100, errors=0)
    }
    thresholds.update(tiers)
    return Calibration(
        alpha=0.05,
        delta=0.1,
        min_cases=30,
        gold_sha256="0" * 64,
        case_digests=[case_digest("held-out")],
        thresholds=thresholds,
        created_at=datetime(2026, 11, 18, tzinfo=UTC),
    )


def test_calibrated_routing_gates_eligible_acceptances(tmp_path: Path) -> None:
    with make_pipeline(tmp_path / "plain") as plain:
        score = acceptance_score(plain.align_value(ALPHA).routing_features)
    strict = _calibration(round(score + 0.05, 2))
    pipeline = make_pipeline(tmp_path / "strict")
    pipeline.decision_router = type(pipeline.decision_router)(pipeline.profile, calibration=strict)
    pipeline.kernel.router = pipeline.decision_router
    with pipeline:
        gated = pipeline.align_artifact(pipeline.make_request(ALPHA))
    assert gated.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert "below the calibrated threshold" in gated.decision.routing_reasons[0]
    assert gated.decision.provenance.routing_policy_version.endswith(
        f"+calibration:{strict.digest}"
    )
    assert verify_artifact(gated).valid

    path = tmp_path / "calibration.json"
    path.write_text(_calibration(None).model_dump_json())
    profile = load_profile("faers_drug_normalization")
    profile.routing.calibration = str(path)
    with make_pipeline(
        tmp_path / "none", profile=TaskProfile.model_validate(profile.model_dump())
    ) as uncertified:
        decision = uncertified.align_value(ALPHA)
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert "no automatic acceptance is certified" in decision.routing_reasons[0]

    lenient = _calibration(
        0.0, exact=TierThreshold(threshold=0.0, calibration_cases=40, accepted=40, errors=0)
    )
    with BioSemAlign(
        two_tier_profile(),
        settings=pipeline.settings,
        cache=pipeline.knowledge.cache,
        calibration=lenient,
    ) as calibrated:
        accepted = calibrated.align_value(ALPHA)
    assert accepted.routing_status is RoutingStatus.AUTO_ACCEPT
    assert "meets the calibrated threshold 0.00 for tier 'exact'" in accepted.routing_reasons[-1]


def test_calibrate_fits_from_an_uncalibrated_run(tmp_path: Path) -> None:
    values = [ALPHA] * 60 + [MISSPELLING] * 60 + [NO_MATCH] * 5
    with make_pipeline(tmp_path, profile=two_tier_profile()) as pipeline:
        decisions = [r.artifact.decision for r in _records(pipeline, values)]
    gold = _gold(values, [["1001"] if v != NO_MATCH else [] for v in values])
    fitted = calibrate(decisions, gold, alpha=0.1, delta=0.1, min_cases=30)
    assert set(fitted.thresholds) == {POOLED, "exact", "full"}
    assert fitted.thresholds[POOLED].calibration_cases == 120
    assert fitted.thresholds[POOLED].threshold is not None
    assert len(fitted.case_digests) == len(values)

    gated = decisions[0].model_copy(
        update={
            "provenance": decisions[0].provenance.model_copy(
                update={"routing_policy_version": "0.2.0+calibration:abc"}
            )
        }
    )
    with pytest.raises(ConfigurationError, match="uncalibrated run"):
        calibrate([gated], gold, alpha=0.1, delta=0.1)
    with pytest.raises(ConfigurationError, match="duplicate prediction"):
        calibrate([decisions[0], decisions[0]], gold, alpha=0.1, delta=0.1)
    with pytest.raises(ConfigurationError, match="duplicate case IDs"):
        calibrate(decisions, gold + gold[:1], alpha=0.1, delta=0.1)


def test_calibration_files_are_versioned(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    payload = json.loads(_calibration(0.5).model_dump_json())
    payload["score_version"] = "0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigurationError, match="refit it"):
        Calibration.load(path)
    with pytest.raises(ConfigurationError, match="cannot read calibration"):
        Calibration.load(tmp_path / "missing.json")


def test_calibrate_and_evaluate_commands_keep_splits_disjoint(tmp_path: Path) -> None:
    # Error-free cases certify only past a size the exact test demands: with
    # alpha 0.1 and the Bonferroni-split delta, 51 is the minimum; 50 is not.
    values = [ALPHA] * 60
    with make_pipeline(tmp_path / "run") as pipeline:
        records = _records(pipeline, values)
    write_models(tmp_path / "records.jsonl", records)
    write_models(tmp_path / "gold.jsonl", _gold(values, [["1001"]] * 60))
    runner = CliRunner()
    fitted = runner.invoke(
        app,
        [
            "calibrate",
            "--gold",
            str(tmp_path / "gold.jsonl"),
            "--predictions",
            str(tmp_path / "records.jsonl"),
            "-o",
            str(tmp_path / "calibration.json"),
            "--alpha",
            "0.1",
        ],
    )
    assert fitted.exit_code == 0, fitted.output
    assert "*: threshold" in fitted.output and "error rate <=" in fitted.output

    overlap = runner.invoke(
        app,
        [
            "evaluate",
            "--gold",
            str(tmp_path / "gold.jsonl"),
            "--predictions",
            str(tmp_path / "records.jsonl"),
            "--calibration",
            str(tmp_path / "calibration.json"),
        ],
    )
    assert overlap.exit_code == 2 and "used to fit the calibration" in overlap.output

    few = runner.invoke(
        app,
        [
            "calibrate",
            "--gold",
            str(tmp_path / "gold.jsonl"),
            "--predictions",
            str(tmp_path / "records.jsonl"),
            "-o",
            str(tmp_path / "strict.json"),
            "--alpha",
            "0.001",
        ],
    )
    assert few.exit_code == 0 and "nothing certified" in few.output
    bad = runner.invoke(
        app,
        [
            "calibrate",
            "--gold",
            str(tmp_path / "gold.jsonl"),
            "--predictions",
            str(tmp_path / "records.jsonl"),
            "-o",
            str(tmp_path / "x.json"),
            "--alpha",
            "2",
        ],
    )
    assert bad.exit_code == 2


# -- the persistent memo -----------------------------------------------------


def test_memo_reuses_identical_inputs_across_runs(tmp_path: Path) -> None:
    values = [ALPHA, PRODUCT, NO_MATCH, ALPHA]
    memo_path = tmp_path / "memo.sqlite"
    with make_pipeline(tmp_path / "plain") as plain:
        expected = _records(plain, values)
    with make_pipeline(tmp_path / "first") as first:
        first.attach_memo(memo_path)
        initial = _records(first, values)
        assert first.kernel.memo_hits == 1  # the repeated ExampleAlpha
    with make_pipeline(tmp_path / "second") as second:
        memo = second.attach_memo(memo_path)
        assert len(memo) == 3
        replayed = _records(second, values)
        assert second.kernel.memo_hits == 4
    for fresh, reused in zip(initial, replayed, strict=True):
        assert reused.artifact.decision.provenance.memo is not None
        assert reused.artifact.decision.provenance.memo.original_run_id == fresh.artifact.run_id
        assert verify_artifact(reused.artifact).valid
        for field in ("decision", "selected_concepts", "relationship_to_source", "routing_status"):
            assert getattr(reused.artifact.decision, field) == getattr(
                fresh.artifact.decision, field
            )
    strip = {"provenance"}
    assert [canonicalize(r.artifact.decision.model_dump(exclude=strip)) for r in replayed] == [
        canonicalize(r.artifact.decision.model_dump(exclude=strip)) for r in expected
    ]


def test_memo_keys_on_exact_inputs_and_configuration(tmp_path: Path) -> None:
    memo_path = tmp_path / "memo.sqlite"
    with make_pipeline(tmp_path / "a") as pipeline:
        pipeline.attach_memo(memo_path)
        pipeline.align_value(ALPHA)
        pipeline.align_value("examplealpha")  # same mapping key, different text
        pipeline.align_value(ALPHA, route="oral")
        assert pipeline.kernel.memo_hits == 0
    shared = CrosswalkMemo(memo_path)
    with make_pipeline(tmp_path / "b", profile=two_tier_profile()) as cascade:
        cascade.attach_memo(shared)
        cascade.align_value(ALPHA)
        assert cascade.kernel.memo_hits == 0  # another workflow never reuses
    # A memo the caller supplied stays open for the caller to close.
    assert len(shared) == 4
    shared.close()


def test_memo_requires_pinned_releases(tmp_path: Path) -> None:
    with make_pipeline(tmp_path) as pipeline:
        pipeline.settings = pipeline.settings.model_copy(update={"rxnav_terminology_release": None})
        with pytest.raises(ConfigurationError, match="pinned RxNorm release"):
            pipeline.attach_memo(tmp_path / "memo.sqlite")


def test_batch_command_records_memo_hits_and_tier_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biosemalign import cli

    monkeypatch.setattr(cli, "_pipeline", lambda *args, **kwargs: make_pipeline(tmp_path / "cache"))
    source = tmp_path / "input.csv"
    source.write_text("drug\nExampleAlpha\nExampleAlpha\nUNMAPPABLE EXAMPLE\n")
    runner = CliRunner()
    for run in ("one", "two"):
        result = runner.invoke(
            app,
            [
                "align-batch",
                "-p",
                "faers_drug_normalization",
                "-i",
                str(source),
                "-o",
                str(tmp_path / f"{run}.csv"),
                "--value-column",
                "drug",
                "--memo",
                str(tmp_path / "memo.sqlite"),
            ],
        )
        assert result.exit_code == 0, result.output
    first = json.loads((tmp_path / "one.manifest.json").read_text())
    second = json.loads((tmp_path / "two.manifest.json").read_text())
    assert (first["memo_hits"], second["memo_hits"]) == (1, 3)
    timing = first["tier_timings"]["default"]
    assert (timing["attempts"], timing["resolved"]) == (2, 2)
    assert math.isfinite(timing["seconds"])
    assert second["tier_timings"]["default"]["attempts"] == 0
    assert first["workflow"]["tiers"][0]["name"] == "default"
