#!/usr/bin/env python3
"""Probe the official RxNav release without modifying committed cassettes.

The canary uses an ephemeral ONLINE cache, evaluates the public example batch,
and writes a compact drift report. A release change is deliberately nonzero:
it asks a maintainer to review the new results and consciously update the
pinned release; it never refreshes fixtures or commits files itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biosemalign import BioSemAlign
from biosemalign.enums import EgressDataClass, ResourceMode
from biosemalign.evaluation import (
    EvaluationPrediction,
    GoldMapping,
    decision_case_id,
    evaluate_predictions,
)
from biosemalign.settings import Settings
from biosemalign.storage.jsonl import read_models

OFFICIAL_RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_canary(
    *,
    expected_release: str,
    gold_path: Path,
    input_path: Path,
) -> dict[str, Any]:
    """Run the public benchmark against the mutable official endpoint."""
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    gold = list(read_models(gold_path, GoldMapping))

    with tempfile.TemporaryDirectory(prefix="biosemalign-rxnorm-canary-") as cache_dir:
        settings = Settings(
            mode=ResourceMode.ONLINE,
            cache_dir=Path(cache_dir),
            rxnav_base_url=OFFICIAL_RXNAV_BASE_URL,
            cache_store_raw_queries=False,
            deployment_environment="local",
            permitted_egress_classes={
                EgressDataClass.PUBLIC_TERMINOLOGY,
                EgressDataClass.SENSITIVE_TEXT,
            },
        )
        with BioSemAlign.from_profile("faers_drug_normalization", settings=settings) as pipeline:
            provider = pipeline.knowledge.primary_provider()
            current_release = provider.terminology_release()
            records = list(
                pipeline.align_batch_records(
                    rows,
                    value_column="prod_ai",
                    id_columns=["primaryid", "drug_seq"],
                    context_columns=["route", "role_cod", "indi_pt"],
                )
            )
            predictions = [
                EvaluationPrediction.from_artifact(record.artifact)
                for record in records
                if record.status == "success" and record.artifact is not None
            ]

    evaluation = evaluate_predictions(gold, predictions)
    expected_by_id = {entry.case_id: set(entry.expected_concept_ids) for entry in gold}
    mapping_drift_ids = sorted(
        decision_case_id(prediction.decision)
        for prediction in predictions
        if decision_case_id(prediction.decision) in expected_by_id
        and {concept.concept_id for concept in prediction.decision.selected_concepts}
        != expected_by_id[decision_case_id(prediction.decision)]
    )
    release_changed = current_release != expected_release
    drifted = release_changed or not evaluation.complete or bool(mapping_drift_ids)
    return {
        "status": "drift" if drifted else "current",
        "checked_at": datetime.now(UTC).isoformat(),
        "endpoint": OFFICIAL_RXNAV_BASE_URL,
        "expected_release": expected_release,
        "current_release": current_release,
        "release_changed": release_changed,
        "mapping_drift_ids": mapping_drift_ids,
        "evaluation": evaluation.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-release", required=True)
    parser.add_argument("--gold", type=Path, default=Path("examples/gold_drug_mappings.jsonl"))
    parser.add_argument("--input", type=Path, default=Path("examples/faers_sample_products.csv"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        payload = run_canary(
            expected_release=args.expected_release,
            gold_path=args.gold,
            input_path=args.input,
        )
    except Exception as exc:
        payload = {
            "status": "error",
            "checked_at": datetime.now(UTC).isoformat(),
            "endpoint": OFFICIAL_RXNAV_BASE_URL,
            "expected_release": args.expected_release,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report(args.report, payload)
        return 2

    _write_report(args.report, payload)
    return int(payload["status"] != "current")


if __name__ == "__main__":
    raise SystemExit(main())
