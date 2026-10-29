"""Run the invented baseline from a checkout: python examples/offline_baseline.py.

Install the package first (pip install -e .). This example reuses the checkout's
synthetic fixture definitions, requires no pytest import, and never uses a live
terminology endpoint. Its IDs and names are not real RxNorm data.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.support import SYNTHETIC_RELEASE, baseline_rows, make_pipeline

    from biosemalign.input.adapters.existing import ExistingCleanerAdapter

    with TemporaryDirectory(prefix="biosemalign-synthetic-") as directory:
        with make_pipeline(Path(directory)) as pipeline:
            records = list(
                ExistingCleanerAdapter(pipeline).clean_records(
                    baseline_rows(),
                    value_column="drugname",
                    id_columns=["record_id"],
                    context_columns=["route"],
                )
            )
        print(
            json.dumps(
                {
                    "fixture": "invented offline responses; no terminology or patient data",
                    "release": SYNTHETIC_RELEASE,
                    "input_rows": len(records),
                    "outcomes": dict(Counter(record.status for record in records)),
                    "network_calls": sum(
                        resource.network_calls
                        for record in records
                        for resource in record.resource_provenance
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
