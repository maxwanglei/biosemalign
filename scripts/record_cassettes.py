#!/usr/bin/env python3
"""Record the provider cassettes the test suite replays.

The test suite runs in ``OFFLINE`` mode, where a cache miss is an error rather
than a network call. That makes the suite hermetic and fast, at the cost of
needing every provider call recorded in advance — including the exact parameter
combinations, since ``max_entries`` and the requested predicate set are part of
the cache key.

Usage::

    python scripts/record_cassettes.py            # fill in what is missing
    python scripts/record_cassettes.py --refresh  # re-fetch everything

A diff in ``tests/cassettes/`` after ``--refresh`` means RxNorm changed
upstream. That is the reproducibility guard working, not a problem with the
script: review the diff before committing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CASSETTES = REPO_ROOT / "tests" / "cassettes"

#: Values exercised end to end by the golden suite. Each is a category from the
#: difficult-name taxonomy in §22.4 rather than an arbitrary example.
GOLDEN_VALUES = [
    "TOPROL XL 50 MG",  # branded product with strength; obsolete-atom trap
    "LISINOPRIL-HCTZ",  # combination product
    "metoprolol",  # exact generic name
    "metoprolol tartrate",  # salt form
    "LITHIUM CARB ER 450 MG",  # release formulation with strength
    "Tylenol",  # brand name
    "acetaminophen 500 mg",  # strength-bearing generic
    "amoxicilin",  # misspelling
    "ZZZQQQ NOT A DRUG 999",  # no valid mapping
    "aspirin",  # exact generic name
]

#: Direct provider calls the contract suite makes with their own parameters.
CONTRACT_CONCEPTS = ["6918", "220348", "99999999", "866438"]
CONTRACT_APPROXIMATE = [("TOPROL XL 50 MG", 4)]
CONTRACT_PREDICATE_SETS = [
    ["has_ingredient"],
    ["has_ingredient", "has_precise_ingredient"],
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch every response instead of filling in only what is missing",
    )
    parser.add_argument("--cassettes", default=str(CASSETTES))
    args = parser.parse_args(argv)

    from biosemalign import BioSemAlign
    from biosemalign.enums import EgressDataClass, ResourceMode
    from biosemalign.knowledge.providers.rxnorm import RxNavProvider
    from biosemalign.settings import Settings
    from biosemalign.storage.cache import ResponseCache

    mode = ResourceMode.ONLINE if args.refresh else ResourceMode.CACHE_PREFER
    cache = ResponseCache(args.cassettes, mode=mode)
    permitted_egress = {
        EgressDataClass.PUBLIC_TERMINOLOGY,
        EgressDataClass.SENSITIVE_TEXT,
    }
    settings = Settings(
        mode=mode,
        cache_dir=Path(args.cassettes),
        deployment_environment="local",
        permitted_egress_classes=permitted_egress,
    )

    print(f"recording into {args.cassettes} (mode={mode.value})")

    before = len(list(cache.entries()))

    # --- end-to-end pipeline calls ---
    with BioSemAlign.from_profile(
        "faers_drug_normalization", cache=cache, settings=settings
    ) as pipeline:
        release = pipeline.knowledge.primary_provider().terminology_release()
        print(f"RxNorm release: {release}")
        for value in GOLDEN_VALUES:
            decision = pipeline.align_value(value)
            selected = ", ".join(c.concept_id for c in decision.selected_concepts) or "-"
            print(f"  {value:<24} -> {decision.routing_status.value:<13} [{selected}]")

    # --- direct provider calls made only by the contract suite ---
    with RxNavProvider(
        base_url="https://rxnav.nlm.nih.gov/REST",
        cache=cache,
        permitted_egress_classes=permitted_egress,
    ) as provider:
        provider.terminology_release()
        for concept_id in CONTRACT_CONCEPTS:
            provider.lookup(concept_id)
            provider.validate_concept(concept_id)
        provider.find_by_name("metoprolol")
        provider.find_by_name("TOPROL XL 50 MG")
        for term, max_entries in CONTRACT_APPROXIMATE:
            provider.approximate(term, max_entries=max_entries, current_only=True)
            provider.approximate(term, max_entries=max_entries, current_only=False)
        provider.search("metoprolol", top_k=5)
        for predicates in CONTRACT_PREDICATE_SETS:
            provider.get_relationships("866438", predicates=predicates)
            provider.get_relationships("220348", predicates=predicates)

    after = len(list(cache.entries()))
    print(f"\ncassettes: {after} ({after - before:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
