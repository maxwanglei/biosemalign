"""Small, invented RxNav response universe for hermetic regression tests.

Names, identifiers and release strings below are synthetic. They make no claim
about RxNorm and must never be sent to a terminology endpoint as real examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from biosemalign import BioSemAlign
from biosemalign.api import BatchInputError
from biosemalign.domains.drug.providers.rxnav import PREDICATE_TO_TTY
from biosemalign.enums import ResourceMode
from biosemalign.modules import Resolver
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.settings import Settings
from biosemalign.storage.cache import CacheKey, ResponseCache

SYNTHETIC_RELEASE = "synthetic-rxnorm-v1"
ENDPOINT = "https://rxnav.nlm.nih.gov/REST"
ALPHA = "ExampleAlpha"
BETA = "ExampleBeta"
SALT = "ExampleAlpha sodium"
PRODUCT = "ExampleAlpha sodium 10 MG Oral Tablet [ExampleBrand]"
COMBINATION = "ExampleAlpha / ExampleBeta"
BRAND = "ExampleBrand"
NO_MATCH = "UNMAPPABLE EXAMPLE"
HISTORICAL = "ExampleRetired"
UNSUPPORTED = "ExampleUnsupported 10 MG Oral Tablet"
AMBIGUOUS = "ExampleAmbiguous"
MISSPELLING = "ExampelAlpha"
REJECTED = "ZZZQQQ 999"

# The numeric-looking values are invented, local fixture IDs, not real RxCUIs.
CONCEPTS: dict[str, tuple[str, str, str]] = {
    "1001": (ALPHA, "IN", "Active"),
    "1002": (BETA, "IN", "Active"),
    "1101": (SALT, "PIN", "Active"),
    "1102": ("ExampleAlpha hydrochloride", "PIN", "Active"),
    "2001": (PRODUCT, "SBD", "Active"),
    "2002": (COMBINATION, "MIN", "Active"),
    "3001": (BRAND, "BN", "Active"),
    "3002": (AMBIGUOUS, "BN", "Active"),
    "4001": ("Oral Tablet", "DF", "Active"),
    "5001": ("ExampleAlpha sodium 10 MG", "SCDC", "Active"),
    "6001": (HISTORICAL, "SBD", "NotCurrent"),
    "7001": (UNSUPPORTED, "SCD", "Active"),
    "8001": ("Unrelated", "IN", "Active"),
    "9001": ("ExampleRemapped", "SBD", "Remapped"),
}
RELATIONSHIPS: dict[str, dict[str, list[str]]] = {
    "1101": {"has_ingredient": ["1001"]},
    "1102": {"has_ingredient": ["1001"]},
    "2001": {
        "has_ingredient": ["1001"],
        "has_precise_ingredient": ["1101"],
        "has_tradename": ["3001"],
        "has_dose_form": ["4001"],
        "consists_of": ["5001"],
        "has_boss": ["1001"],
    },
    "2002": {"has_ingredient": ["1002", "1001"]},
    "3001": {
        "has_ingredient": ["1001"],
        "has_precise_ingredient": ["1101"],
    },
    "3002": {
        "has_ingredient": ["1001"],
        "has_precise_ingredient": ["1101", "1102"],
    },
}


def synthetic_settings(cache_dir: Path, **updates: Any) -> Settings:
    """Explicit settings prevent a developer's environment altering fixtures."""
    values: dict[str, Any] = {
        "mode": ResourceMode.FROZEN,
        "cache_dir": cache_dir,
        "rxnav_terminology_release": SYNTHETIC_RELEASE,
        "rxnav_base_url": ENDPOINT,
        "rxnav_requests_per_second": 0,
        "identifier_hmac_key": None,
        "cache_retention_days": None,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def response_key(operation: str, query: str, **kwargs: Any) -> CacheKey:
    return CacheKey(
        provider="rxnav",
        provider_version="0.1.0",
        resource_name="RxNorm",
        terminology_release=SYNTHETIC_RELEASE,
        endpoint=ENDPOINT,
        operation=operation,
        query=query,
        **kwargs,
    )


def seed_synthetic_cache(cache_dir: Path) -> ResponseCache:
    """Populate only invented responses; FROZEN replay never permits egress."""
    cache = ResponseCache(cache_dir, mode=ResourceMode.FROZEN)
    metadata = {"terminology_release": SYNTHETIC_RELEASE, "vocabulary": "RxNorm"}

    def store(operation: str, query: str, payload: Any, **kwargs: Any) -> None:
        cache.set(response_key(operation, query, **kwargs), payload, metadata)

    profile = load_profile("faers_drug_normalization")
    predicate_sets = [
        profile.knowledge.allowed_relationships,
        ["has_ingredient"],
        ["has_ingredient", "has_precise_ingredient"],
        ["has_boss"],
    ]
    for concept_id, (name, tty, status) in CONCEPTS.items():
        properties = (
            {"rxcui": concept_id, "name": name, "tty": tty, "synonym": "", "suppress": "N"}
            if status == "Active"
            else {}
        )
        history: dict[str, Any] = {
            "metaData": {"status": status, "source": "RXNORM"},
            "attributes": {
                "rxcui": concept_id,
                "name": name,
                "tty": tty,
                "isMultipleIngredient": "YES" if concept_id == "2002" else "NO",
                "isBranded": "YES" if tty in {"BN", "SBD"} else "NO",
            },
        }
        if status == "Remapped":
            history["derivedConcepts"] = {"remappedConcept": [{"remappedRxCui": "2001"}]}
        store("properties", concept_id, {"properties": properties})
        store("history_status", concept_id, {"rxcuiStatusHistory": history})
        relationships = RELATIONSHIPS.get(concept_id, {})
        for predicates in predicate_sets:
            tty_groups = []
            for predicate in predicates:
                target_tty = PREDICATE_TO_TTY.get(predicate)
                if target_tty is None:
                    continue
                tty_groups.append(
                    {
                        "tty": target_tty,
                        "conceptProperties": [
                            {"rxcui": target, "name": CONCEPTS[target][0]}
                            for target in relationships.get(predicate, [])
                        ],
                    }
                )
            if tty_groups:
                store(
                    "related_by_tty",
                    concept_id,
                    {"relatedGroup": {"conceptGroup": tty_groups}},
                    allowed_predicates=sorted(predicates),
                    max_hops=1,
                )
        for predicate in ["has_boss", "boss_of"]:
            groups = [
                {
                    "tty": CONCEPTS[target][1],
                    "conceptProperties": [{"rxcui": target, "name": CONCEPTS[target][0]}],
                }
                for target in relationships.get(predicate, [])
            ]
            store(
                "related_by_rela",
                concept_id,
                {"relatedGroup": {"conceptGroup": groups}},
                allowed_predicates=[predicate],
                max_hops=1,
            )

    store("properties", "99999999", {})
    store(
        "history_status",
        "99999999",
        {"rxcuiStatusHistory": {"metaData": {"status": "UNKNOWN", "source": "NONE"}}},
    )
    exact = {name: concept_id for concept_id, (name, _, _) in CONCEPTS.items()}
    exact.pop(HISTORICAL)
    exact["  ExampleAlpha\t "] = "1001"
    exact["examplealpha"] = "1001"
    queries = [*exact, NO_MATCH, HISTORICAL, MISSPELLING, REJECTED, "ExampleBrand 10 MG"]
    for query in queries:
        concept_id = exact.get(query)
        store("find_by_name", query, {"idGroup": {"rxnormId": [concept_id] if concept_id else []}})
        store(
            "spelling_suggestions",
            query,
            {"suggestionGroup": {"suggestionList": {"suggestion": []}}},
        )
        for current in [True, False]:
            matched = concept_id
            if query in {MISSPELLING, "ExampleBrand 10 MG"}:
                matched = "1001" if query == MISSPELLING else "2001"
            elif query == REJECTED:
                matched = "8001"
            elif query == HISTORICAL:
                matched = None if current else "6001"
            for maximum in [1, 4, 5, 10]:
                candidates = (
                    [
                        {
                            "rxcui": matched,
                            "rxaui": f"synthetic-atom-{matched}",
                            "source": "RXNORM",
                            "score": "95.0",
                            "rank": "1",
                        }
                    ]
                    if matched
                    else []
                )
                store(
                    "approximate_current" if current else "approximate_historical",
                    query,
                    {"approximateGroup": {"candidate": candidates}},
                    filters={"maxEntries": maximum, "option": 1 if current else 0},
                )
    return cache


def make_pipeline(
    cache_dir: Path,
    *,
    profile: TaskProfile | None = None,
    resolver: Resolver | None = None,
    workflow: Any = None,
) -> BioSemAlign:
    cache = seed_synthetic_cache(cache_dir)
    return BioSemAlign(
        profile or load_profile("faers_drug_normalization"),
        settings=synthetic_settings(cache_dir),
        cache=cache,
        resolver=resolver,
        workflow=workflow,
    )


def canonicalize(value: Any, *, temp_root: Path | None = None) -> Any:
    """Normalize only run-specific snapshot fields, after model validation.

    SAU/mapping identifiers, versions, counters, source values and semantic
    decisions are deliberately compared exactly. Normalized dictionaries are
    snapshots, not valid artifacts with recomputed content hashes.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        is_artifact = {"knowledge_package", "proposal", "decision"} <= value.keys()
        is_provenance = {"package_version", "profile_name", "created_at"} <= value.keys()
        is_resource = {"resource_name", "provider_name", "mode"} <= value.keys()
        volatile = set()
        if is_artifact:
            volatile.update({"run_id", "artifact_id"})
        if is_provenance:
            volatile.update({"run_id", "created_at"})
        if is_resource:
            volatile.add("accessed_at")
        return {
            key: (
                f"<{key}>"
                if key in volatile
                else item.replace(str(temp_root), "<temp_root>")
                if key == "error_message" and isinstance(item, str) and temp_root is not None
                else canonicalize(item, temp_root=temp_root)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [canonicalize(item, temp_root=temp_root) for item in value]
    return value


def baseline_rows() -> list[dict[str, Any] | BatchInputError]:
    """Frozen FAERS-shaped inputs, including every current row outcome."""
    values = [
        ALPHA,
        PRODUCT,
        COMBINATION,
        SALT,
        NO_MATCH,
        REJECTED,
        HISTORICAL,
        AMBIGUOUS,
        UNSUPPORTED,
        "  ",
        ALPHA,
        "  ExampleAlpha\t ",
        MISSPELLING,
        "ExampleMissingSnapshot",
        BETA,
    ]
    rows: list[dict[str, Any] | BatchInputError] = [
        {"record_id": f"synthetic-row-{index}", "drugname": value, "route": "oral"}
        for index, value in enumerate(values, 1)
    ]
    rows.extend(
        [
            {"record_id": "synthetic-missing-value", "route": "oral"},
            {"drugname": ALPHA, "route": "oral"},
            {"record_id": "", "drugname": ALPHA, "route": "oral"},
            {"record_id": "synthetic-missing-context", "drugname": ALPHA},
            BatchInputError("MALFORMED_SOURCE_ROW", "synthetic parser failure"),
        ]
    )
    return rows
