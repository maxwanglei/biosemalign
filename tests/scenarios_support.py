"""Appendix A invented terminology and scenario helpers for WP0 tests.

All codes, names, and relationships here follow ``docs/framework-design.md``
Appendix A. Every identifier is synthetic and contains no real terminology
content.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from biosemalign import BioSemAlign
from biosemalign.cli import app
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.storage.cache import ResponseCache
from tests.support import synthetic_settings

__all__ = [
    "CONCEPTS",
    "DECLARATION",
    "RELATIONSHIPS",
    "SYNONYMS",
    "appendix_a_drug_profile",
    "appendix_a_pipeline",
    "appendix_a_toy_disease_profile",
    "build_appendix_a_bundle",
    "write_appendix_a_sources",
]

CONCEPTS: list[dict[str, str]] = [
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T100",
        "concept_name": "toyprolol",
        "concept_class": "IN",
        "status": "active",
    },
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T101",
        "concept_name": "toyprolol tartrate",
        "concept_class": "PIN",
        "status": "active",
    },
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T200",
        "concept_name": "Toybrand XL 50 MG Oral Tablet",
        "concept_class": "SBD",
        "status": "active",
    },
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T201",
        "concept_name": "Toybrand XL",
        "concept_class": "BN",
        "status": "active",
    },
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T300",
        "concept_name": "alphazine",
        "concept_class": "IN",
        "status": "active",
    },
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T301",
        "concept_name": "betazine",
        "concept_class": "IN",
        "status": "active",
    },
    {
        "vocabulary_id": "RxNorm",
        "concept_code": "T302",
        "concept_name": "alphazine / betazine Oral Tablet",
        "concept_class": "SCD",
        "status": "active",
    },
    {
        "vocabulary_id": "ToyDisease",
        "concept_code": "D1",
        "concept_name": "toy fever",
        "concept_class": "Disease",
        "status": "active",
    },
    {
        "vocabulary_id": "ToyDisease",
        "concept_code": "D2",
        "concept_name": "severe toy fever",
        "concept_class": "Disease",
        "status": "active",
    },
    {
        "vocabulary_id": "ToyDisease",
        "concept_code": "D3",
        "concept_name": "recurrent toy fever",
        "concept_class": "Disease",
        "status": "active",
    },
    {
        "vocabulary_id": "ToyDisease",
        "concept_code": "G1",
        "concept_name": "toy fever disorders",
        "concept_class": "Group",
        "status": "active",
    },
]

SYNONYMS: list[dict[str, str]] = [
    {"vocabulary_id": "RxNorm", "concept_code": "T200", "synonym": "TOYBRAND XL 50 MG"},
    {"vocabulary_id": "RxNorm", "concept_code": "T302", "synonym": "ALPHABETA"},
    {"vocabulary_id": "ToyDisease", "concept_code": "D1", "synonym": "toy pyrexia"},
]

RELATIONSHIPS: list[dict[str, str]] = [
    {
        "vocabulary_id_1": "RxNorm",
        "concept_code_1": "T200",
        "predicate": "has_ingredient",
        "vocabulary_id_2": "RxNorm",
        "concept_code_2": "T100",
    },
    {
        "vocabulary_id_1": "RxNorm",
        "concept_code_1": "T200",
        "predicate": "has_precise_ingredient",
        "vocabulary_id_2": "RxNorm",
        "concept_code_2": "T101",
    },
    {
        "vocabulary_id_1": "RxNorm",
        "concept_code_1": "T200",
        "predicate": "has_tradename",
        "vocabulary_id_2": "RxNorm",
        "concept_code_2": "T201",
    },
    {
        "vocabulary_id_1": "RxNorm",
        "concept_code_1": "T201",
        "predicate": "has_ingredient",
        "vocabulary_id_2": "RxNorm",
        "concept_code_2": "T100",
    },
    {
        "vocabulary_id_1": "RxNorm",
        "concept_code_1": "T302",
        "predicate": "has_ingredient",
        "vocabulary_id_2": "RxNorm",
        "concept_code_2": "T300",
    },
    {
        "vocabulary_id_1": "RxNorm",
        "concept_code_1": "T302",
        "predicate": "has_ingredient",
        "vocabulary_id_2": "RxNorm",
        "concept_code_2": "T301",
    },
    {
        "vocabulary_id_1": "ToyDisease",
        "concept_code_1": "D2",
        "predicate": "is_a",
        "vocabulary_id_2": "ToyDisease",
        "concept_code_2": "D1",
    },
    {
        "vocabulary_id_1": "ToyDisease",
        "concept_code_1": "D3",
        "predicate": "is_a",
        "vocabulary_id_2": "ToyDisease",
        "concept_code_2": "D1",
    },
    {
        "vocabulary_id_1": "ToyDisease",
        "concept_code_1": "D1",
        "predicate": "in_group",
        "vocabulary_id_2": "ToyDisease",
        "concept_code_2": "G1",
    },
]

DECLARATION: dict[str, Any] = {
    "vocabularies": [
        {"vocabulary_id": "RxNorm", "release": "toy-rxnorm-1", "license": "synthetic"},
        {"vocabulary_id": "ToyDisease", "release": "toy-disease-1", "license": "synthetic"},
    ]
}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_appendix_a_sources(directory: Path) -> tuple[Path, Path]:
    """Write Appendix A source tables and vocabulary declaration to disk."""
    tables_dir = directory / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / "concepts.csv", CONCEPTS)
    _write_csv(tables_dir / "synonyms.csv", SYNONYMS)
    _write_csv(tables_dir / "relationships.csv", RELATIONSHIPS)
    declaration_path = directory / "vocabularies.yaml"
    declaration_path.write_text(yaml.safe_dump(DECLARATION, sort_keys=False), encoding="utf-8")
    return tables_dir, declaration_path


def build_appendix_a_bundle(root: Path) -> tuple[Path, LexiconManifest]:
    """Build the Appendix A lexicon bundle using ``biosemalign lexicon build``."""
    tables_dir, declaration_path = write_appendix_a_sources(root / "source")
    bundle_dir = root / "bundle"
    result = CliRunner().invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(tables_dir),
            "--declaration",
            str(declaration_path),
            "--output",
            str(bundle_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = LexiconManifest.model_validate_json(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return bundle_dir, manifest


def appendix_a_drug_profile(
    *,
    required_context: list[str] | None = None,
    desired_granularity: str | None = "ingredient",
    include_cleaned_tier: bool = True,
) -> TaskProfile:
    """Two-tier drug normalization profile for Appendix A scenarios."""
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["task"]["desired_granularity"] = desired_granularity
    if required_context is not None:
        payload["input"]["required_context"] = list(required_context)

    tiers: list[dict[str, Any]] = [
        {
            "name": "exact",
            "retrievers": [
                {"op": "lexicon", "options": {"vocabularies": ["RxNorm"], "methods": ["exact"]}}
            ],
            "fuser": {"op": "cross_vocab_concordance"},
            "proposer": {"op": "concordant", "options": {"authority_sources": ["RxNorm"]}},
        }
    ]
    if include_cleaned_tier:
        tiers.append(
            {
                "name": "cleaned",
                "transforms": [{"op": "unicode_normalize"}, {"op": "extract_strength"}],
                "retrievers": [
                    {"op": "lexicon", "options": {"vocabularies": ["RxNorm"], "methods": ["exact"]}}
                ],
                "fuser": {"op": "knowledge_package"},
                "proposer": {"op": "adjudicator"},
            }
        )
    payload["workflow"] = {"tiers": tiers}
    return TaskProfile.model_validate(payload)


def appendix_a_toy_disease_profile(*, desired_granularity: str | None = None) -> TaskProfile:
    """Profile targeting the invented ``ToyDisease`` vocabulary."""
    return TaskProfile.model_validate(
        {
            "profile_name": "toy_disease",
            "profile_version": "0.1.0",
            "task": {
                "operation": "disease_coding",
                "mapping_intent": "exact",
                "target_terminology": "ToyDisease",
                "desired_granularity": desired_granularity,
            },
            "knowledge": {"primary_resources": ["ToyDisease"]},
            "workflow": {
                "tiers": [
                    {
                        "name": "exact",
                        "retrievers": [
                            {
                                "op": "lexicon",
                                "options": {"vocabularies": ["ToyDisease"], "methods": ["exact"]},
                            }
                        ],
                    }
                ]
            },
        }
    )


def appendix_a_pipeline(
    root: Path,
    bundle: Path,
    manifest: LexiconManifest,
    *,
    profile: TaskProfile | None = None,
) -> BioSemAlign:
    """Offline FROZEN pipeline backed by the Appendix A lexicon bundle."""
    cache_dir = root / "cache"
    configured = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    return BioSemAlign(
        profile or appendix_a_drug_profile(),
        settings=configured,
        cache=ResponseCache(cache_dir, mode=configured.mode),
    )
