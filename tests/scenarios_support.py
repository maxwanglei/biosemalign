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
from pydantic import BaseModel
from typer.testing import CliRunner

from biosemalign import BioSemAlign
from biosemalign.cli import app
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.modules.resolve.backend import GenerationConfig
from biosemalign.modules.resolve.leashed import LeashedLLMResolver
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.artifact import BatchAlignmentRecord
from biosemalign.storage.cache import ResponseCache
from tests.support import synthetic_settings

__all__ = [
    "CONCEPTS",
    "DECLARATION",
    "RELATIONSHIPS",
    "SYNONYMS",
    "TOY_DISEASE_GROUP_SPEC",
    "appendix_a_coding_profile",
    "appendix_a_drug_profile",
    "appendix_a_llm_profile",
    "appendix_a_phenotyping_profile",
    "appendix_a_pipeline",
    "appendix_a_toy_disease_group_profile",
    "appendix_a_toy_disease_hierarchy_profile",
    "appendix_a_toy_disease_profile",
    "build_appendix_a_bundle",
    "generate_schema_0_3_0_records",
    "schema_0_3_0_profiles",
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

TOY_DISEASE_GROUP_SPEC: dict[str, Any] = {
    "spec_id": "toydisease",
    "version": "1.0",
    "target": "ToyDisease",
    "family": ["ToyDisease"],
    "granularities": {
        "group": {
            "via": ["in_group"],
            "terminal": ["Group"],
        }
    },
    "release_bound_predicates": ["in_group"],
    "edge_target_classes": {"in_group": "Group"},
}


class _ScriptedLLMBackend:
    """Minimal scripted backend for offline schema-fixture generation."""

    backend_name = "scripted"

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply

    def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        generation_config: GenerationConfig,
    ) -> BaseModel:
        del messages, generation_config
        return output_schema.model_validate(self.reply)


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
    profile_name: str = "appendix_a_drug_normalization",
    required_context: list[str] | None = None,
    desired_granularity: str | None = "ingredient",
    include_cleaned_tier: bool = True,
) -> TaskProfile:
    """Two-tier drug normalization profile for Appendix A scenarios."""
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["profile_name"] = profile_name
    payload["profile_version"] = "0.1.0"
    payload["target"]["level"] = desired_granularity
    if required_context is not None:
        fields = [dict(f) for f in payload["context"]["fields"]]
        by_name = {f["name"]: f for f in fields}
        for name in required_context:
            if name in by_name:
                by_name[name]["required"] = True
            else:
                fields.append({"name": name, "required": True})
        payload["context"]["fields"] = fields

    tiers: list[dict[str, Any]] = [
        {
            "name": "exact",
            "match": [{"op": "exact", "options": {"vocabularies": ["RxNorm"]}}],
            "link": {"op": "standard", "options": {"concordance": True}},
            "resolve": {"op": "concordant", "options": {"authority_sources": ["RxNorm"]}},
        }
    ]
    if include_cleaned_tier:
        tiers.append(
            {
                "name": "cleaned",
                "clean": [{"op": "unicode_normalize"}, {"op": "drug.extract_strength"}],
                "match": [{"op": "exact", "options": {"vocabularies": ["RxNorm"]}}],
                "expand": {"op": "refine_by_slots"},
                "link": {"op": "standard"},
                "resolve": {"op": "rules"},
            }
        )
    payload["tiers"] = tiers
    return TaskProfile.model_validate(payload)


def appendix_a_llm_profile() -> TaskProfile:
    """Single-tier drug profile used with a scripted LLM resolver."""
    payload = appendix_a_drug_profile(
        profile_name="appendix_a_drug_llm", include_cleaned_tier=False
    ).model_dump(mode="json")
    payload["tiers"] = [
        {
            "name": "llm_tier",
            "match": [{"op": "exact", "options": {"vocabularies": ["RxNorm"]}}],
            "link": {"op": "standard", "options": {"concordance": True}},
            "resolve": {"op": "rules"},
        }
    ]
    return TaskProfile.model_validate(payload)


def appendix_a_coding_profile(
    *,
    profile_name: str = "appendix_a_coding",
    include_cleaned_tier: bool = True,
) -> TaskProfile:
    """Two-tier coding profile (exact -> cleaned) targeting RxNorm ingredients."""
    base = appendix_a_drug_profile(profile_name=profile_name).model_dump(mode="json")
    base["workflow"] = "coding"
    base["task"] = {"operation": "drug_coding", "mapping_intent": "coding_only"}
    tiers: list[dict[str, Any]] = [
        {
            "name": "exact",
            "match": [{"op": "exact", "options": {"vocabularies": ["RxNorm"]}}],
            "link": {"op": "standard"},
        }
    ]
    if include_cleaned_tier:
        tiers.append(
            {
                "name": "cleaned",
                "clean": [{"op": "unicode_normalize"}, {"op": "drug.extract_strength"}],
                "match": [{"op": "exact", "options": {"vocabularies": ["RxNorm"]}}],
                "link": {"op": "standard"},
            }
        )
    base["tiers"] = tiers
    return TaskProfile.model_validate(base)


def appendix_a_phenotyping_profile(*, desired_granularity: str | None = "Disease") -> TaskProfile:
    """Phenotyping-domain profile targeting ``ToyDisease`` via ``hierarchy``."""
    return TaskProfile.model_validate(
        {
            "profile_schema": 2,
            "profile_name": "appendix_a_phenotyping",
            "profile_version": "0.1.0",
            "domain": "hierarchy",
            "workflow": "phenotyping",
            "task": {
                "operation": "disease_phenotyping",
                "mapping_intent": "concept_set_curation",
            },
            "target": {
                "terminology": "ToyDisease",
                "spec": "toydisease",
                "level": desired_granularity,
            },
            "context": {
                "fields": [
                    {"name": "route", "kind": "attribute", "use": ["key", "select"]},
                    {"name": "indication", "kind": "context", "use": [], "blinded": True},
                ]
            },
            "knowledge": {
                "resources": ["ToyDisease"],
                "allowed_relationships": ["is_a", "in_group"],
            },
            "tiers": [
                {
                    "name": "phenotype",
                    "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
                    "expand": [{"op": "hierarchy"}],
                    "select": {"op": "hierarchy"},
                }
            ],
        }
    )


def appendix_a_toy_disease_profile(*, desired_granularity: str | None = None) -> TaskProfile:
    """Profile targeting the invented ``ToyDisease`` vocabulary."""
    return TaskProfile.model_validate(
        {
            "profile_schema": 2,
            "profile_name": "toy_disease",
            "profile_version": "0.1.0",
            "task": {"operation": "disease_coding", "mapping_intent": "exact"},
            "target": {"terminology": "ToyDisease", "level": desired_granularity},
            "knowledge": {"resources": ["ToyDisease"]},
            "tiers": [
                {
                    "name": "exact",
                    "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
                }
            ],
        }
    )


def appendix_a_toy_disease_group_profile(spec_path: Path | None = None) -> TaskProfile:
    """Profile mapping ``ToyDisease`` terms to ``Group`` via a non-built-in spec."""
    return TaskProfile.model_validate(
        {
            "profile_schema": 2,
            "profile_name": "appendix_a_toy_disease_group",
            "profile_version": "0.1.0",
            "task": {
                "operation": "disease_grouping",
                "mapping_intent": "group_level_analysis",
            },
            "target": {
                "terminology": "ToyDisease",
                "level": "group",
                "spec": str(spec_path) if spec_path is not None else None,
            },
            "knowledge": {
                "resources": ["ToyDisease"],
                "allowed_relationships": ["in_group", "is_a"],
            },
            "tiers": [
                {
                    "name": "exact",
                    "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
                    "link": {"op": "standard", "options": {"concordance": True}},
                    "resolve": {
                        "op": "concordant",
                        "options": {"authority_sources": ["ToyDisease"]},
                    },
                }
            ],
        }
    )


def appendix_a_toy_disease_hierarchy_profile(
    *, desired_granularity: str | None = "Group"
) -> TaskProfile:
    """Hierarchy-domain profile targeting ``ToyDisease`` via the built-in pack."""
    return TaskProfile.model_validate(
        {
            "profile_schema": 2,
            "profile_name": "appendix_a_toy_disease_hierarchy",
            "profile_version": "0.1.0",
            "domain": "hierarchy",
            "task": {
                "operation": "disease_grouping",
                "mapping_intent": "group_level_analysis",
            },
            "target": {
                "terminology": "ToyDisease",
                "spec": "toydisease",
                "level": desired_granularity,
            },
            "knowledge": {
                "resources": ["ToyDisease"],
                "allowed_relationships": ["is_a", "in_group"],
            },
            "tiers": [
                {
                    "name": "exact",
                    "match": [{"op": "exact", "options": {"vocabularies": ["ToyDisease"]}}],
                    "link": {"op": "standard"},
                    "resolve": {"op": "rules"},
                }
            ],
            "checks": ["hierarchy.semantics"],
        }
    )


def schema_0_3_0_profiles() -> dict[str, TaskProfile]:
    """Profiles matching the records in ``tests/golden/schema_0_3_0_records.jsonl``."""
    return {
        "appendix_a_drug_normalization": appendix_a_drug_profile(),
        "appendix_a_drug_llm": appendix_a_llm_profile(),
        "appendix_a_drug_exact_only": appendix_a_drug_profile(
            profile_name="appendix_a_drug_exact_only", include_cleaned_tier=False
        ),
        "appendix_a_toy_disease_group": appendix_a_toy_disease_group_profile(),
    }


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


def generate_schema_0_3_0_records(root: Path) -> list[BatchAlignmentRecord]:
    """Generate the 8 deterministic ``0.3.0`` fixture records from Appendix A data.

    Used to build ``tests/golden/schema_0_3_0_records.jsonl`` with
    ``run_id="schema-0-3-0-fixture-run"`` and zero real terminology content:
    rows 1–5 cover exact product (with ``target_tty`` and ``PROJECTION`` link),
    exact ingredient (``EQUIVALENCE`` link), combination product, cleaned-tier
    fallback (with ``transforms``, ``retrieval_text``, ``strength`` slot, and
    ``links=None``), and unmappable abstention; row 6 records ``extensions["llm"]``
    via a scripted LLM adjudicator; row 7 records an exhausted-cascade ``ABSTAINED``
    decision routed to ``HUMAN_REVIEW``; row 8 targets the non-built-in
    ``toydisease@1.0`` spec.
    """
    bundle, manifest = build_appendix_a_bundle(root)
    run_id = "schema-0-3-0-fixture-run"
    records: list[BatchAlignmentRecord] = []

    drug_profile = appendix_a_drug_profile()
    drug_rows = [
        {"drugname": "TOYBRAND XL 50 MG", "route": "oral"},
        {"drugname": "toyprolol", "route": None},
        {"drugname": "alphabeta", "route": None},
        {"drugname": "TOYBRAND XL 75 MG", "route": None},
        {"drugname": "zzzqqq", "route": None},
    ]
    with appendix_a_pipeline(root / "drug", bundle, manifest, profile=drug_profile) as pipeline:
        records.extend(
            pipeline.align_batch_records(
                drug_rows,
                value_column="drugname",
                context_columns=["route"],
                start_row=1,
                run=pipeline.new_run(run_id=run_id),
            )
        )

    llm_profile = appendix_a_llm_profile()
    backend = _ScriptedLLMBackend(
        {
            "candidate": 1,
            "abstain_reason": "none",
            "relationship": "TARGET_BROADER_THAN_SOURCE",
            "semantic_loss": ["brand_identity", "route", "salt_form", "strength"],
            "needs_review": False,
            "rationale": "branded tablet matching T200",
        }
    )
    resolver = LeashedLLMResolver(llm_profile, backend, model_name="scripted-llm-v1")
    llm_cache_dir = root / "llm" / "cache"
    llm_settings = synthetic_settings(
        llm_cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        llm_profile,
        settings=llm_settings,
        cache=ResponseCache(llm_cache_dir, mode=llm_settings.mode),
        resolver=resolver,
    ) as llm_pipeline:
        records.extend(
            llm_pipeline.align_batch_records(
                [{"drugname": "TOYBRAND XL 50 MG", "route": "oral"}],
                value_column="drugname",
                context_columns=["route"],
                start_row=6,
                run=llm_pipeline.new_run(run_id=run_id),
            )
        )

    exhausted_profile = appendix_a_drug_profile(
        profile_name="appendix_a_drug_exact_only", include_cleaned_tier=False
    )
    with appendix_a_pipeline(
        root / "exhausted", bundle, manifest, profile=exhausted_profile
    ) as exhausted_pipeline:
        records.extend(
            exhausted_pipeline.align_batch_records(
                [{"drugname": "zzzqqq"}],
                value_column="drugname",
                start_row=7,
                run=exhausted_pipeline.new_run(run_id=run_id),
            )
        )

    spec_path = root / "toydisease.yaml"
    spec_path.write_text(yaml.safe_dump(TOY_DISEASE_GROUP_SPEC, sort_keys=False), encoding="utf-8")
    td_profile = appendix_a_toy_disease_group_profile(spec_path)
    with appendix_a_pipeline(
        root / "toydisease", bundle, manifest, profile=td_profile
    ) as td_pipeline:
        records.extend(
            td_pipeline.align_batch_records(
                [{"term": "toy fever"}],
                value_column="term",
                start_row=8,
                run=td_pipeline.new_run(run_id=run_id),
            )
        )

    return records
