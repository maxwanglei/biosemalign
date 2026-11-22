"""An invented multi-vocabulary lexicon for hermetic tests.

RxNorm here mirrors the synthetic RxNav universe in ``tests.support`` concept
for concept, so a lexicon-served pipeline can be compared with the RxNav one.
UNII, MeSH, and UMLS are invented too, arranged to exercise each concordance
outcome. None of these codes or names is real terminology content.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from biosemalign import BioSemAlign
from biosemalign.knowledge.lexicon.build import build_lexicon
from biosemalign.knowledge.lexicon.contract import LexiconDeclaration, LexiconManifest
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.storage.cache import ResponseCache
from tests.support import CONCEPTS, RELATIONSHIPS, synthetic_settings

_STATUS = {"Active": "active", "NotCurrent": "not_current", "Remapped": "remapped"}

#: Mentions and the concordance outcome each is built to produce.
CONCORDANT = "ExampleAlpha"  # RxNorm, UNII, MeSH, UMLS agree: three independent sources
REDISTRIBUTED = "ExampleBeta"  # RxNorm, plus a UMLS atom whose source is RxNorm: one source
COLLISION = "Examplex"  # names two unrelated concepts
CONFLICT = "Alphasalt"  # an over-broad UNII key merges a salt with its base
ACRONYM = "EXA"  # too short to resolve by name
MESH_ONLY = "Meshonlyname"  # only MeSH has it; no RxNorm concept in the group
TYPO = "ExampleAlpah"  # no exact match anywhere

DECLARATION = LexiconDeclaration.model_validate(
    {
        "vocabularies": [
            {"vocabulary_id": "RxNorm", "release": "synthetic-rxnorm-v1", "license": "public"},
            {"vocabulary_id": "UNII", "release": "synthetic-unii-v1", "license": "public"},
            {"vocabulary_id": "MeSH", "release": "synthetic-mesh-v1", "license": "public"},
            {"vocabulary_id": "UMLS", "release": "synthetic-umls-v1", "license": "restricted"},
        ],
        "source_aliases": {"RXNORM": "RxNorm", "MSH": "MeSH", "FDA SRS": "UNII"},
    }
)


def source_rows() -> dict[str, list[dict[str, Any]]]:
    concepts: list[dict[str, Any]] = [
        {
            "vocabulary_id": "RxNorm",
            "concept_code": code,
            "concept_name": name,
            "concept_class": tty,
            "status": _STATUS[status],
            "replaced_by": "2001" if status == "Remapped" else "",
            "valid_start_date": "",
            "valid_end_date": "2020-12-31" if code == "6001" else "",
            "name_source": "",
        }
        for code, (name, tty, status) in CONCEPTS.items()
    ]
    # UMLS preferred names are atoms from other sources: the alpha CUI's from
    # MeSH, the beta CUI's from RxNorm itself.
    for vocabulary, code, name, klass, name_source in [
        ("UNII", "U-ALPHA", "EXAMPLEALPHA", "substance", ""),
        ("UNII", "U-SALT", "Alphasalt unii", "substance", ""),
        ("MeSH", "D-ALPHA", "ExampleAlpha", "descriptor", ""),
        ("MeSH", "D-OTHER", "Examplex", "descriptor", ""),
        ("MeSH", "D-ONLY", MESH_ONLY, "descriptor", ""),
        ("UMLS", "C-ALPHA", "ExampleAlpha", "cui", "MSH"),
        ("UMLS", "C-BETA", "ExampleBeta", "cui", "RXNORM"),
    ]:
        concepts.append(
            {
                "vocabulary_id": vocabulary,
                "concept_code": code,
                "concept_name": name,
                "concept_class": klass,
                "status": "active",
                "replaced_by": "",
                "valid_start_date": "",
                "valid_end_date": "",
                "name_source": name_source,
            }
        )
    synonyms = [
        # UNII's name for alpha is asserted by the FDA's substance registry.
        {
            "vocabulary_id": "UNII",
            "concept_code": "U-ALPHA",
            "synonym": CONCORDANT,
            "source": "FDA SRS",
        },
        {"vocabulary_id": "RxNorm", "concept_code": "8001", "synonym": COLLISION, "source": ""},
        {"vocabulary_id": "RxNorm", "concept_code": "1001", "synonym": CONFLICT, "source": ""},
        {"vocabulary_id": "RxNorm", "concept_code": "1101", "synonym": CONFLICT, "source": ""},
        {
            "vocabulary_id": "UNII",
            "concept_code": "U-SALT",
            "synonym": CONFLICT,
            "source": "FDA SRS",
        },
        {"vocabulary_id": "RxNorm", "concept_code": "1001", "synonym": ACRONYM, "source": ""},
    ]
    relationships = [
        {
            "vocabulary_id_1": "RxNorm",
            "concept_code_1": subject,
            "predicate": predicate,
            "vocabulary_id_2": "RxNorm",
            "concept_code_2": obj,
        }
        for subject, predicates in RELATIONSHIPS.items()
        for predicate, objects in predicates.items()
        for obj in objects
    ]
    xrefs = [
        {
            "vocabulary_id": "UNII",
            "concept_code": "U-ALPHA",
            "key_type": "RxNorm",
            "key_value": "1001",
        },
        {
            "vocabulary_id": "UMLS",
            "concept_code": "C-ALPHA",
            "key_type": "RxNorm",
            "key_value": "1001",
        },
        {
            "vocabulary_id": "UMLS",
            "concept_code": "C-ALPHA",
            "key_type": "MeSH",
            "key_value": "D-ALPHA",
        },
        {
            "vocabulary_id": "UMLS",
            "concept_code": "C-BETA",
            "key_type": "RxNorm",
            "key_value": "1002",
        },
        # Over-broad: one substance key on both the base and its salt.
        {
            "vocabulary_id": "UNII",
            "concept_code": "U-SALT",
            "key_type": "RxNorm",
            "key_value": "1001",
        },
        {
            "vocabulary_id": "UNII",
            "concept_code": "U-SALT",
            "key_type": "RxNorm",
            "key_value": "1101",
        },
    ]
    return {
        "concepts": concepts,
        "synonyms": synonyms,
        "relationships": relationships,
        "xrefs": xrefs,
    }


def write_sources(directory: Path, **overrides: list[dict[str, Any]] | None) -> Path:
    """Write the contract tables as CSV; an override of None omits a table."""
    directory.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, Any]] | None] = {**source_rows(), **overrides}
    for name, rows in tables.items():
        if rows is None:
            continue
        columns = list(rows[0]) if rows else []
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return directory


def build_bundle(
    root: Path, **overrides: list[dict[str, Any]] | None
) -> tuple[Path, LexiconManifest]:
    source = write_sources(root / "source", **overrides)
    bundle = root / "bundle"
    return bundle, build_lexicon(source, bundle, declaration=DECLARATION)


def lexicon_profile(
    *,
    tiers: list[dict[str, Any]] | None = None,
    resources: list[str] | None = None,
    secondary_resources: list[str] | None = None,
    **knowledge: Any,
) -> TaskProfile:
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["tiers"] = tiers
    if resources is not None:
        payload["knowledge"]["resources"] = list(resources)
    elif secondary_resources is not None:
        existing = list(payload["knowledge"].get("resources") or [])
        payload["knowledge"]["resources"] = list(dict.fromkeys([*existing, *secondary_resources]))
    payload["knowledge"].update(knowledge)
    return TaskProfile.model_validate(payload)


CONCORDANCE_TIERS: list[dict[str, Any]] = [
    {
        "name": "concordant",
        "match": [
            {
                "op": "exact",
                "options": {"vocabularies": ["RxNorm", "UNII", "MeSH", "UMLS"]},
            }
        ],
        "link": {"op": "standard", "options": {"concordance": True}},
        "resolve": {"op": "concordant"},
    },
    {
        "name": "fallback",
        "match": [
            {"op": "exact", "options": {"vocabularies": ["RxNorm"]}},
            {"op": "lexical", "options": {"vocabularies": ["RxNorm"], "method": "trigram"}},
        ],
    },
]


def concordance_profile(**resolver_options: Any) -> TaskProfile:
    tiers = [dict(tier) for tier in CONCORDANCE_TIERS]
    if resolver_options:
        tiers[0] = {
            **tiers[0],
            "resolve": {"op": "concordant", "options": resolver_options},
        }
    return lexicon_profile(tiers=tiers, secondary_resources=["UNII", "MeSH", "UMLS"])


def lexicon_pipeline(
    root: Path,
    bundle: Path,
    manifest: LexiconManifest,
    *,
    profile: TaskProfile | None = None,
    **settings: Any,
) -> BioSemAlign:
    """A FROZEN pipeline whose RxNorm comes only from the lexicon bundle."""
    cache_dir = root / "cache"
    values: dict[str, Any] = {"lexicon_path": bundle, "lexicon_sha256": manifest.bundle_sha256}
    values.update(settings)
    configured = synthetic_settings(cache_dir, **values)
    return BioSemAlign(
        profile or lexicon_profile(),
        settings=configured,
        cache=ResponseCache(cache_dir, mode=configured.mode),
    )
