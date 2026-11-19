"""The local lexicon contract (v0.2).

BioSemAlign bundles no terminology. A client compiles the vocabularies it is
licensed to use into a *lexicon bundle*: a read-only DuckDB index plus a
manifest that pins its content by SHA-256. Source tables follow a small,
engine-defined contract:

``concepts`` (required)
    ``vocabulary_id``, ``concept_code``, ``concept_name``; optional
    ``concept_class``, ``status`` (a :class:`ConceptStatus` value, default
    ``active``), ``valid_start_date``, ``valid_end_date`` (ISO dates),
    ``replaced_by`` (a code in the same vocabulary), and ``name_source`` (who
    asserted the preferred name; default the concept's vocabulary — set it for
    aggregators such as UMLS, whose preferred names come from other sources).
``synonyms``
    ``vocabulary_id``, ``concept_code``, ``synonym``; optional
    ``synonym_type`` and ``source``. ``source`` names the authority that
    asserted the name — for a UMLS atom, its source vocabulary — and defaults
    to the concept's own vocabulary. Concordance counts sources, not
    vocabularies, so a redistributed name is never counted twice.
``relationships``
    ``vocabulary_id_1``, ``concept_code_1``, ``predicate``,
    ``vocabulary_id_2``, ``concept_code_2``; optional ``assertion_source``
    (who asserted or derived the edge). Predicates use the engine's names
    (``has_ingredient``, ``has_tradename``, ...), which profiles allowlist.
    Edges normally stay inside one vocabulary; a declaration's
    ``relationship_families`` lets them cross between vocabularies that extend
    one another, such as RxNorm Extension products whose ingredients are
    RxNorm concepts.
``xrefs``
    ``vocabulary_id``, ``concept_code``, ``key_type``, ``key_value``; optional
    ``assertion_source``. Equivalence keys such as ``UNII`` or ``UMLS_CUI``,
    one row per key. Every concept also carries its own identity
    (``key_type`` = its vocabulary, ``key_value`` = its code) implicitly.

Each table may be Parquet, CSV, or TSV. A declaration lists every vocabulary
with its release and licence, and canonical names for synonym sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.util import stable_digest

__all__ = [
    "INDEX_FILE",
    "LEXICON_CONTRACT_VERSION",
    "MANIFEST_FILE",
    "NORMALIZER_VERSION",
    "RANKED_INDEXES",
    "TABLES",
    "LexiconDeclaration",
    "LexiconManifest",
    "TableSpec",
    "VocabularyDeclaration",
    "family_of",
]

LEXICON_CONTRACT_VERSION = "1.0"

#: Version of ``util.normalize_text``, ``util.tokenize``, and the trigram
#: scheme as used for indexing. A bundle indexed under a different version
#: would silently miss matches, so the provider refuses it and asks for a
#: rebuild.
NORMALIZER_VERSION = "1"

#: Ranked-retrieval indexes a bundle can carry. ``exact`` is always present.
RANKED_INDEXES = ("bm25", "trigram")

MANIFEST_FILE = "manifest.json"
INDEX_FILE = "lexicon.duckdb"


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Required and optional columns of one source table."""

    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    mandatory: bool = False

    @property
    def columns(self) -> tuple[str, ...]:
        return self.required + self.optional


TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in (
        TableSpec(
            "concepts",
            required=("vocabulary_id", "concept_code", "concept_name"),
            optional=(
                "concept_class",
                "status",
                "valid_start_date",
                "valid_end_date",
                "replaced_by",
                "name_source",
            ),
            mandatory=True,
        ),
        TableSpec(
            "synonyms",
            required=("vocabulary_id", "concept_code", "synonym"),
            optional=("synonym_type", "source"),
        ),
        TableSpec(
            "relationships",
            required=(
                "vocabulary_id_1",
                "concept_code_1",
                "predicate",
                "vocabulary_id_2",
                "concept_code_2",
            ),
            optional=("assertion_source",),
        ),
        TableSpec(
            "xrefs",
            required=("vocabulary_id", "concept_code", "key_type", "key_value"),
            optional=("assertion_source",),
        ),
    )
}


class VocabularyDeclaration(BioSemAlignModel):
    """One vocabulary a bundle serves."""

    vocabulary_id: str = Field(min_length=1)
    release: str = Field(min_length=1, description="Release identifier, e.g. '2026-10-06'.")
    license: str = Field(
        default="unspecified",
        description="Licence class, recorded so outputs can be checked before redistribution.",
    )
    description: str | None = None


class LexiconDeclaration(BioSemAlignModel):
    """What a builder declares about the vocabularies it compiles."""

    vocabularies: list[VocabularyDeclaration] = Field(min_length=1)
    source_aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Canonical names for synonym sources, e.g. {'MSH': 'MeSH'}.",
    )
    relationship_families: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Vocabularies whose relationships may cross into one another, e.g. "
            "[['RxNorm', 'RxNorm Extension']]. Every other edge stays inside one vocabulary."
        ),
    )

    @field_validator("vocabularies")
    @classmethod
    def _unique_vocabularies(
        cls, vocabularies: list[VocabularyDeclaration]
    ) -> list[VocabularyDeclaration]:
        ids = [v.vocabulary_id.casefold() for v in vocabularies]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"vocabularies are declared more than once: {duplicates}")
        return vocabularies

    @model_validator(mode="after")
    def _families_are_declared(self) -> LexiconDeclaration:
        validate_families(self.relationship_families, self.vocabularies)
        return self


class LexiconManifest(BioSemAlignModel):
    """Identity and integrity record of a built lexicon bundle."""

    contract_version: str
    normalizer_version: str
    builder: str
    duckdb_version: str
    built_at: datetime
    index_file: str = INDEX_FILE
    index_sha256: str = Field(min_length=64, max_length=64)
    inputs_sha256: dict[str, str] = Field(default_factory=dict)
    vocabularies: list[VocabularyDeclaration] = Field(min_length=1)
    source_aliases: dict[str, str] = Field(default_factory=dict)
    relationship_families: list[list[str]] = Field(default_factory=list)
    row_counts: dict[str, int] = Field(default_factory=dict)
    indexes: list[str] = Field(
        default_factory=lambda: ["exact", "bm25"],
        description="Retrieval methods the index supports: exact, bm25, trigram.",
    )
    bundle_sha256: str = Field(
        min_length=64,
        max_length=64,
        description=(
            "Digest of the index and every declaration that changes what it means. "
            "Deployments pin this value; the build timestamp is excluded."
        ),
    )

    @model_validator(mode="after")
    def _bundle_digest_matches(self) -> LexiconManifest:
        if self.bundle_sha256 != self.expected_bundle_sha256():
            raise ValueError("manifest bundle_sha256 does not match its own contents")
        return self

    def expected_bundle_sha256(self) -> str:
        return bundle_digest(
            contract_version=self.contract_version,
            normalizer_version=self.normalizer_version,
            index_sha256=self.index_sha256,
            vocabularies=self.vocabularies,
            source_aliases=self.source_aliases,
            relationship_families=self.relationship_families,
        )

    def vocabulary(self, vocabulary_id: str) -> VocabularyDeclaration | None:
        wanted = vocabulary_id.casefold()
        return next((v for v in self.vocabularies if v.vocabulary_id.casefold() == wanted), None)

    def family(self, vocabulary_id: str) -> tuple[str, ...]:
        """The vocabulary and every vocabulary its relationships may cross into."""
        return family_of(vocabulary_id, self.relationship_families)


def family_of(vocabulary_id: str, families: list[list[str]]) -> tuple[str, ...]:
    """``vocabulary_id`` first, then its family members in declared order."""
    wanted = vocabulary_id.casefold()
    for family in families:
        if any(member.casefold() == wanted for member in family):
            return (vocabulary_id, *(m for m in family if m.casefold() != wanted))
    return (vocabulary_id,)


def validate_families(families: list[list[str]], vocabularies: list[VocabularyDeclaration]) -> None:
    """Every family names at least two declared vocabularies, each in one family only."""
    declared = {v.vocabulary_id.casefold() for v in vocabularies}
    seen: set[str] = set()
    for family in families:
        members = [member.casefold() for member in family]
        if len(set(members)) < 2:
            raise ValueError(f"relationship family {family} needs at least two vocabularies")
        unknown = sorted(set(members) - declared)
        if unknown:
            raise ValueError(
                f"relationship family {family} names undeclared vocabularies {unknown}"
            )
        repeated = sorted(seen & set(members))
        if repeated:
            raise ValueError(f"vocabularies {repeated} appear in more than one relationship family")
        seen |= set(members)


def bundle_digest(
    *,
    contract_version: str,
    normalizer_version: str,
    index_sha256: str,
    vocabularies: list[VocabularyDeclaration],
    source_aliases: dict[str, str],
    relationship_families: list[list[str]] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "contract_version": contract_version,
        "normalizer_version": normalizer_version,
        "index_sha256": index_sha256,
        "vocabularies": [v.model_dump(mode="json") for v in vocabularies],
        "source_aliases": dict(sorted(source_aliases.items())),
    }
    # Families change what a bundle's relationships mean, so they are pinned —
    # but only when declared, keeping every earlier bundle's digest stable.
    if relationship_families:
        payload["relationship_families"] = [list(f) for f in relationship_families]
    return stable_digest(payload)
