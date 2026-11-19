"""Convert OMOP vocabulary tables (an Athena download) into lexicon tables.

Athena distributes RxNorm, RxNorm Extension, ATC, NDC, HCPCS, and many more
vocabularies in one tab-delimited layout. This importer selects vocabularies
and writes the lexicon contract tables plus a declaration, ready for
``biosemalign lexicon build``:

* concept classes of the RxNorm family become RxNorm term types (``IN``,
  ``SCD``, ``SBD``, ...), which the drug rules key on;
* ``invalid_reason`` becomes concept status, and ``Concept replaced by``
  becomes ``replaced_by``;
* relationships are kept only when :data:`DEFAULT_RELATIONSHIP_MAP` (or the
  caller's map) names an engine predicate for them; everything else is
  counted in the report, never dropped silently;
* product-to-ingredient edges are derived from ``CONCEPT_ANCESTOR``, the
  OHDSI-standard path from any drug to its ingredients;
* every concept carries ``OMOP_CONCEPT_ID``, and ``Maps to`` relationships
  become ``OMOP_STANDARD`` keys a profile may allow for concordance.

Athena's licences travel with the data: SNOMED CT, MedDRA, and others need
their own agreements, which the declaration's licence field should record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.lexicon.build import DUCKDB_CONFIG, require_duckdb
from biosemalign.knowledge.lexicon.contract import LexiconDeclaration, VocabularyDeclaration
from biosemalign.targets import builtin_specs

__all__ = [
    "DEFAULT_RELATIONSHIP_MAP",
    "RXNORM_CLASS_TO_TTY",
    "OmopImportReport",
    "import_omop",
]

#: OHDSI relationship identifiers mapped to engine predicates. Configurable:
#: Athena's relationship vocabulary evolves, and other tasks need others.
DEFAULT_RELATIONSHIP_MAP: dict[str, str] = {
    "RxNorm has ing": "has_ingredient",
    "Has precise ing": "has_precise_ingredient",
    "Has tradename": "has_tradename",
    "RxNorm has dose form": "has_dose_form",
    "Consists of": "consists_of",
}

#: OMOP concept classes of RxNorm and RxNorm Extension, as RxNorm term types —
#: the RxNorm target spec's class aliases.
RXNORM_CLASS_TO_TTY: dict[str, str] = dict(builtin_specs()["rxnorm"].class_aliases)

_RXNORM_FAMILY = tuple(builtin_specs()["rxnorm"].family)

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "CONCEPT": (
        "concept_id",
        "concept_name",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    ),
    "VOCABULARY": ("vocabulary_id", "vocabulary_name", "vocabulary_version"),
    "CONCEPT_SYNONYM": ("concept_id", "concept_synonym_name"),
    "CONCEPT_RELATIONSHIP": (
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "invalid_reason",
    ),
    "CONCEPT_ANCESTOR": ("ancestor_concept_id", "descendant_concept_id"),
}


@dataclass(slots=True)
class OmopImportReport:
    """What an import wrote, and what it deliberately left out."""

    output_dir: Path
    vocabularies: list[str]
    row_counts: dict[str, int] = field(default_factory=dict)
    unmapped_relationships: dict[str, int] = field(default_factory=dict)


def import_omop(
    athena_dir: str | Path,
    output_dir: str | Path,
    *,
    vocabularies: Sequence[str],
    relationship_map: Mapping[str, str] | None = None,
    derive_ingredients: bool = True,
    license: str = "unspecified",
) -> OmopImportReport:
    """Write lexicon contract tables for ``vocabularies`` from an Athena download."""
    duckdb = require_duckdb()
    source = Path(athena_dir)
    output = Path(output_dir)
    if not vocabularies:
        raise LexiconError("select at least one vocabulary to import")
    mapping = dict(DEFAULT_RELATIONSHIP_MAP if relationship_map is None else relationship_map)
    output.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(config=DUCKDB_CONFIG)
    try:
        present = {name: _load(con, source, name) for name in _REQUIRED_COLUMNS}
        for name in ("CONCEPT", "VOCABULARY"):
            if not present[name]:
                raise LexiconError(f"{name}.csv is missing from {source}")

        selected = list(dict.fromkeys(vocabularies))
        known = {row[0] for row in con.execute("SELECT vocabulary_id FROM VOCABULARY").fetchall()}
        unknown = sorted(set(selected) - known)
        if unknown:
            raise LexiconError(f"vocabularies {unknown} are not in VOCABULARY.csv")

        _select_concepts(con, selected)
        _write_concepts(con, output, has_relationships=present["CONCEPT_RELATIONSHIP"])
        _write_synonyms(con, output, present["CONCEPT_SYNONYM"])
        unmapped = _write_relationships(
            con,
            output,
            mapping,
            has_relationships=present["CONCEPT_RELATIONSHIP"],
            ancestors=derive_ingredients and present["CONCEPT_ANCESTOR"],
        )
        _write_xrefs(con, output, present["CONCEPT_RELATIONSHIP"])
        counts = {
            table: _count(con, f"out_{table}")
            for table in ("concepts", "synonyms", "relationships", "xrefs")
        }
        declarations = [
            VocabularyDeclaration(
                vocabulary_id=vocabulary_id,
                release=version or "unversioned",
                license=license,
                description=name,
            )
            for vocabulary_id, name, version in con.execute(
                "SELECT vocabulary_id, vocabulary_name, vocabulary_version FROM VOCABULARY "
                "WHERE list_contains(?::VARCHAR[], vocabulary_id) ORDER BY vocabulary_id",
                [selected],
            ).fetchall()
        ]
    finally:
        con.close()

    imported = {d.vocabulary_id for d in declarations}
    # RxNorm Extension products and brands take RxNorm ingredients: their edges
    # cross between the two, which a declared family permits.
    families = [list(_RXNORM_FAMILY)] if set(_RXNORM_FAMILY) <= imported else []
    declaration = LexiconDeclaration(vocabularies=declarations, relationship_families=families)
    (output / "vocabularies.yaml").write_text(
        yaml.safe_dump(declaration.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return OmopImportReport(
        output_dir=output,
        vocabularies=[d.vocabulary_id for d in declarations],
        row_counts=counts,
        unmapped_relationships=unmapped,
    )


def _load(con: Any, source: Path, name: str) -> bool:
    path = source / f"{name}.csv"
    if not path.is_file():
        return False
    # Athena files are tab-delimited and unquoted; names may contain quotes.
    con.read_csv(str(path), header=True, delimiter="\t", quotechar="", all_varchar=True).create(
        name
    )
    columns = {row[0].lower() for row in con.execute(f"DESCRIBE {name}").fetchall()}
    missing = sorted(set(_REQUIRED_COLUMNS[name]) - columns)
    if missing:
        raise LexiconError(f"{path.name} is missing columns {missing}")
    return True


def _count(con: Any, table: str) -> int:
    row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _select_concepts(con: Any, selected: list[str]) -> None:
    classes = [[omop, tty] for omop, tty in RXNORM_CLASS_TO_TTY.items()]
    con.execute(
        "CREATE TABLE tty_map AS SELECT m[1] AS concept_class_id, m[2] AS tty "
        "FROM (SELECT unnest(?::VARCHAR[][]) AS m)",
        [classes],
    )
    con.execute(
        "CREATE TABLE selected AS SELECT c.*, CASE "
        "WHEN list_contains(?::VARCHAR[], c.vocabulary_id) THEN COALESCE(t.tty, c.concept_class_id) "
        "ELSE c.concept_class_id END AS concept_class "
        "FROM CONCEPT c LEFT JOIN tty_map t USING (concept_class_id) "
        "WHERE list_contains(?::VARCHAR[], c.vocabulary_id)",
        [list(_RXNORM_FAMILY), selected],
    )


def _date(column: str) -> str:
    return (
        f"CASE WHEN {column} IS NULL OR trim({column}) = '' THEN NULL "
        f"ELSE strftime(strptime(trim({column}), '%Y%m%d'), '%Y-%m-%d') END"
    )


def _write_concepts(con: Any, output: Path, *, has_relationships: bool) -> None:
    replaced = (
        "LEFT JOIN (SELECT r.concept_id_1, any_value(n.concept_code) AS replaced_by "
        "FROM CONCEPT_RELATIONSHIP r JOIN selected o ON o.concept_id = r.concept_id_1 "
        "JOIN selected n ON n.concept_id = r.concept_id_2 AND n.vocabulary_id = o.vocabulary_id "
        "WHERE r.relationship_id = 'Concept replaced by' GROUP BY r.concept_id_1) x "
        "ON x.concept_id_1 = s.concept_id"
        if has_relationships
        else "LEFT JOIN (SELECT NULL AS concept_id_1, NULL AS replaced_by WHERE false) x "
        "ON x.concept_id_1 = s.concept_id"
    )
    con.execute(
        "CREATE TABLE out_concepts AS SELECT s.vocabulary_id, s.concept_code, s.concept_name, "
        "s.concept_class, CASE upper(trim(COALESCE(s.invalid_reason, ''))) "
        "WHEN 'D' THEN 'obsolete' WHEN 'U' THEN 'remapped' ELSE 'active' END AS status, "
        f"{_date('s.valid_start_date')} AS valid_start_date, "
        f"{_date('s.valid_end_date')} AS valid_end_date, x.replaced_by "
        f"FROM selected s {replaced} ORDER BY s.vocabulary_id, s.concept_code"
    )
    _copy(con, "out_concepts", output / "concepts.parquet")


def _write_synonyms(con: Any, output: Path, present: bool) -> None:
    if present:
        con.execute(
            "CREATE TABLE out_synonyms AS SELECT DISTINCT s.vocabulary_id, s.concept_code, "
            "y.concept_synonym_name AS synonym, 'synonym' AS synonym_type, "
            "s.vocabulary_id AS source FROM CONCEPT_SYNONYM y "
            "JOIN selected s ON s.concept_id = y.concept_id "
            "WHERE trim(COALESCE(y.concept_synonym_name, '')) <> '' "
            "ORDER BY 1, 2, 3"
        )
    else:
        con.execute(
            "CREATE TABLE out_synonyms (vocabulary_id VARCHAR, concept_code VARCHAR, "
            "synonym VARCHAR, synonym_type VARCHAR, source VARCHAR)"
        )
    _copy(con, "out_synonyms", output / "synonyms.parquet")


def _write_relationships(
    con: Any,
    output: Path,
    mapping: dict[str, str],
    *,
    has_relationships: bool,
    ancestors: bool,
) -> dict[str, int]:
    con.execute(
        "CREATE TABLE out_relationships (vocabulary_id_1 VARCHAR, concept_code_1 VARCHAR, "
        "predicate VARCHAR, vocabulary_id_2 VARCHAR, concept_code_2 VARCHAR)"
    )
    unmapped: dict[str, int] = {}
    if has_relationships:
        pairs = [[relationship, predicate] for relationship, predicate in sorted(mapping.items())]
        con.execute(
            "CREATE TABLE predicate_map AS SELECT m[1] AS relationship_id, m[2] AS predicate "
            "FROM (SELECT unnest(?::VARCHAR[][]) AS m)",
            [pairs],
        )
        con.execute(
            "CREATE TABLE valid_edges AS SELECT r.relationship_id, a.vocabulary_id AS v1, "
            "a.concept_code AS c1, b.vocabulary_id AS v2, b.concept_code AS c2 "
            "FROM CONCEPT_RELATIONSHIP r JOIN selected a ON a.concept_id = r.concept_id_1 "
            "JOIN selected b ON b.concept_id = r.concept_id_2 "
            "WHERE trim(COALESCE(r.invalid_reason, '')) = '' AND r.concept_id_1 <> r.concept_id_2"
        )
        con.execute(
            "INSERT INTO out_relationships SELECT DISTINCT v1, c1, m.predicate, v2, c2 "
            "FROM valid_edges JOIN predicate_map m USING (relationship_id)"
        )
        unmapped = {
            relationship: int(count)
            for relationship, count in con.execute(
                "SELECT relationship_id, count(*) FROM valid_edges ANTI JOIN predicate_map "
                "USING (relationship_id) GROUP BY relationship_id ORDER BY relationship_id"
            ).fetchall()
        }
    if ancestors:
        con.execute(
            "INSERT INTO out_relationships SELECT DISTINCT d.vocabulary_id, d.concept_code, "
            "'has_ingredient', a.vocabulary_id, a.concept_code FROM CONCEPT_ANCESTOR x "
            "JOIN selected d ON d.concept_id = x.descendant_concept_id "
            "JOIN selected a ON a.concept_id = x.ancestor_concept_id "
            "WHERE a.concept_class_id = 'Ingredient' AND d.concept_id <> a.concept_id "
            "AND list_contains(?::VARCHAR[], d.vocabulary_id) "
            "AND list_contains(?::VARCHAR[], a.vocabulary_id)",
            [list(_RXNORM_FAMILY), list(_RXNORM_FAMILY)],
        )
    con.execute(
        "CREATE TABLE out_relationships_distinct AS SELECT DISTINCT * FROM out_relationships "
        "ORDER BY 1, 2, 3, 4, 5"
    )
    con.execute("DROP TABLE out_relationships")
    con.execute("ALTER TABLE out_relationships_distinct RENAME TO out_relationships")
    _copy(con, "out_relationships", output / "relationships.parquet")
    return unmapped


def _write_xrefs(con: Any, output: Path, has_relationships: bool) -> None:
    con.execute(
        "CREATE TABLE out_xrefs AS SELECT vocabulary_id, concept_code, 'OMOP_CONCEPT_ID' AS "
        "key_type, concept_id AS key_value, 'OMOP' AS assertion_source FROM selected "
        "UNION ALL SELECT vocabulary_id, concept_code, 'OMOP_STANDARD', concept_id, 'OMOP' "
        "FROM selected WHERE standard_concept = 'S'"
    )
    if has_relationships:
        con.execute(
            "INSERT INTO out_xrefs SELECT a.vocabulary_id, a.concept_code, 'OMOP_STANDARD', "
            "r.concept_id_2, 'OMOP' FROM CONCEPT_RELATIONSHIP r "
            "JOIN selected a ON a.concept_id = r.concept_id_1 "
            "WHERE r.relationship_id = 'Maps to' AND r.concept_id_1 <> r.concept_id_2 "
            "AND trim(COALESCE(r.invalid_reason, '')) = ''"
        )
    con.execute(
        "CREATE TABLE out_xrefs_distinct AS SELECT DISTINCT * FROM out_xrefs ORDER BY 1, 2, 3, 4"
    )
    con.execute("DROP TABLE out_xrefs")
    con.execute("ALTER TABLE out_xrefs_distinct RENAME TO out_xrefs")
    _copy(con, "out_xrefs", output / "xrefs.parquet")


def _copy(con: Any, table: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    con.execute(f"COPY {table} TO '{_sql_path(path)}' (FORMAT PARQUET)")


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")
