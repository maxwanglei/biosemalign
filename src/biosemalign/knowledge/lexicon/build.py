"""Compile contract tables into a verified, read-only lexicon bundle.

Validation is strict and happens before anything is written: a typo in a
column name, a synonym pointing at a concept that does not exist, or an
undeclared vocabulary fails the build with every problem listed, rather than
producing a bundle that silently misses matches.

Names are normalized and tokenized here with the engine's own
``normalize_text`` and ``tokenize`` — the exact functions applied to mentions —
so exact lookup can never disagree with the SAU about what "the same string"
means. That rules out SQL ``lower()``, which is not Unicode case folding.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from biosemalign._version import __version__
from biosemalign.enums import ConceptStatus
from biosemalign.exceptions import ConfigurationError, LexiconError
from biosemalign.knowledge.lexicon.contract import (
    INDEX_FILE,
    LEXICON_CONTRACT_VERSION,
    MANIFEST_FILE,
    NORMALIZER_VERSION,
    RANKED_INDEXES,
    TABLES,
    LexiconDeclaration,
    LexiconManifest,
    TableSpec,
    bundle_digest,
)
from biosemalign.util import normalize_text, tokenize

if TYPE_CHECKING:
    import duckdb

__all__ = [
    "DUCKDB_CONFIG",
    "build_lexicon",
    "file_sha256",
    "load_declaration",
    "require_duckdb",
    "trigrams",
]

#: Never install or autoload extensions: in an enclave that would mean an
#: outbound download, and the index needs none.
DUCKDB_CONFIG: dict[str, Any] = {
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
}

_BATCH_ROWS = 50_000
_STATUSES = sorted(status.value for status in ConceptStatus)


def require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ConfigurationError(
            "the local lexicon needs DuckDB; install biosemalign[lexicon]"
        ) from exc
    return duckdb


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_declaration(path: str | Path) -> LexiconDeclaration:
    """Read a vocabulary declaration from YAML or JSON."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
        return LexiconDeclaration.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise LexiconError(f"cannot read lexicon declaration {source}: {exc}") from exc


def trigrams(text: str) -> list[str]:
    """Character trigrams of each token, padded so word edges count.

    The ranked index for misspellings: "ExampelAlpha" shares most trigrams
    with "ExampleAlpha" though not a single whole token.
    """
    grams: list[str] = []
    for token in tokenize(text):
        padded = f"  {token} "
        grams.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    return grams


def build_lexicon(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    declaration: LexiconDeclaration,
    overwrite: bool = False,
    trigram_index: bool = True,
) -> LexiconManifest:
    """Validate the contract tables in ``source_dir`` and write a bundle.

    The trigram index roughly multiplies postings by the average name length;
    a very large bundle that never needs typo tolerance can omit it.
    """
    indexes = ["exact", *(k for k in RANKED_INDEXES if k != "trigram" or trigram_index)]
    duckdb = require_duckdb()
    source = Path(source_dir)
    output = Path(output_dir)
    inputs = _locate_tables(source)

    manifest_path = output / MANIFEST_FILE
    index_path = output / INDEX_FILE
    if manifest_path.exists() and not overwrite:
        raise LexiconError(f"{output} already holds a lexicon bundle; pass overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    partial = output / f".{INDEX_FILE}.partial"
    partial.unlink(missing_ok=True)

    try:
        with _connect(duckdb, partial) as con:
            for name, path in inputs.items():
                _load_raw(con, TABLES[name], path)
            _type_tables(con, inputs)
            _validate(con, declaration)
            corpus = _build_synonyms(con, declaration, indexes[1:])
            for kind in indexes[1:]:
                _build_postings(con, kind, corpus[kind])
            _drop_working_tables(con)
            _create_indexes(con)
            row_counts = _row_counts(con, declaration)
            con.execute("CHECKPOINT")
        index_sha256 = file_sha256(partial)
        manifest_path.unlink(missing_ok=True)
        os.replace(partial, index_path)
    finally:
        partial.unlink(missing_ok=True)

    manifest = LexiconManifest(
        contract_version=LEXICON_CONTRACT_VERSION,
        normalizer_version=NORMALIZER_VERSION,
        builder=f"biosemalign {__version__}",
        duckdb_version=duckdb.__version__,
        built_at=datetime.now(UTC),
        index_sha256=index_sha256,
        inputs_sha256={path.name: file_sha256(path) for path in inputs.values()},
        vocabularies=declaration.vocabularies,
        source_aliases=declaration.source_aliases,
        row_counts=row_counts,
        indexes=indexes,
        bundle_sha256=bundle_digest(
            contract_version=LEXICON_CONTRACT_VERSION,
            normalizer_version=NORMALIZER_VERSION,
            index_sha256=index_sha256,
            vocabularies=declaration.vocabularies,
            source_aliases=declaration.source_aliases,
        ),
    )
    # The manifest is written last: its presence marks a complete bundle.
    temporary = output / f".{MANIFEST_FILE}.partial"
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


# -- loading -----------------------------------------------------------------


@contextmanager
def _connect(duckdb: Any, path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(path), config=DUCKDB_CONFIG)
    try:
        yield con
    finally:
        con.close()


def _locate_tables(source: Path) -> dict[str, Path]:
    if not source.is_dir():
        raise LexiconError(f"lexicon source directory {source} does not exist")
    found: dict[str, Path] = {}
    for name, spec in TABLES.items():
        candidates = [
            path
            for path in (source / f"{name}{suffix}" for suffix in (".parquet", ".csv", ".tsv"))
            if path.exists()
        ]
        if len(candidates) > 1:
            raise LexiconError(
                f"table {name!r} is supplied more than once: {[p.name for p in candidates]}"
            )
        if candidates:
            found[name] = candidates[0]
        elif spec.mandatory:
            raise LexiconError(f"required table {name!r} is missing from {source}")
    return found


def _load_raw(con: duckdb.DuckDBPyConnection, spec: TableSpec, path: Path) -> None:
    if path.suffix == ".parquet":
        relation = con.read_parquet(str(path))
    else:
        relation = con.read_csv(
            str(path),
            header=True,
            all_varchar=True,
            delimiter="\t" if path.suffix == ".tsv" else ",",
        )
    relation.create(f"raw_{spec.name}")
    columns = [row[0] for row in con.execute(f"DESCRIBE raw_{spec.name}").fetchall()]
    missing = sorted(set(spec.required) - set(columns))
    unknown = sorted(set(columns) - set(spec.columns))
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing required columns {missing}")
        if unknown:
            problems.append(f"unknown columns {unknown} (allowed: {list(spec.columns)})")
        raise LexiconError(f"table {spec.name!r} ({path.name}): " + "; ".join(problems))


def _column(name: str, present: set[str]) -> str:
    return f'CAST("{name}" AS VARCHAR)' if name in present else "CAST(NULL AS VARCHAR)"


def _type_tables(con: duckdb.DuckDBPyConnection, inputs: dict[str, Path]) -> None:
    present = {
        name: {row[0] for row in con.execute(f"DESCRIBE raw_{name}").fetchall()} for name in inputs
    }
    for name, spec in TABLES.items():
        columns = present.get(name)
        if columns is None:
            selects = ", ".join(f"CAST(NULL AS VARCHAR) AS {column}" for column in spec.columns)
            con.execute(f"CREATE TABLE typed_{name} AS SELECT {selects} WHERE false")
            continue
        selects = ", ".join(f"{_column(column, columns)} AS {column}" for column in spec.columns)
        con.execute(f"CREATE TABLE typed_{name} AS SELECT {selects} FROM raw_{name}")


# -- validation --------------------------------------------------------------


def _validate(con: duckdb.DuckDBPyConnection, declaration: LexiconDeclaration) -> None:
    problems: list[str] = []

    def count(sql: str, params: list[Any] | None = None) -> int:
        row = con.execute(sql, params or []).fetchone()
        return int(row[0]) if row else 0

    identifier_columns = {
        "concepts": ("vocabulary_id", "concept_code", "concept_name"),
        "synonyms": ("vocabulary_id", "concept_code", "synonym"),
        "relationships": TABLES["relationships"].required,
        "xrefs": TABLES["xrefs"].required,
    }
    for table, columns in identifier_columns.items():
        for column in columns:
            blank = count(
                f"SELECT count(*) FROM typed_{table} WHERE {column} IS NULL OR trim({column}) = ''"
            )
            if blank:
                problems.append(f"{table}.{column}: {blank} blank value(s)")
            if column != "synonym" and column != "concept_name":
                padded = count(
                    f"SELECT count(*) FROM typed_{table} WHERE {column} <> trim({column})"
                )
                if padded:
                    problems.append(
                        f"{table}.{column}: {padded} value(s) with leading or trailing whitespace"
                    )

    duplicates = count(
        "SELECT count(*) FROM (SELECT vocabulary_id, concept_code FROM typed_concepts "
        "GROUP BY ALL HAVING count(*) > 1)"
    )
    if duplicates:
        problems.append(f"concepts: {duplicates} duplicated (vocabulary_id, concept_code) key(s)")

    invalid_status = con.execute(
        "SELECT DISTINCT lower(trim(status)) FROM typed_concepts WHERE status IS NOT NULL "
        "AND trim(status) <> '' AND NOT list_contains(?::VARCHAR[], lower(trim(status)))",
        [_STATUSES],
    ).fetchall()
    if invalid_status:
        problems.append(
            f"concepts.status: unknown values {sorted(row[0] for row in invalid_status)}; "
            f"allowed: {_STATUSES}"
        )

    for column in ("valid_start_date", "valid_end_date"):
        bad = count(
            f"SELECT count(*) FROM typed_concepts WHERE {column} IS NOT NULL "
            f"AND trim({column}) <> '' AND TRY_CAST({column} AS DATE) IS NULL"
        )
        if bad:
            problems.append(f"concepts.{column}: {bad} value(s) are not ISO dates")

    declared = [v.vocabulary_id for v in declaration.vocabularies]
    undeclared = con.execute(
        "SELECT DISTINCT vocabulary_id FROM typed_concepts "
        "WHERE NOT list_contains(?::VARCHAR[], vocabulary_id)",
        [declared],
    ).fetchall()
    if undeclared:
        problems.append(
            f"concepts: vocabularies {sorted(row[0] for row in undeclared)} are not declared"
        )
    empty = [
        vocabulary
        for vocabulary in declared
        if not count("SELECT count(*) FROM typed_concepts WHERE vocabulary_id = ?", [vocabulary])
    ]
    if empty:
        problems.append(f"declared vocabularies have no concepts: {empty}")

    references = {
        "synonyms": [("vocabulary_id", "concept_code")],
        "xrefs": [("vocabulary_id", "concept_code")],
        "relationships": [
            ("vocabulary_id_1", "concept_code_1"),
            ("vocabulary_id_2", "concept_code_2"),
        ],
    }
    for table, pairs in references.items():
        for vocabulary_column, code_column in pairs:
            dangling = count(
                f"SELECT count(*) FROM typed_{table} t ANTI JOIN typed_concepts c "
                f"ON t.{vocabulary_column} = c.vocabulary_id AND t.{code_column} = c.concept_code"
            )
            if dangling:
                problems.append(
                    f"{table}: {dangling} row(s) reference concepts absent from concepts "
                    f"({vocabulary_column}, {code_column})"
                )
    replaced = count(
        "SELECT count(*) FROM typed_concepts t WHERE replaced_by IS NOT NULL "
        "AND trim(replaced_by) <> '' AND NOT EXISTS (SELECT 1 FROM typed_concepts c "
        "WHERE c.vocabulary_id = t.vocabulary_id AND c.concept_code = t.replaced_by)"
    )
    if replaced:
        problems.append(
            f"concepts.replaced_by: {replaced} value(s) name no concept in the same vocabulary"
        )

    if problems:
        raise LexiconError("lexicon contract violations:\n- " + "\n- ".join(problems))


# -- index construction ------------------------------------------------------


def _build_synonyms(
    con: duckdb.DuckDBPyConnection,
    declaration: LexiconDeclaration,
    kinds: Sequence[str],
) -> dict[str, tuple[int, float]]:
    """Index every name, streaming ranked-index postings as names are read.

    Returns each ranked index's corpus statistics (documents, mean length).
    """
    # Every table is stored sorted by its lookup key, so the zone maps of a
    # read-only bundle prune most row groups for each point lookup. Concept
    # keys follow that order, so each vocabulary owns one contiguous range.
    con.execute(
        "CREATE TABLE concepts AS SELECT (row_number() OVER "
        "(ORDER BY vocabulary_id, concept_code) - 1)::INTEGER AS concept_key, "
        "vocabulary_id, concept_code, concept_name, "
        "NULLIF(trim(concept_class), '') AS concept_class, "
        "COALESCE(NULLIF(lower(trim(status)), ''), 'active') AS status, "
        "TRY_CAST(NULLIF(trim(valid_start_date), '') AS DATE) AS valid_start_date, "
        "TRY_CAST(NULLIF(trim(valid_end_date), '') AS DATE) AS valid_end_date, "
        "NULLIF(trim(replaced_by), '') AS replaced_by FROM typed_concepts "
        "ORDER BY vocabulary_id, concept_code"
    )
    aliases = [[raw, canonical] for raw, canonical in sorted(declaration.source_aliases.items())]
    con.execute(
        "CREATE TABLE source_aliases AS SELECT a[1] AS raw, a[2] AS canonical "
        "FROM (SELECT unnest(?::VARCHAR[][]) AS a)",
        [aliases],
    )
    con.execute(
        "CREATE TABLE synonym_rows AS SELECT r.vocabulary_id, r.concept_code, r.synonym, "
        "r.synonym_type, COALESCE(a.canonical, r.source) AS source FROM ("
        "SELECT vocabulary_id, concept_code, concept_name AS synonym, 'preferred' AS synonym_type, "
        "COALESCE(NULLIF(trim(name_source), ''), vocabulary_id) AS source FROM typed_concepts "
        "UNION ALL SELECT vocabulary_id, concept_code, synonym, NULLIF(trim(synonym_type), ''), "
        "COALESCE(NULLIF(trim(source), ''), vocabulary_id) FROM typed_synonyms) r "
        "LEFT JOIN source_aliases a ON a.raw = r.source"
    )
    con.execute(
        "CREATE TABLE sources AS SELECT (row_number() OVER (ORDER BY source))::INTEGER AS "
        "source_key, source FROM (SELECT DISTINCT source FROM synonym_rows)"
    )
    con.execute(
        "CREATE TABLE synonym_distinct AS SELECT row_number() OVER "
        "(ORDER BY s.vocabulary_id, s.concept_code, s.synonym, s.source) AS synonym_id, "
        "s.*, c.concept_key, k.source_key FROM ("
        "SELECT vocabulary_id, concept_code, synonym, source, min(synonym_type) AS synonym_type "
        "FROM synonym_rows GROUP BY ALL) s "
        "JOIN concepts c USING (vocabulary_id, concept_code) JOIN sources k USING (source)"
    )
    con.execute("CREATE TABLE synonym_text (synonym_id BIGINT, normalized VARCHAR)")
    for kind in kinds:
        con.execute(
            f"CREATE TABLE {kind}_staging (token VARCHAR, synonym_id INTEGER, tf INTEGER, "
            "length INTEGER, concept_key INTEGER, source_key INTEGER)"
        )
    import pyarrow as pa

    grams_of = {"bm25": tokenize, "trigram": trigrams}
    documents = dict.fromkeys(kinds, 0)
    lengths = dict.fromkeys(kinds, 0)
    # Stream through a separate cursor: executing the inserts below on the
    # connection that owns a pending result would silently end the stream
    # after its first batch.
    source = con.cursor()
    reader = source.execute(
        "SELECT synonym_id, synonym, concept_key, source_key FROM synonym_distinct "
        "ORDER BY synonym_id"
    ).to_arrow_reader(_BATCH_ROWS)
    for batch in reader:
        ids, texts, concepts, sources = (column.to_pylist() for column in batch.columns)
        normalized = [normalize_text(text) for text in texts]
        con.register(
            "text_batch",
            pa.table({"synonym_id": pa.array(ids, pa.int64()), "normalized": normalized}),
        )
        con.execute("INSERT INTO synonym_text SELECT * FROM text_batch")
        con.unregister("text_batch")
        for kind in kinds:
            columns: tuple[list[Any], ...] = ([], [], [], [], [], [])
            for synonym_id, text, norm, concept_key, source_key in zip(
                ids, texts, normalized, concepts, sources, strict=True
            ):
                grams = grams_of[kind](text) if norm else []
                if not grams:
                    continue
                documents[kind] += 1
                lengths[kind] += len(grams)
                for token, tf in Counter(grams).items():
                    for column, value in zip(
                        columns,
                        (token, synonym_id, tf, len(grams), concept_key, source_key),
                        strict=True,
                    ):
                        column.append(value)
            con.register(
                "posting_batch",
                pa.table(
                    {
                        "token": pa.array(columns[0], pa.string()),
                        **{
                            name: pa.array(values, pa.int32())
                            for name, values in zip(
                                ("synonym_id", "tf", "length", "concept_key", "source_key"),
                                columns[1:],
                                strict=True,
                            )
                        },
                    }
                ),
            )
            con.execute(f"INSERT INTO {kind}_staging SELECT * FROM posting_batch")
            con.unregister("posting_batch")
    source.close()
    written = con.execute("SELECT count(*) FROM synonym_text").fetchone()
    expected = con.execute("SELECT count(*) FROM synonym_distinct").fetchone()
    if written != expected:  # pragma: no cover - guards the streaming invariant above
        raise LexiconError(f"normalized {written} of {expected} names; the build is incomplete")
    con.execute(
        "CREATE TABLE synonyms AS SELECT d.synonym_id, d.vocabulary_id, d.concept_code, "
        "d.synonym, d.synonym_type, d.source, t.normalized "
        "FROM synonym_distinct d JOIN synonym_text t USING (synonym_id) WHERE t.normalized <> '' "
        "ORDER BY d.vocabulary_id, d.concept_code, d.synonym_id"
    )
    con.execute(
        "CREATE TABLE relationships AS SELECT DISTINCT vocabulary_id_1, concept_code_1, "
        "trim(predicate) AS predicate, vocabulary_id_2, concept_code_2 FROM typed_relationships "
        "ORDER BY vocabulary_id_1, concept_code_1, predicate, vocabulary_id_2, concept_code_2"
    )
    con.execute(
        "CREATE TABLE xrefs AS SELECT DISTINCT vocabulary_id, concept_code, trim(key_type) AS "
        "key_type, trim(key_value) AS key_value, NULLIF(trim(assertion_source), '') AS "
        "assertion_source FROM typed_xrefs ORDER BY vocabulary_id, concept_code, key_type, key_value"
    )
    return {
        kind: (documents[kind], lengths[kind] / documents[kind] if documents[kind] else 1.0)
        for kind in kinds
    }


def _drop_working_tables(con: duckdb.DuckDBPyConnection) -> None:
    for table in (
        "synonym_text",
        "synonym_distinct",
        "synonym_rows",
        "source_aliases",
        *(f"typed_{name}" for name in TABLES),
        *(f"raw_{name}" for name in TABLES),
    ):
        con.execute(f"DROP TABLE IF EXISTS {table}")


def _build_postings(con: duckdb.DuckDBPyConnection, kind: str, corpus: tuple[int, float]) -> None:
    """Sort one ranked index's postings by token and record its statistics.

    Postings are narrow — a token and five integers — so the sort that makes
    ``token IN (...)`` a pruned scan stays in memory for large bundles.
    """
    con.execute(
        f"CREATE TABLE {kind}_postings AS SELECT * FROM {kind}_staging ORDER BY token, synonym_id"
    )
    con.execute(f"DROP TABLE {kind}_staging")
    con.execute(
        f"CREATE TABLE {kind}_stats AS SELECT token, count(*)::INTEGER AS df "
        f"FROM {kind}_postings GROUP BY token ORDER BY token"
    )
    con.execute(
        f"CREATE TABLE {kind}_corpus AS SELECT ?::BIGINT AS n, ?::DOUBLE AS avgdl",
        [corpus[0], corpus[1]],
    )


def _create_indexes(con: duckdb.DuckDBPyConnection) -> None:
    for statement in (
        "CREATE UNIQUE INDEX concepts_key ON concepts (vocabulary_id, concept_code)",
        "CREATE UNIQUE INDEX concepts_concept_key ON concepts (concept_key)",
        "CREATE INDEX synonyms_normalized ON synonyms (normalized)",
        "CREATE INDEX synonyms_concept ON synonyms (vocabulary_id, concept_code)",
        "CREATE INDEX relationships_subject ON relationships (vocabulary_id_1, concept_code_1)",
        "CREATE INDEX xrefs_concept ON xrefs (vocabulary_id, concept_code)",
        "CREATE INDEX xrefs_key ON xrefs (key_type, key_value)",
    ):
        con.execute(statement)


def _row_counts(con: duckdb.DuckDBPyConnection, declaration: LexiconDeclaration) -> dict[str, int]:
    counts: dict[str, int] = {}
    tables = [
        name
        for (name,) in con.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        if name in {"concepts", "synonyms", "relationships", "xrefs"} or name.endswith("_postings")
    ]
    for table in tables:
        row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
        counts[table] = int(row[0]) if row else 0
    for vocabulary in declaration.vocabularies:
        row = con.execute(
            "SELECT count(*) FROM concepts WHERE vocabulary_id = ?", [vocabulary.vocabulary_id]
        ).fetchone()
        counts[f"concepts:{vocabulary.vocabulary_id}"] = int(row[0]) if row else 0
    return counts
