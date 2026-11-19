"""Step 1a — bulk coding: every exact dictionary code for every term (v0.2).

Coding is the foundation of normalization. Every source term is matched
exactly — under the engine's own normalizer, the one the bundle was indexed
with — against every vocabulary of a lexicon bundle, and **every code is
kept**: all vocabularies, homonyms included, whether or not the code links to
the target terminology. Terms that match nothing are kept as uncoded. Nothing
here decides a mapping; resolution (Step 1b) and later steps start from it.

The work is set-based: distinct terms are normalized once and joined to the
bundle's name index in a single query, so millions of reported strings code
in minutes rather than the per-mention pipeline's ~0.1 s each. Each code is
annotated from the bundle's crosswalk with the target ingredients it reaches.

Outputs, written to one directory:

``occurrences.parquet``
    One row per input row: its row number, id columns, text, and ``term_key``.
``terms.parquet``
    One row per distinct normalized term: occurrences, codes, vocabularies,
    the target ingredients its codes reach, and how many distinct ingredient
    sets they propose.
``codes.parquet``
    One row per (term, code): the concept, every matched name and source, the
    vocabulary release, and the code's crosswalk annotation.
``uncoded.parquet``
    Terms with no exact match — the next step's worklist.
``coding_report.md`` and ``coding_manifest.json``
    Counts, the bundle digest, and the options, written last.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biosemalign._version import __version__
from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.lexicon.build import bulk_config, file_sha256, require_duckdb
from biosemalign.knowledge.lexicon.contract import NORMALIZER_VERSION
from biosemalign.knowledge.lexicon.crosswalk import build_crosswalk
from biosemalign.knowledge.providers.lexicon import LexiconStore
from biosemalign.targets import load_target_spec
from biosemalign.util import normalize_text

__all__ = ["CODER_VERSION", "CodingReport", "code_terms", "term_key"]

CODER_VERSION = "0.1.0"
MANIFEST = "coding_manifest.json"
_BATCH_ROWS = 50_000


def term_key(normalized: str) -> str:
    """A stable key for a normalized term, the same in every run and dataset."""
    return "T" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


@dataclass(slots=True)
class CodingReport:
    """What a coding run read and wrote."""

    output_dir: Path
    bundle_sha256: str
    rows: int = 0
    blank_rows: int = 0
    terms: int = 0
    coded_terms: int = 0
    coded_rows: int = 0
    codes: int = 0
    terms_reaching_target: int = 0
    rows_reaching_target: int = 0
    by_vocabulary: dict[str, dict[str, int]] = field(default_factory=dict)


def code_terms(
    bundle: str | Path,
    source: str | Path,
    output_dir: str | Path,
    *,
    text_column: str,
    id_columns: Sequence[str] = (),
    vocabularies: Sequence[str] | None = None,
    target: str = "RxNorm",
    granularity: str = "ingredient",
    equivalence_keys: Sequence[str] | None = None,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> CodingReport:
    """Code every term of ``source`` against the bundle and write the outputs."""
    duckdb = require_duckdb()
    import pyarrow as pa

    bundle_path = Path(bundle)
    store = LexiconStore(bundle_path, expected_sha256=expected_sha256)
    manifest = store.manifest
    index = bundle_path / manifest.index_file
    served = [v.vocabulary_id for v in manifest.vocabularies]
    try:
        selected = (
            [_served(store, vocabulary) for vocabulary in vocabularies] if vocabularies else served
        )
        target_id = _served(store, target)
    finally:
        store.close()
    keys = list(equivalence_keys) if equivalence_keys else served
    spec = load_target_spec(None, terminology=target_id)
    if spec.predicate(granularity) is None:
        raise LexiconError(
            f"target spec {spec.label} has no granularity {granularity!r}; "
            f"it has {sorted(spec.granularities)}"
        )

    input_path = Path(source)
    if not input_path.is_file():
        raise LexiconError(f"input {input_path} does not exist")
    output = Path(output_dir)
    if (output / MANIFEST).exists() and not overwrite:
        raise LexiconError(f"{output} already holds coding output; pass overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    (output / MANIFEST).unlink(missing_ok=True)
    work = output / ".coding.duckdb"
    work.unlink(missing_ok=True)

    report = CodingReport(output_dir=output, bundle_sha256=manifest.bundle_sha256)
    # Row numbers follow the input file, so insertion order is kept here.
    con = duckdb.connect(str(work), config={**bulk_config(), "preserve_insertion_order": True})
    try:
        _load_input(con, input_path, text_column, id_columns)
        _normalize(con, pa)
        con.execute(f"ATTACH '{_sql(index)}' AS lex (READ_ONLY)")
        _code(con, selected, manifest.vocabularies)
        build_crosswalk(
            con,
            concepts="(SELECT DISTINCT vocabulary_id, concept_code FROM matched)",
            output="crosswalk",
            schema="lex",
            target=target_id,
            family=manifest.family(target_id),
            equivalence_keys=keys,
            ingredient_classes=spec.spellings(spec.terminal_classes(granularity)),
            predicate=spec.predicate(granularity) or "has_ingredient",
        )
        _assemble(con, id_columns)
        for table in ("occurrences_out", "terms_out", "codes_out", "uncoded_out"):
            name = table.removesuffix("_out")
            path = output / f"{name}.parquet"
            path.unlink(missing_ok=True)
            con.execute(f"COPY {table} TO '{_sql(path)}' (FORMAT PARQUET)")
        _count(con, report)
        con.execute("DETACH lex")
    finally:
        con.close()
        work.unlink(missing_ok=True)
        work.with_suffix(".duckdb.wal").unlink(missing_ok=True)

    options = {
        "text_column": text_column,
        "id_columns": list(id_columns),
        "vocabularies": selected,
        "target": target_id,
        "granularity": granularity,
        "target_spec": spec.label,
        "equivalence_keys": keys,
    }
    (output / "coding_report.md").write_text(_markdown(report, options), encoding="utf-8")
    payload = {
        "coder": f"biosemalign {__version__} coder {CODER_VERSION}",
        "normalizer_version": NORMALIZER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "bundle": str(bundle_path),
        "bundle_sha256": manifest.bundle_sha256,
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "options": options,
        "counts": {
            "rows": report.rows,
            "blank_rows": report.blank_rows,
            "terms": report.terms,
            "coded_terms": report.coded_terms,
            "coded_rows": report.coded_rows,
            "codes": report.codes,
            "terms_reaching_target": report.terms_reaching_target,
            "rows_reaching_target": report.rows_reaching_target,
        },
        "by_vocabulary": report.by_vocabulary,
    }
    (output / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report


# -- steps -------------------------------------------------------------------


def _load_input(con: Any, path: Path, text_column: str, id_columns: Sequence[str]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        relation = con.read_parquet(str(path))
    elif suffix in {".csv", ".tsv", ".txt"}:
        relation = con.read_csv(
            str(path),
            header=True,
            all_varchar=True,
            delimiter="\t" if suffix in {".tsv", ".txt"} else ",",
        )
    else:
        raise LexiconError(f"input {path} must be .csv, .tsv, .txt, or .parquet")
    # A table's row ids follow insertion order, which follows the file.
    relation.create("input_raw")
    columns = {row[0] for row in con.execute("DESCRIBE input_raw").fetchall()}
    missing = [c for c in (text_column, *id_columns) if c not in columns]
    if missing:
        raise LexiconError(f"input {path.name} lacks columns {missing}; it has {sorted(columns)}")
    ids = "".join(f', CAST("{c}" AS VARCHAR) AS "{c}"' for c in id_columns)
    con.execute(
        f'CREATE TABLE occurrences AS SELECT rowid + 1 AS row_number{ids}, CAST("{text_column}" '
        "AS VARCHAR) AS text FROM input_raw ORDER BY rowid"
    )
    con.execute("DROP TABLE input_raw")


def _normalize(con: Any, pa: Any) -> None:
    """Normalize each distinct text once, in Python, as mentions are normalized."""
    con.execute("CREATE TABLE text_terms (text VARCHAR, normalized VARCHAR, term_key VARCHAR)")
    # Stream on a separate cursor: inserting on the connection that owns a
    # pending result would end the stream after its first batch.
    reader = (
        con.cursor()
        .execute("SELECT DISTINCT text FROM occurrences WHERE text IS NOT NULL")
        .to_arrow_reader(_BATCH_ROWS)
    )
    for batch in reader:
        texts = batch.column(0).to_pylist()
        normalized = [normalize_text(text) for text in texts]
        con.register(
            "text_batch",
            pa.table(
                {
                    "text": pa.array(texts, pa.string()),
                    "normalized": pa.array(normalized, pa.string()),
                    "term_key": pa.array(
                        [term_key(n) if n else None for n in normalized], pa.string()
                    ),
                }
            ),
        )
        con.execute("INSERT INTO text_terms SELECT * FROM text_batch")
        con.unregister("text_batch")
    con.execute(
        "CREATE TABLE terms AS SELECT t.term_key, t.normalized, sum(o.n)::BIGINT AS occurrences "
        "FROM text_terms t JOIN (SELECT text, count(*) AS n FROM occurrences GROUP BY text) o "
        "USING (text) WHERE t.term_key IS NOT NULL GROUP BY ALL"
    )


def _code(con: Any, vocabularies: list[str], declared: Sequence[Any]) -> None:
    releases = [[v.vocabulary_id, v.release] for v in declared]
    con.execute(
        "CREATE TABLE releases AS SELECT r[1] AS vocabulary_id, r[2] AS vocabulary_release "
        "FROM (SELECT unnest(?::VARCHAR[][]) AS r)",
        [releases],
    )
    con.execute(
        "CREATE TABLE matched AS SELECT t.term_key, s.vocabulary_id, s.concept_code, "
        "list(DISTINCT s.synonym ORDER BY s.synonym) AS matched_names, "
        "list(DISTINCT s.source ORDER BY s.source) AS sources, "
        "bool_or(s.synonym_type = 'preferred') AS matched_preferred_name "
        "FROM terms t JOIN lex.synonyms s ON s.normalized = t.normalized "
        "WHERE list_contains(?::VARCHAR[], s.vocabulary_id) GROUP BY ALL",
        [vocabularies],
    )


def _assemble(con: Any, id_columns: Sequence[str]) -> None:
    ids = "".join(f', o."{c}"' for c in id_columns)
    con.execute(
        "CREATE TABLE codes_out AS SELECT m.term_key, t.normalized, m.vocabulary_id, "
        "m.concept_code, c.concept_name, c.concept_class, c.status, r.vocabulary_release, "
        "m.matched_names, m.sources, m.matched_preferred_name, x.link AS target_link, "
        "x.linked_concepts, x.ingredient_codes AS target_ingredients, x.ingredient_names AS "
        "target_ingredient_names, x.ingredient_sets, x.all_in_target "
        "FROM matched m JOIN terms t USING (term_key) "
        "JOIN lex.concepts c ON c.vocabulary_id = m.vocabulary_id "
        "AND c.concept_code = m.concept_code "
        "LEFT JOIN releases r ON r.vocabulary_id = m.vocabulary_id "
        "LEFT JOIN crosswalk x ON x.vocabulary_id = m.vocabulary_id "
        "AND x.concept_code = m.concept_code "
        "ORDER BY m.term_key, m.vocabulary_id, m.concept_code"
    )
    con.execute(
        "CREATE TABLE terms_out AS WITH per_code AS (SELECT term_key, vocabulary_id, "
        "concept_code, target_ingredients, array_to_string(target_ingredients, '|') AS "
        "signature FROM codes_out), agg AS (SELECT term_key, count(*)::INTEGER AS codes, "
        "list(DISTINCT vocabulary_id ORDER BY vocabulary_id) AS vocabularies, "
        "count(DISTINCT signature) FILTER (WHERE signature <> '')::INTEGER AS ingredient_sets "
        "FROM per_code GROUP BY term_key), ing AS (SELECT term_key, list(DISTINCT i "
        "ORDER BY i) AS target_ingredients FROM (SELECT term_key, unnest(target_ingredients) "
        "AS i FROM per_code) GROUP BY term_key) "
        "SELECT t.term_key, t.normalized, t.occurrences, COALESCE(a.codes, 0) AS codes, "
        "COALESCE(a.vocabularies, []::VARCHAR[]) AS vocabularies, "
        "COALESCE(i.target_ingredients, []::VARCHAR[]) AS target_ingredients, "
        "COALESCE(a.ingredient_sets, 0) AS ingredient_sets FROM terms t "
        "LEFT JOIN agg a USING (term_key) LEFT JOIN ing i USING (term_key) "
        "ORDER BY t.occurrences DESC, t.normalized"
    )
    con.execute(
        f"CREATE TABLE occurrences_out AS SELECT o.row_number{ids}, o.text, x.term_key "
        "FROM occurrences o LEFT JOIN text_terms x ON x.text = o.text ORDER BY o.row_number"
    )
    con.execute(
        "CREATE TABLE uncoded_out AS SELECT term_key, normalized, occurrences FROM terms_out "
        "WHERE codes = 0 ORDER BY occurrences DESC, normalized"
    )


def _count(con: Any, report: CodingReport) -> None:
    def one(sql: str) -> int:
        row = con.execute(sql).fetchone()
        return int(row[0] or 0) if row else 0

    report.rows = one("SELECT count(*) FROM occurrences_out")
    report.blank_rows = one("SELECT count(*) FROM occurrences_out WHERE term_key IS NULL")
    report.terms = one("SELECT count(*) FROM terms_out")
    report.coded_terms = one("SELECT count(*) FROM terms_out WHERE codes > 0")
    report.coded_rows = one("SELECT sum(occurrences) FROM terms_out WHERE codes > 0")
    report.codes = one("SELECT count(*) FROM codes_out")
    report.terms_reaching_target = one(
        "SELECT count(*) FROM terms_out WHERE len(target_ingredients) > 0"
    )
    report.rows_reaching_target = one(
        "SELECT sum(occurrences) FROM terms_out WHERE len(target_ingredients) > 0"
    )
    report.by_vocabulary = {
        str(vocabulary): {"terms": int(terms), "rows": int(rows or 0)}
        for vocabulary, terms, rows in con.execute(
            "SELECT c.vocabulary_id, count(DISTINCT c.term_key), sum(t.occurrences) FROM "
            "(SELECT DISTINCT term_key, vocabulary_id FROM codes_out) c JOIN terms_out t "
            "USING (term_key) GROUP BY 1 ORDER BY 1"
        ).fetchall()
    }


# -- helpers -----------------------------------------------------------------


def _served(store: LexiconStore, vocabulary: str) -> str:
    canonical = store.vocabulary_id(vocabulary)
    if canonical is None:
        served = [v.vocabulary_id for v in store.manifest.vocabularies]
        raise LexiconError(f"the bundle does not serve {vocabulary!r}; it serves {served}")
    return canonical


def _sql(path: Path) -> str:
    return str(path).replace("'", "''")


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _markdown(report: CodingReport, options: dict[str, Any]) -> str:
    rows_with_text = report.rows - report.blank_rows
    lines = [
        "# Coding report",
        "",
        f"Bundle `{report.bundle_sha256}`; target `{options['target']}`.",
        "",
        "| | Terms (distinct) | Rows |",
        "|---|---|---|",
        f"| Input | {report.terms:,} | {report.rows:,} ({report.blank_rows:,} blank) |",
        f"| Coded (≥1 exact code) | {report.coded_terms:,} "
        f"({_percent(report.coded_terms, report.terms)}) | {report.coded_rows:,} "
        f"({_percent(report.coded_rows, rows_with_text)}) |",
        f"| A code reaches a {options['target']} ingredient | "
        f"{report.terms_reaching_target:,} ({_percent(report.terms_reaching_target, report.terms)})"
        f" | {report.rows_reaching_target:,} "
        f"({_percent(report.rows_reaching_target, rows_with_text)}) |",
        f"| Uncoded (next step's worklist) | {report.terms - report.coded_terms:,} | "
        f"{rows_with_text - report.coded_rows:,} |",
        "",
        f"{report.codes:,} (term, code) rows in `codes.parquet`.",
        "",
        "## Coded by vocabulary",
        "",
        "| Vocabulary | Terms | Rows |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {vocabulary} | {counts['terms']:,} | {counts['rows']:,} |"
        for vocabulary, counts in report.by_vocabulary.items()
    )
    lines += [
        "",
        "A term usually has codes in several vocabularies, so these rows overlap. "
        "Reaching an ingredient is an annotation, not a decision: resolution checks that "
        "the codes agree.",
        "",
    ]
    return "\n".join(lines)
