"""Bulk exact and cleaned dictionary coding, set-based via DuckDB (§5.4)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biosemalign._version import __version__
from biosemalign.enums import ReadinessStatus, SourceType
from biosemalign.exceptions import ConfigurationError, LexiconError
from biosemalign.kernel.guards import require_non_destructive
from biosemalign.knowledge.lexicon.build import bulk_config, file_sha256, require_duckdb
from biosemalign.knowledge.lexicon.contract import NORMALIZER_VERSION
from biosemalign.knowledge.lexicon.crosswalk import build_crosswalk
from biosemalign.knowledge.providers.lexicon import LexiconStore
from biosemalign.modules import OpContext, build
from biosemalign.modules.base import Cleaner
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.provenance import Provenance, RunContext
from biosemalign.schemas.request import MappingTask
from biosemalign.schemas.sau import (
    ContextEnvelope,
    ContextualAttributes,
    Mention,
    ReadinessAssessment,
    SemanticAlignmentUnit,
)
from biosemalign.schemas.source import SourceRecord
from biosemalign.targets import builtin_specs, load_target_spec
from biosemalign.util import normalize_text

__all__ = ["CODER_VERSION", "CodingReport", "code_terms", "term_key"]

CODER_VERSION = "0.2.0"
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
    coded_terms_exact: int = 0
    coded_rows_exact: int = 0
    coded_terms_cleaned: int = 0
    coded_rows_cleaned: int = 0
    codes: int = 0
    terms_reaching_target: int = 0
    rows_reaching_target: int = 0
    by_vocabulary: dict[str, dict[str, int]] = field(default_factory=dict)
    by_pass: dict[str, dict[str, int]] = field(default_factory=dict)
    target: str = ""
    granularity: str | None = None


def code_terms(
    bundle: str | Path,
    source: str | Path,
    output_dir: str | Path,
    *,
    text_column: str,
    id_columns: Sequence[str] = (),
    vocabularies: Sequence[str] | None = None,
    target: str | None = None,
    granularity: str | None = None,
    equivalence_keys: Sequence[str] | None = None,
    expected_sha256: str | None = None,
    overwrite: bool = False,
    profile: TaskProfile | None = None,
    cleaners: Sequence[Cleaner] | None = None,
) -> CodingReport:
    """Code every term of ``source`` against the bundle and write the outputs."""
    duckdb = require_duckdb()
    import pyarrow as pa

    exact_pass_name, cleaning_passes = _resolve_passes(profile=profile, cleaners=cleaners)

    bundle_path = Path(bundle)
    store = LexiconStore(bundle_path, expected_sha256=expected_sha256)
    manifest = store.manifest
    index = bundle_path / manifest.index_file
    served = [v.vocabulary_id for v in manifest.vocabularies]
    try:
        if target is not None:
            raw_target = target
        elif profile is not None:
            raw_target = profile.target.terminology
        else:
            known_targets = {s.target.casefold() for s in builtin_specs().values()}
            raw_target = next(
                (v for v in served if v.casefold() in known_targets),
                served[0],
            )
        selected = _resolve_vocabularies(store, served, vocabularies, profile)
        target_id = _served(store, raw_target)
    finally:
        store.close()
    keys = list(equivalence_keys) if equivalence_keys else served
    spec = (
        profile.target_spec()
        if (profile is not None and target is None)
        else load_target_spec(None, terminology=target_id)
    )
    resolved_granularity = (
        granularity
        if granularity is not None
        else (profile.target.level if profile is not None else next(iter(spec.granularities), None))
    )
    if resolved_granularity is not None and not spec.has_granularity(resolved_granularity):
        raise LexiconError(
            f"target spec {spec.label} has no granularity {resolved_granularity!r}; "
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

    ingredient_classes = (
        spec.spellings(spec.terminal_classes(resolved_granularity))
        if resolved_granularity is not None
        else ()
    )
    predicate = spec.predicate(resolved_granularity) if resolved_granularity is not None else None
    provenance_template = _bulk_provenance(profile)

    report = CodingReport(
        output_dir=output,
        bundle_sha256=manifest.bundle_sha256,
        target=target_id,
        granularity=resolved_granularity,
    )
    # Row numbers follow the input file, so insertion order is kept here.
    con = duckdb.connect(str(work), config={**bulk_config(), "preserve_insertion_order": True})
    try:
        _load_input(con, input_path, text_column, id_columns)
        _normalize(con, pa)
        con.execute(f"ATTACH '{_sql(index)}' AS lex (READ_ONLY)")
        _code(con, selected, manifest.vocabularies, exact_pass_name=exact_pass_name)
        for pass_name, pass_cleaners in cleaning_passes:
            _code_cleaned(
                con,
                pa,
                selected,
                pass_name=pass_name,
                cleaners=pass_cleaners,
                target=target_id,
                granularity=resolved_granularity,
                provenance=provenance_template,
            )
        build_crosswalk(
            con,
            concepts="(SELECT DISTINCT vocabulary_id, concept_code FROM matched)",
            output="crosswalk",
            schema="lex",
            target=target_id,
            family=manifest.family(target_id),
            equivalence_keys=keys,
            ingredient_classes=ingredient_classes,
            predicate=predicate,
        )
        _assemble(con, id_columns)
        for table in ("occurrences_out", "terms_out", "codes_out", "uncoded_out"):
            name = table.removesuffix("_out")
            path = output / f"{name}.parquet"
            path.unlink(missing_ok=True)
            con.execute(f"COPY {table} TO '{_sql(path)}' (FORMAT PARQUET)")
        _count(con, report, exact_pass_name=exact_pass_name)
        con.execute("DETACH lex")
    finally:
        con.close()
        work.unlink(missing_ok=True)
        work.with_suffix(".duckdb.wal").unlink(missing_ok=True)

    cleaner_ids = [
        cleaner.op_id for _, pass_cleaners in cleaning_passes for cleaner in pass_cleaners
    ]
    options: dict[str, Any] = {
        "text_column": text_column,
        "id_columns": list(id_columns),
        "vocabularies": selected,
        "target": target_id,
        "granularity": resolved_granularity,
        "target_spec": spec.label,
        "equivalence_keys": keys,
        "profile": (f"{profile.profile_name}@{profile.profile_version}" if profile else None),
        "cleaners": cleaner_ids,
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
            "coded_terms_exact": report.coded_terms_exact,
            "coded_rows_exact": report.coded_rows_exact,
            "coded_terms_cleaned": report.coded_terms_cleaned,
            "coded_rows_cleaned": report.coded_rows_cleaned,
            "codes": report.codes,
            "terms_reaching_target": report.terms_reaching_target,
            "rows_reaching_target": report.rows_reaching_target,
        },
        "by_vocabulary": report.by_vocabulary,
        "by_pass": report.by_pass,
    }
    (output / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report


# -- configuration helpers ---------------------------------------------------


def _resolve_vocabularies(
    store: LexiconStore,
    served: list[str],
    vocabularies: Sequence[str] | None,
    profile: TaskProfile | None,
) -> list[str]:
    if vocabularies is not None:
        return [_served(store, vocabulary) for vocabulary in vocabularies]
    if profile is not None:
        from_tiers: list[str] = []
        if profile.tiers:
            for tier in profile.tiers:
                for op in tier.match:
                    raw = op.options.get("vocabularies")
                    if isinstance(raw, list):
                        for v in raw:
                            canonical = _served(store, str(v))
                            if canonical not in from_tiers:
                                from_tiers.append(canonical)
        if from_tiers:
            return from_tiers
        from_resources: list[str] = []
        for resource in profile.knowledge.resources:
            served_id = store.vocabulary_id(resource)
            if served_id is not None and served_id not in from_resources:
                from_resources.append(served_id)
        if from_resources:
            return from_resources
    return served


def _resolve_passes(
    *,
    profile: TaskProfile | None,
    cleaners: Sequence[Cleaner] | None,
) -> tuple[str, list[tuple[str, tuple[Cleaner, ...]]]]:
    exact_pass_name = "exact"
    if profile is not None and profile.tiers:
        for tier in profile.tiers:
            if not tier.clean:
                exact_pass_name = tier.name
                break

    passes: list[tuple[str, tuple[Cleaner, ...]]] = []
    if cleaners is not None:
        resolved: list[Cleaner] = []
        for cleaner in cleaners:
            if not isinstance(cleaner, Cleaner):
                raise ConfigurationError(f"cleaner {cleaner!r} does not implement Cleaner protocol")
            resolved.append(cleaner)
        if resolved:
            passes.append(("cleaned", tuple(resolved)))
    elif profile is not None and profile.tiers:
        op_context = OpContext(profile=profile)
        for tier in profile.tiers:
            if not tier.clean:
                continue
            built = tuple(build("clean", op.op, op_context, op.options) for op in tier.clean)
            passes.append((tier.name, built))

    for _, pass_cleaners in passes:
        for cleaner in pass_cleaners:
            if bool(cleaner.uses_model):
                raise ConfigurationError(
                    f"bulk coding forbids model-backed cleaners ({cleaner.op_id!r}); "
                    "use local deterministic cleaners only"
                )
    return exact_pass_name, passes


def _bulk_provenance(profile: TaskProfile | None) -> Provenance:
    name = profile.profile_name if profile is not None else "bulk_coding"
    version = profile.profile_version if profile is not None else CODER_VERSION
    return RunContext(
        run_id="bulk",
        package_version=__version__,
        profile_name=name,
        profile_version=version,
    ).freeze()


def _bulk_sau(
    text: str,
    *,
    normalized: str | None = None,
    target: str,
    granularity: str | None,
    provenance: Provenance,
) -> SemanticAlignmentUnit:
    norm = normalized if normalized is not None else normalize_text(text)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12].upper()
    return SemanticAlignmentUnit(
        sau_id=f"SAU-{digest}",
        mapping_key=f"KEY-{digest}",
        source=SourceRecord(source_type=SourceType.STRUCTURED_FIELD, raw_value=text),
        mention=Mention(text=text, normalized_text=norm),
        context=ContextEnvelope(),
        attributes=ContextualAttributes(),
        task=MappingTask(
            operation="coding",
            mapping_intent="bulk_coding",
            target_terminology=target,
            desired_granularity=granularity,
        ),
        provenance=provenance,
        readiness=ReadinessAssessment(status=ReadinessStatus.READY_FOR_MAPPING),
    )


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
        "CREATE TABLE terms AS SELECT t.term_key, t.normalized, "
        "first(t.text ORDER BY o.n DESC, t.text) AS sample_text, "
        "sum(o.n)::BIGINT AS occurrences "
        "FROM text_terms t JOIN (SELECT text, count(*) AS n FROM occurrences GROUP BY text) o "
        "USING (text) WHERE t.term_key IS NOT NULL GROUP BY t.term_key, t.normalized"
    )


def _code(
    con: Any,
    vocabularies: list[str],
    declared: Sequence[Any],
    *,
    exact_pass_name: str,
) -> None:
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
        "bool_or(s.synonym_type = 'preferred') AS matched_preferred_name, "
        "CAST(? AS VARCHAR) AS coding_pass, "
        "[]::VARCHAR[] AS cleaned_by, "
        "CAST(NULL AS VARCHAR) AS cleaned_query "
        "FROM terms t JOIN lex.synonyms s ON s.normalized = t.normalized "
        "WHERE list_contains(?::VARCHAR[], s.vocabulary_id) GROUP BY ALL",
        [exact_pass_name, vocabularies],
    )


def _code_cleaned(
    con: Any,
    pa: Any,
    vocabularies: list[str],
    *,
    pass_name: str,
    cleaners: tuple[Cleaner, ...],
    target: str,
    granularity: str | None,
    provenance: Provenance,
) -> None:
    """Second coding pass over still-uncoded distinct terms through ``cleaners``."""
    if not cleaners:
        return
    con.execute(
        "CREATE OR REPLACE TABLE cleaned_queries ("
        "term_key VARCHAR, cleaned_query VARCHAR, cleaned_by VARCHAR[])"
    )
    reader = (
        con.cursor()
        .execute(
            "SELECT term_key, normalized, sample_text FROM terms "
            "WHERE term_key NOT IN (SELECT DISTINCT term_key FROM matched)"
        )
        .to_arrow_reader(_BATCH_ROWS)
    )
    for batch in reader:
        keys_col = batch.column(0).to_pylist()
        norms_col = batch.column(1).to_pylist()
        samples_col = batch.column(2).to_pylist()
        out_keys: list[str] = []
        out_queries: list[str] = []
        out_cleaners: list[list[str]] = []
        for tk, norm, sample_text in zip(keys_col, norms_col, samples_col, strict=True):
            view = _bulk_sau(
                sample_text,
                normalized=norm,
                target=target,
                granularity=granularity,
                provenance=provenance,
            )
            applied: list[str] = []
            for cleaner in cleaners:
                before = view
                cleaned = cleaner.clean(before.model_copy(deep=True))
                require_non_destructive(before, cleaned, cleaner.op_id)
                if (
                    cleaned.mention.retrieval_text != before.mention.retrieval_text
                    or cleaned.attributes != before.attributes
                ):
                    applied.append(cleaner.op_id)
                view = cleaned
            cleaned_norm = normalize_text(view.mention.retrieval_text or view.mention.text)
            if cleaned_norm and cleaned_norm != norm:
                out_keys.append(tk)
                out_queries.append(cleaned_norm)
                out_cleaners.append(applied)
        if not out_keys:
            continue
        con.register(
            "cleaned_batch",
            pa.table(
                {
                    "term_key": pa.array(out_keys, pa.string()),
                    "cleaned_query": pa.array(out_queries, pa.string()),
                    "cleaned_by": pa.array(out_cleaners, pa.list_(pa.string())),
                }
            ),
        )
        con.execute("INSERT INTO cleaned_queries SELECT * FROM cleaned_batch")
        con.unregister("cleaned_batch")

    con.execute(
        "INSERT INTO matched "
        "SELECT q.term_key, s.vocabulary_id, s.concept_code, "
        "list(DISTINCT s.synonym ORDER BY s.synonym) AS matched_names, "
        "list(DISTINCT s.source ORDER BY s.source) AS sources, "
        "bool_or(s.synonym_type = 'preferred') AS matched_preferred_name, "
        "CAST(? AS VARCHAR) AS coding_pass, "
        "any_value(q.cleaned_by) AS cleaned_by, "
        "any_value(q.cleaned_query) AS cleaned_query "
        "FROM cleaned_queries q JOIN lex.synonyms s ON s.normalized = q.cleaned_query "
        "WHERE list_contains(?::VARCHAR[], s.vocabulary_id) "
        "GROUP BY q.term_key, s.vocabulary_id, s.concept_code",
        [pass_name, vocabularies],
    )
    con.execute("DROP TABLE cleaned_queries")


def _assemble(con: Any, id_columns: Sequence[str]) -> None:
    ids = "".join(f', o."{c}"' for c in id_columns)
    con.execute(
        "CREATE TABLE codes_out AS SELECT m.term_key, t.normalized, m.vocabulary_id, "
        "m.concept_code, c.concept_name, c.concept_class, c.status, r.vocabulary_release, "
        "m.matched_names, m.sources, m.matched_preferred_name, x.link AS target_link, "
        "x.linked_concepts, x.ingredient_codes AS target_ingredients, x.ingredient_names AS "
        "target_ingredient_names, x.ingredient_sets, x.all_in_target, "
        "m.coding_pass, m.cleaned_by, m.cleaned_query "
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
        "signature, coding_pass, cleaned_by, cleaned_query FROM codes_out), agg AS ("
        "SELECT term_key, count(*)::INTEGER AS codes, "
        "list(DISTINCT vocabulary_id ORDER BY vocabulary_id) AS vocabularies, "
        "count(DISTINCT signature) FILTER (WHERE signature <> '')::INTEGER AS ingredient_sets, "
        "any_value(coding_pass) AS coding_pass, "
        "any_value(cleaned_by) AS cleaned_by, "
        "any_value(cleaned_query) AS cleaned_query "
        "FROM per_code GROUP BY term_key), ing AS (SELECT term_key, list(DISTINCT i "
        "ORDER BY i) AS target_ingredients FROM (SELECT term_key, unnest(target_ingredients) "
        "AS i FROM per_code) GROUP BY term_key) "
        "SELECT t.term_key, t.normalized, t.occurrences, COALESCE(a.codes, 0) AS codes, "
        "COALESCE(a.vocabularies, []::VARCHAR[]) AS vocabularies, "
        "COALESCE(i.target_ingredients, []::VARCHAR[]) AS target_ingredients, "
        "COALESCE(a.ingredient_sets, 0) AS ingredient_sets, "
        "a.coding_pass, COALESCE(a.cleaned_by, []::VARCHAR[]) AS cleaned_by, "
        "a.cleaned_query FROM terms t "
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


def _count(con: Any, report: CodingReport, *, exact_pass_name: str) -> None:
    def one(sql: str, params: list[Any] | None = None) -> int:
        row = con.execute(sql, params or []).fetchone()
        return int(row[0] or 0) if row else 0

    report.rows = one("SELECT count(*) FROM occurrences_out")
    report.blank_rows = one("SELECT count(*) FROM occurrences_out WHERE term_key IS NULL")
    report.terms = one("SELECT count(*) FROM terms_out")
    report.coded_terms = one("SELECT count(*) FROM terms_out WHERE codes > 0")
    report.coded_rows = one("SELECT sum(occurrences) FROM terms_out WHERE codes > 0")
    report.coded_terms_exact = one(
        "SELECT count(*) FROM terms_out WHERE coding_pass = ?", [exact_pass_name]
    )
    report.coded_rows_exact = one(
        "SELECT sum(occurrences) FROM terms_out WHERE coding_pass = ?", [exact_pass_name]
    )
    report.coded_terms_cleaned = report.coded_terms - report.coded_terms_exact
    report.coded_rows_cleaned = report.coded_rows - report.coded_rows_exact
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
    report.by_pass = {
        str(pass_name): {"terms": int(terms), "rows": int(rows or 0)}
        for pass_name, terms, rows in con.execute(
            "SELECT coding_pass, count(*), sum(occurrences) FROM terms_out "
            "WHERE coding_pass IS NOT NULL GROUP BY 1 ORDER BY 1"
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
        f"| Coded (≥1 exact code) | {report.coded_terms_exact:,} "
        f"({_percent(report.coded_terms_exact, report.terms)}) | {report.coded_rows_exact:,} "
        f"({_percent(report.coded_rows_exact, rows_with_text)}) |",
    ]
    if options.get("cleaners"):
        lines.extend(
            [
                f"| Coded after cleaning | {report.coded_terms_cleaned:,} "
                f"({_percent(report.coded_terms_cleaned, report.terms)}) | "
                f"{report.coded_rows_cleaned:,} "
                f"({_percent(report.coded_rows_cleaned, rows_with_text)}) |",
                f"| Coded (total) | {report.coded_terms:,} "
                f"({_percent(report.coded_terms, report.terms)}) | {report.coded_rows:,} "
                f"({_percent(report.coded_rows, rows_with_text)}) |",
            ]
        )
    lines += [
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
