"""Link-completeness audit of a lexicon bundle (v0.2).

Exact coding is only as useful as the links behind it: a term coded to a
brand name that reaches no ingredient still has to go to a later step. The
audit measures, over every active concept of a bundle, how much of the
dictionary reaches the target terminology's ingredients — per class of the
target family (does every RxNorm brand name reach its ingredients?) and per
vocabulary (does every UNII or MeSH code link to an RxNorm concept?).

Run it on a bundle before and after adding links; it is the standing QA
report for the dictionary. Optionally it also writes the full code-to-
ingredient crosswalk as Parquet.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.lexicon.build import bulk_config, require_duckdb
from biosemalign.knowledge.lexicon.crosswalk import build_crosswalk
from biosemalign.knowledge.providers.lexicon import LexiconStore
from biosemalign.targets import load_target_spec

__all__ = ["AuditReport", "audit_bundle"]


@dataclass(slots=True)
class AuditReport:
    """Link completeness of one bundle."""

    bundle_sha256: str
    target: str
    family: list[str]
    #: (vocabulary, class) -> counts: active, reach (an ingredient), target (only target INs)
    family_classes: dict[str, dict[str, int]] = field(default_factory=dict)
    #: vocabulary -> counts: active, linked (to the family), reach, target, ambiguous
    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)


def audit_bundle(
    bundle: str | Path,
    output_dir: str | Path,
    *,
    target: str = "RxNorm",
    granularity: str = "ingredient",
    equivalence_keys: Sequence[str] | None = None,
    expected_sha256: str | None = None,
    write_crosswalk: bool = False,
) -> AuditReport:
    """Measure how much of a bundle reaches the target's ingredients; write the report."""
    duckdb = require_duckdb()
    path = Path(bundle)
    store = LexiconStore(path, expected_sha256=expected_sha256)
    manifest = store.manifest
    try:
        target_id = store.vocabulary_id(target)
    finally:
        store.close()
    if target_id is None:
        raise LexiconError(f"the bundle does not serve the target {target!r}")
    served = [v.vocabulary_id for v in manifest.vocabularies]
    family = list(manifest.family(target_id))
    spec = load_target_spec(None, terminology=target_id)
    if spec.predicate(granularity) is None:
        raise LexiconError(
            f"target spec {spec.label} has no granularity {granularity!r}; "
            f"it has {sorted(spec.granularities)}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # A file-backed working database lets intermediate tables spill to disk.
    work = output / ".audit.duckdb"
    work.unlink(missing_ok=True)
    con = duckdb.connect(str(work), config=bulk_config())
    try:
        index = str(path / manifest.index_file).replace("'", "''")
        con.execute(f"ATTACH '{index}' AS lex (READ_ONLY)")
        build_crosswalk(
            con,
            concepts="(SELECT vocabulary_id, concept_code FROM lex.concepts "
            "WHERE status IN ('active', 'unknown'))",
            output="crosswalk",
            schema="lex",
            target=target_id,
            family=family,
            equivalence_keys=list(equivalence_keys) if equivalence_keys else served,
            ingredient_classes=spec.spellings(spec.terminal_classes(granularity)),
            predicate=spec.predicate(granularity) or "has_ingredient",
        )
        report = AuditReport(bundle_sha256=manifest.bundle_sha256, target=target_id, family=family)
        for vocabulary, concept_class, active, reach, only in con.execute(
            "SELECT x.vocabulary_id, COALESCE(c.concept_class, '-'), count(*), "
            "count(*) FILTER (WHERE len(x.ingredient_codes) > 0), "
            "count(*) FILTER (WHERE len(x.ingredient_codes) > 0 AND x.all_in_target) "
            "FROM crosswalk x JOIN lex.concepts c USING (vocabulary_id, concept_code) "
            "WHERE list_contains(?::VARCHAR[], x.vocabulary_id) GROUP BY ALL ORDER BY 1, 2",
            [family],
        ).fetchall():
            report.family_classes[f"{vocabulary}|{concept_class}"] = {
                "active": int(active),
                "reach": int(reach),
                "target": int(only),
            }
        for vocabulary, active, linked, reach, only, ambiguous in con.execute(
            "SELECT vocabulary_id, count(*), count(*) FILTER (WHERE link <> 'none'), "
            "count(*) FILTER (WHERE len(ingredient_codes) > 0), "
            "count(*) FILTER (WHERE len(ingredient_codes) > 0 AND all_in_target), "
            "count(*) FILTER (WHERE ingredient_sets > 1) FROM crosswalk GROUP BY 1 ORDER BY 1"
        ).fetchall():
            report.vocabularies[str(vocabulary)] = {
                "active": int(active),
                "linked": int(linked),
                "reach": int(reach),
                "target": int(only),
                "ambiguous": int(ambiguous),
            }
        if write_crosswalk:
            crosswalk = output / "code_to_ingredient.parquet"
            crosswalk.unlink(missing_ok=True)
            con.execute(
                f"COPY crosswalk TO '{str(crosswalk).replace(chr(39), chr(39) * 2)}' "
                "(FORMAT PARQUET)"
            )
        con.execute("DETACH lex")
    finally:
        con.close()
        work.unlink(missing_ok=True)
        work.with_suffix(".duckdb.wal").unlink(missing_ok=True)

    (output / "audit.json").write_text(
        json.dumps(_payload(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / "audit.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _payload(report: AuditReport) -> dict[str, Any]:
    return {
        "bundle_sha256": report.bundle_sha256,
        "target": report.target,
        "family": report.family,
        "family_classes": report.family_classes,
        "vocabularies": report.vocabularies,
    }


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _markdown(report: AuditReport) -> str:
    lines = [
        "# Link-completeness audit",
        "",
        f"Bundle `{report.bundle_sha256}`; target `{report.target}` "
        f"(family: {', '.join(report.family)}). Current concepts only: status active, or "
        "unknown for sources that do not track it.",
        "",
        f"## {report.target} family: classes reaching an ingredient",
        "",
        f"| Vocabulary | Class | Current | Reach an ingredient | Only {report.target} INs |",
        "|---|---|---|---|---|",
    ]
    for key, counts in report.family_classes.items():
        vocabulary, concept_class = key.split("|", 1)
        lines.append(
            f"| {vocabulary} | {concept_class} | {counts['active']:,} | {counts['reach']:,} "
            f"({_percent(counts['reach'], counts['active'])}) | {counts['target']:,} "
            f"({_percent(counts['target'], counts['active'])}) |"
        )
    lines += [
        "",
        f"## Every vocabulary: linked to the {report.target} family",
        "",
        f"| Vocabulary | Current | Linked | Reach an ingredient | Only {report.target} INs "
        "| Ambiguous (>1 set) |",
        "|---|---|---|---|---|---|",
    ]
    for vocabulary, counts in report.vocabularies.items():
        lines.append(
            f"| {vocabulary} | {counts['active']:,} | {counts['linked']:,} "
            f"({_percent(counts['linked'], counts['active'])}) | {counts['reach']:,} "
            f"({_percent(counts['reach'], counts['active'])}) | {counts['target']:,} "
            f"({_percent(counts['target'], counts['active'])}) | {counts['ambiguous']:,} |"
        )
    lines += [
        "",
        "*Linked*: shares an equivalence key with a family concept (itself, for family "
        "concepts). *Reach an ingredient*: a linked concept is an active ingredient or has "
        "`has_ingredient` edges to one. *Ambiguous*: linked concepts propose different "
        "ingredient sets.",
        "",
    ]
    return "\n".join(lines)
