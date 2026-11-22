"""Derive missing dictionary links as overlay tables (v0.2).

A dictionary assembled from many sources is only as useful for normalization
as its links: an exact match on a brand name, a salt, or a UNII code resolves
to an RxNorm ingredient only if some edge or key leads there. This step
derives the missing links from the raw sources and writes them **beside** the
dictionary's own tables — never into them — as three overlays a build can
include or leave out:

``internal/``
    ``relationships``: ``has_ingredient`` edges inside the target family —
    R1 precise ingredients, R2 multiple-ingredient concepts, R3 brand names.
``external_id/``
    ``xrefs``: keys other sources assert by code — X1 FDA UNII Records,
    X2 MeSH registry numbers, X3 OHDSI mappings of SNOMED and ATC, X4 UMLS
    CUIs as shared keys.
``external_derived/``
    ``xrefs``: X5 product listings, linked by name *and* composition.

Every row names its ``assertion_source``; every rule reports what it linked
and, with examples, why it declined the rest (``link_report.md``). A copy of
the dictionary's declaration that adds the target family as a relationship
family is written alongside, for the build.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from biosemalign.domains.drug.tools.links.common import (
    INGREDIENT_CLASSES,
    LinkContext,
    RuleResult,
)
from biosemalign.domains.drug.tools.links.products import rule_products
from biosemalign.domains.drug.tools.links.rules import (
    rule_brand_names,
    rule_mesh_registry_numbers,
    rule_multiple_ingredients,
    rule_omop_mappings,
    rule_precise_ingredients,
    rule_umls_hub,
    rule_unii_records,
)
from biosemalign.domains.drug.tools.links.sources import (
    load_athena,
    load_mesh_registry_numbers,
    load_unii_records,
)
from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.lexicon.build import (
    bulk_config,
    load_contract_tables,
    load_declaration,
    require_duckdb,
)
from biosemalign.knowledge.lexicon.contract import LexiconDeclaration

__all__ = ["LINK_OVERLAYS", "LinkReport", "RuleResult", "derive_links"]

#: Overlay directories, in the order a build lists them.
LINK_OVERLAYS = ("internal", "external_id", "external_derived")
REPORT = "link_report.json"

#: Vocabularies whose dictionary content Athena supplies, beyond the family.
_ATHENA_VOCABULARIES = ("SNOMED", "ATC")


@dataclass(slots=True)
class LinkReport:
    """What a link derivation wrote."""

    output_dir: Path
    target: str
    family: list[str]
    results: list[RuleResult] = field(default_factory=list)
    overlay_rows: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def derive_links(
    base: str | Path,
    output_dir: str | Path,
    *,
    declaration: LexiconDeclaration | None = None,
    athena: str | Path | None = None,
    unii: str | Path | None = None,
    mesh: Sequence[str | Path] = (),
    product_evidence: str | Path | None = None,
    target: str = "RxNorm",
    family: Sequence[str] | None = None,
    allow_version_mismatch: bool = False,
    overwrite: bool = False,
) -> LinkReport:
    """Derive link overlays for the dictionary in ``base`` from the given sources.

    Every source is optional; rules whose source is absent are reported as
    skipped. R1 (through the dictionary's own OMOP ``Maps to`` keys), R2, and
    X4 need nothing beyond the dictionary.
    """
    duckdb = require_duckdb()
    base_dir = Path(base)
    if declaration is None:
        declaration = load_declaration(base_dir / "vocabularies.yaml")
    served = {v.vocabulary_id.casefold(): v.vocabulary_id for v in declaration.vocabularies}
    target_id = served.get(target.casefold())
    if target_id is None:
        raise LexiconError(f"the dictionary does not declare the target {target!r}")
    wanted_family = list(family) if family else [target_id, "RxNorm Extension"]
    members = [served[v.casefold()] for v in wanted_family if v.casefold() in served]
    if target_id not in members:
        members.insert(0, target_id)

    output = Path(output_dir)
    if (output / REPORT).exists() and not overwrite:
        raise LexiconError(f"{output} already holds links; pass overwrite to replace them")
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT).unlink(missing_ok=True)
    work = output / ".links.duckdb"
    work.unlink(missing_ok=True)

    report = LinkReport(output_dir=output, target=target_id, family=members)
    con = duckdb.connect(str(work), config=bulk_config())
    try:
        context = LinkContext(con=con, target=target_id, family=members, served=served)
        _load_base(context, base_dir)
        releases = {v.vocabulary_id: v.release for v in declaration.vocabularies}

        if athena is not None:
            athena_vocabularies = members + [
                served[v.casefold()] for v in _ATHENA_VOCABULARIES if v.casefold() in served
            ]
            load_athena(
                context,
                Path(athena),
                vocabularies=athena_vocabularies,
                releases=releases,
                allow_version_mismatch=allow_version_mismatch,
            )
        rule_precise_ingredients(context, athena=athena is not None)
        rule_multiple_ingredients(context, athena=athena is not None)
        if athena is not None:
            rule_brand_names(context)
        else:
            context.skip(
                RuleResult("R3", "Brand name (BN) → its products' ingredients", "internal"),
                "needs Athena (brand-to-product relationships)",
            )

        if unii is not None:
            unii_path = Path(unii)
            count = load_unii_records(context, unii_path, output / ".unii")
            context.notes.append(f"UNII Records: {count:,} records from {unii_path.name}")
            rule_unii_records(context, release=unii_path.stem)
        else:
            context.skip(RuleResult("X1", "UNII ↔ RxNorm and NCIt", "external_id"), "no --unii")
        if mesh:
            count = load_mesh_registry_numbers(context, [Path(p) for p in mesh])
            context.notes.append(f"MeSH: {count:,} preferred-concept registry numbers")
            rule_mesh_registry_numbers(context)
        else:
            context.skip(RuleResult("X2", "MeSH → UNII", "external_id"), "no --mesh")
        if athena is not None:
            rule_omop_mappings(context)
        else:
            context.skip(RuleResult("X3", "SNOMED and ATC → RxNorm", "external_id"), "no --athena")
        umls = context.vocabulary("UMLS")
        rule_umls_hub(context, release=releases.get(umls, "") if umls else "")
        if product_evidence is not None:
            rule_products(context, Path(product_evidence))
        else:
            context.skip(
                RuleResult("X5", "Product listings → brand or ingredient", "external_derived"),
                "no --product-evidence",
            )

        report.overlay_rows = _write_overlays(con, output)
        report.results = context.results
        report.notes = context.notes
    finally:
        con.close()
        work.unlink(missing_ok=True)
        work.with_suffix(".duckdb.wal").unlink(missing_ok=True)
        for leftover in (output / ".unii").glob("*") if (output / ".unii").is_dir() else ():
            leftover.unlink()
        if (output / ".unii").is_dir():
            (output / ".unii").rmdir()

    linked = declaration.model_copy(
        update={
            "relationship_families": _families(declaration, members),
        }
    )
    (output / "vocabularies.yaml").write_text(
        yaml.safe_dump(
            LexiconDeclaration.model_validate(linked.model_dump()).model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output / REPORT).write_text(json.dumps(_payload(report), indent=2) + "\n", encoding="utf-8")
    (output / "link_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


# -- steps -------------------------------------------------------------------


def _load_base(context: LinkContext, base: Path) -> None:
    con = context.con
    load_contract_tables(con, [base])
    con.execute(
        "CREATE TABLE base_concepts AS SELECT vocabulary_id, concept_code, concept_name, "
        "NULLIF(trim(concept_class), '') AS concept_class, "
        "COALESCE(NULLIF(lower(trim(status)), ''), 'active') AS status, "
        "NULLIF(trim(replaced_by), '') AS replaced_by FROM typed_concepts"
    )
    con.execute(
        "CREATE TABLE base_synonyms AS SELECT vocabulary_id, concept_code, synonym "
        "FROM typed_synonyms"
    )
    con.execute(
        "CREATE TABLE base_relationships AS SELECT vocabulary_id_1, concept_code_1, "
        "trim(predicate) AS predicate, vocabulary_id_2, concept_code_2 FROM typed_relationships"
    )
    con.execute(
        "CREATE TABLE base_xrefs AS SELECT vocabulary_id, concept_code, trim(key_type) AS "
        "key_type, trim(key_value) AS key_value FROM typed_xrefs"
    )
    for table in ("concepts", "synonyms", "relationships", "xrefs"):
        con.execute(f"DROP TABLE typed_{table}")
    con.execute(
        "CREATE TABLE family_ingredients AS SELECT vocabulary_id, concept_code, concept_name "
        "FROM base_concepts WHERE list_contains(?::VARCHAR[], vocabulary_id) "
        "AND list_contains(?::VARCHAR[], concept_class) AND status = 'active'",
        [context.family, list(INGREDIENT_CLASSES)],
    )
    con.execute(
        "CREATE TABLE emit_relationships (vocabulary_id_1 VARCHAR, concept_code_1 VARCHAR, "
        "predicate VARCHAR, vocabulary_id_2 VARCHAR, concept_code_2 VARCHAR, "
        "assertion_source VARCHAR, rule VARCHAR)"
    )
    con.execute(
        "CREATE TABLE emit_xrefs (vocabulary_id VARCHAR, concept_code VARCHAR, "
        "key_type VARCHAR, key_value VARCHAR, assertion_source VARCHAR, rule VARCHAR, "
        "overlay VARCHAR)"
    )


def _write_overlays(con: Any, output: Path) -> dict[str, int]:
    """Write each overlay: new rows only, one per link, every asserting rule named."""
    rows: dict[str, int] = {}
    directory = output / "internal"
    directory.mkdir(exist_ok=True)
    con.execute(
        "CREATE TABLE out_internal AS SELECT vocabulary_id_1, concept_code_1, predicate, "
        "vocabulary_id_2, concept_code_2, array_to_string(list_sort(list(DISTINCT "
        "assertion_source)), '; ') AS assertion_source FROM emit_relationships e "
        "WHERE NOT EXISTS (SELECT 1 FROM base_relationships b WHERE "
        "b.vocabulary_id_1 = e.vocabulary_id_1 AND b.concept_code_1 = e.concept_code_1 "
        "AND b.predicate = e.predicate AND b.vocabulary_id_2 = e.vocabulary_id_2 "
        "AND b.concept_code_2 = e.concept_code_2) GROUP BY ALL ORDER BY ALL"
    )
    rows["internal"] = _copy(con, "out_internal", directory / "relationships.parquet")
    for overlay in ("external_id", "external_derived"):
        directory = output / overlay
        directory.mkdir(exist_ok=True)
        con.execute(
            f"CREATE TABLE out_{overlay} AS SELECT vocabulary_id, concept_code, key_type, "
            "key_value, array_to_string(list_sort(list(DISTINCT assertion_source)), '; ') "
            "AS assertion_source FROM emit_xrefs e WHERE e.overlay = ? AND NOT EXISTS ("
            "SELECT 1 FROM base_xrefs b WHERE b.vocabulary_id = e.vocabulary_id "
            "AND b.concept_code = e.concept_code AND b.key_type = e.key_type "
            "AND b.key_value = e.key_value) GROUP BY vocabulary_id, concept_code, key_type, "
            "key_value ORDER BY ALL",
            [overlay],
        )
        rows[overlay] = _copy(con, f"out_{overlay}", directory / "xrefs.parquet")
    return rows


def _copy(con: Any, table: str, path: Path) -> int:
    path.unlink(missing_ok=True)
    con.execute(f"COPY {table} TO '{str(path).replace(chr(39), chr(39) * 2)}' (FORMAT PARQUET)")
    row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _families(declaration: LexiconDeclaration, members: list[str]) -> list[list[str]]:
    others = [
        family
        for family in declaration.relationship_families
        if not {m.casefold() for m in family} & {m.casefold() for m in members}
    ]
    return [*others, members] if len(members) > 1 else others


# -- report ------------------------------------------------------------------


def _payload(report: LinkReport) -> dict[str, Any]:
    return {
        "target": report.target,
        "family": report.family,
        "overlay_rows": report.overlay_rows,
        "rules": [
            {
                "rule": r.rule,
                "description": r.description,
                "overlay": r.overlay,
                "skipped": r.skipped,
                "considered": r.considered,
                "linked": r.linked,
                "emitted": r.emitted,
                "dropped": r.dropped,
                "examples": r.examples,
            }
            for r in report.results
        ],
        "notes": report.notes,
    }


def _markdown(report: LinkReport) -> str:
    lines = [
        "# Link derivation report",
        "",
        f"Target `{report.target}`; family {', '.join(report.family)}.",
        "",
        "| Rule | What | Overlay | Considered | Linked | Rows emitted | Declined |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(report.results, key=lambda r: r.rule):
        if r.skipped:
            lines.append(
                f"| {r.rule} | {r.description} | {r.overlay} | skipped: {r.skipped} | | | |"
            )
            continue
        declined = ", ".join(f"{reason} {count:,}" for reason, count in r.dropped.items())
        lines.append(
            f"| {r.rule} | {r.description} | {r.overlay} | {r.considered:,} | {r.linked:,} | "
            f"{r.emitted:,} | {declined or '—'} |"
        )
    lines += ["", "New rows per overlay (rows the dictionary already had are not repeated):", ""]
    lines += [f"- `{name}/`: {count:,}" for name, count in report.overlay_rows.items()]
    lines += ["", "## Declined, with examples", ""]
    for r in sorted(report.results, key=lambda r: r.rule):
        for reason, examples in r.examples.items():
            lines.append(f"**{r.rule} {reason}** ({r.dropped.get(reason, 0):,})")
            lines += [f"- {example}" for example in examples]
            lines.append("")
    if report.notes:
        lines += ["## Notes", "", "```", *report.notes, "```", ""]
    return "\n".join(lines)
