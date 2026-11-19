"""X5: product listings (Drugs@FDA, Orange Book, EMA Article 57) → the RxNorm family.

These lists carry no RxNorm code or UNII, so a product is linked only when
two independent facts agree:

1. its ingredient field resolves — every ingredient, under the engine's
   normalizer — to exactly one active target-family ingredient, through an
   ingredient's or precise ingredient's own name, or through a UNII name whose
   UNII carries an RXCUI (so "paracetamol" resolves to acetaminophen);
2. its product name is either the name of exactly one active brand whose
   ingredient set is exactly that resolved set, or — for a generic product —
   the ingredient's own name.

Anything else is left unlinked and counted. Links land in the
``external_derived`` overlay: they rest on names, corroborated by
composition, rather than on a code someone asserted.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.lexicon.links.common import (
    BRAND_CLASSES,
    LinkContext,
    RuleResult,
)
from biosemalign.util import normalize_text

__all__ = ["rule_products"]

_BATCH = 50_000
_SPLIT = re.compile(r"\s*;\s*")
_SUBSPLIT = re.compile(r"\s*(?:,|/|\+| and )\s*")

Key = tuple[str, str]


def rule_products(context: LinkContext, evidence: Path) -> RuleResult:
    """Link each listed product to a brand or ingredient when name and composition agree."""
    import pyarrow as pa

    result = RuleResult(
        "X5",
        "Drugs@FDA / Orange Book / EMA products → brand or ingredient (name + composition)",
        overlay="external_derived",
    )
    con = context.con
    relation = (
        f"read_csv('{str(evidence).replace(chr(39), chr(39) * 2)}', header=true, all_varchar=true)"
    )
    columns = {str(row[0]) for row in context.rows(f"DESCRIBE SELECT * FROM {relation}")}
    required = {"vocabulary_id", "concept_code", "product_name", "ingredient_text"}
    if not required <= columns:
        raise LexiconError(
            f"{evidence.name} lacks columns {sorted(required - columns)}; it has {sorted(columns)}"
        )

    ingredient_of = _ingredient_names(context)
    brands = _brands(context)
    family = frozenset(context.family)

    rows = con.cursor().execute(
        f"SELECT e.vocabulary_id, e.concept_code, e.product_name, e.ingredient_text, "
        f"b.concept_code IS NOT NULL FROM {relation} e LEFT JOIN base_concepts b "
        f"ON b.vocabulary_id = e.vocabulary_id AND b.concept_code = e.concept_code"
    )
    outcomes: dict[str, list[Any]] = {"subject": [], "outcome": []}
    emitted: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    while batch := rows.fetchmany(_BATCH):
        for vocabulary, code, product_name, ingredient_text, known in batch:
            subject = f"{vocabulary}:{code} {product_name!r} [{ingredient_text}]"
            outcome, target = _link(
                known,
                product_name,
                ingredient_text,
                ingredient_of=ingredient_of,
                brands=brands,
                family=family,
            )
            outcomes["subject"].append(subject)
            outcomes["outcome"].append(outcome)
            if target is None:
                continue
            row = (str(vocabulary), str(code), target[0], target[1])
            if row in seen:
                continue
            seen.add(row)
            for name, value in zip(
                ("vocabulary_id", "concept_code", "key_type", "key_value"), row, strict=True
            ):
                emitted[name].append(value)
            emitted["assertion_source"].append(
                f"derived: {vocabulary} product name + ingredients ({outcome.removeprefix('linked_')})"
            )
    con.register("x5_outcomes", pa.table(outcomes))
    con.execute(
        "CREATE TABLE x5_out AS SELECT subject, CASE WHEN outcome LIKE 'linked_%' THEN 'linked' "
        "ELSE outcome END AS outcome, outcome AS detail FROM x5_outcomes"
    )
    con.unregister("x5_outcomes")
    con.register(
        "x5_emitted",
        pa.table(
            {
                name: pa.array(emitted.get(name, []), pa.string())
                for name in (
                    "vocabulary_id",
                    "concept_code",
                    "key_type",
                    "key_value",
                    "assertion_source",
                )
            }
        ),
    )
    con.execute("INSERT INTO emit_xrefs SELECT *, 'X5', 'external_derived' FROM x5_emitted")
    con.unregister("x5_emitted")
    linked_kinds = context.rows(
        "SELECT detail, count(*) FROM x5_out WHERE outcome = 'linked' GROUP BY 1 ORDER BY 1"
    )
    context.notes.append(
        "X5 links by kind: " + ", ".join(f"{kind} {count:,}" for kind, count in linked_kinds)
    )
    return context.record(
        result,
        outcomes="x5_out",
        emitted_sql="SELECT count(*) FROM emit_xrefs WHERE rule = ?",
        emitted_params=["X5"],
    )


def _link(
    known: bool,
    product_name: str | None,
    ingredient_text: str | None,
    *,
    ingredient_of: dict[str, set[tuple[Key, Key]]],
    brands: dict[str, list[tuple[Key, frozenset[Key]]]],
    family: frozenset[str],
) -> tuple[str, Key | None]:
    """The outcome for one product, and the family concept it links to."""
    if not known:
        return "product_not_in_dictionary", None
    resolved = _resolve(ingredient_text or "", ingredient_of)
    if resolved is None:
        return "ingredients_unresolved", None
    ingredients = frozenset(ingredient for _, ingredient in resolved)
    name = normalize_text(product_name or "")
    if not name:
        return "no_product_name", None

    candidates = brands.get(name, [])
    matching = [brand for brand, brand_set in candidates if brand_set == ingredients]
    if len(matching) == 1:
        return "linked_brand", matching[0]
    if len(matching) > 1:
        return "brand_ambiguous", None

    # A generic product is named for its single ingredient. The name may be
    # both an RxNorm-family concept's and a UNII substance's; the family's wins.
    if len(resolved) == 1 and name == normalize_text(ingredient_text or ""):
        concepts = {concept for concept, _ in ingredient_of.get(name, set())}
        preferred = {c for c in concepts if c[0] in family} or concepts
        if len(preferred) == 1:
            return "linked_generic", next(iter(preferred))
        return "generic_name_ambiguous", None
    if candidates:
        return "brand_ingredients_differ", None
    return "no_matching_brand", None


def _resolve(
    text: str, ingredient_of: dict[str, set[tuple[Key, Key]]]
) -> list[tuple[Key, Key]] | None:
    """Each listed ingredient → (the concept named, its target-family ingredient)."""
    found: list[tuple[Key, Key]] = []
    for piece in _SPLIT.split(text.strip()):
        if not piece:
            continue
        hit = _unique(normalize_text(piece), ingredient_of)
        if hit is not None:
            found.append(hit)
            continue
        # "paracetamol, caffeine" and the like: try the piece's own parts.
        parts = [p for p in _SUBSPLIT.split(piece) if p]
        hits = [_unique(normalize_text(p), ingredient_of) for p in parts]
        if len(parts) < 2 or any(h is None for h in hits):
            return None
        found.extend(h for h in hits if h is not None)
    return found or None


def _unique(name: str, ingredient_of: dict[str, set[tuple[Key, Key]]]) -> tuple[Key, Key] | None:
    """The one ingredient a name resolves to (the named concept may vary)."""
    hits = ingredient_of.get(name)
    if not hits or len({ingredient for _, ingredient in hits}) != 1:
        return None
    # Prefer the ingredient's own concept as the concept named.
    ordered = sorted(hits, key=lambda hit: (hit[0] != hit[1], hit[0]))
    return ordered[0]


def _ingredient_names(context: LinkContext) -> dict[str, set[tuple[Key, Key]]]:
    """Normalized ingredient names → (concept named, target-family ingredient).

    Names of active family ingredients and precise ingredients (through R1),
    and UNII names whose UNII links to one of them (through X1).
    """
    unii = context.vocabulary("UNII")
    union = [
        "SELECT i.vocabulary_id AS v, i.concept_code AS c, i.vocabulary_id AS iv, "
        "i.concept_code AS ic FROM family_ingredients i",
        "SELECT p.v, p.c, p.iv, p.ic FROM pin_ingredient p JOIN base_concepts b "
        "ON b.vocabulary_id = p.v AND b.concept_code = p.c WHERE b.status = 'active'",
    ]
    if unii is not None and _exists(context, "x1_out"):
        union.append(
            "SELECT DISTINCT x.vocabulary_id, x.concept_code, COALESCE(p.iv, i.vocabulary_id), "
            "COALESCE(p.ic, i.concept_code) FROM emit_xrefs x "
            "LEFT JOIN pin_ingredient p ON p.v = x.key_type AND p.c = x.key_value "
            "LEFT JOIN family_ingredients i ON i.vocabulary_id = x.key_type "
            "AND i.concept_code = x.key_value WHERE x.rule = 'X1' AND x.vocabulary_id = '"
            + unii.replace("'", "''")
            + "' AND (p.v IS NOT NULL OR i.vocabulary_id IS NOT NULL)"
        )
    context.con.execute("CREATE TABLE x5_named AS " + " UNION ".join(union))
    names: dict[str, set[tuple[Key, Key]]] = defaultdict(set)
    rows = context.con.cursor().execute(
        "SELECT t.name, n.v, n.c, n.iv, n.ic FROM x5_named n JOIN ("
        "SELECT vocabulary_id, concept_code, concept_name AS name FROM base_concepts "
        "UNION ALL SELECT vocabulary_id, concept_code, synonym FROM base_synonyms) t "
        "ON t.vocabulary_id = n.v AND t.concept_code = n.c"
    )
    while batch := rows.fetchmany(_BATCH):
        for name, v, c, iv, ic in batch:
            normalized = normalize_text(name or "")
            if normalized:
                names[normalized].add(((v, c), (iv, ic)))
    return names


def _brands(context: LinkContext) -> dict[str, list[tuple[Key, frozenset[Key]]]]:
    """Normalized brand names → (brand, its ingredient set), for brands with one."""
    sets: dict[Key, set[Key]] = defaultdict(set)
    for v, c, iv, ic in context.rows(
        "SELECT r.v, r.c, r.iv, r.ic FROM (SELECT vocabulary_id_1 AS v, concept_code_1 AS c, "
        "vocabulary_id_2 AS iv, concept_code_2 AS ic FROM emit_relationships "
        "WHERE predicate = 'has_ingredient' UNION SELECT vocabulary_id_1, concept_code_1, "
        "vocabulary_id_2, concept_code_2 FROM base_relationships "
        "WHERE predicate = 'has_ingredient') r JOIN base_concepts b ON b.vocabulary_id = r.v "
        "AND b.concept_code = r.c AND b.status = 'active' "
        "AND list_contains(?::VARCHAR[], b.concept_class) "
        "JOIN family_ingredients i ON i.vocabulary_id = r.iv AND i.concept_code = r.ic",
        [list(BRAND_CLASSES)],
    ):
        sets[(v, c)].add((iv, ic))
    brands: dict[str, list[tuple[Key, frozenset[Key]]]] = defaultdict(list)
    keys = [[v, c] for v, c in sets]
    if not keys:
        return brands
    context.con.execute(
        "CREATE TABLE x5_brands AS SELECT k[1] AS v, k[2] AS c FROM "
        "(SELECT unnest(?::VARCHAR[][]) AS k)",
        [keys],
    )
    rows = context.con.cursor().execute(
        "SELECT DISTINCT t.name, b.v, b.c FROM x5_brands b JOIN ("
        "SELECT vocabulary_id, concept_code, concept_name AS name FROM base_concepts "
        "UNION ALL SELECT vocabulary_id, concept_code, synonym FROM base_synonyms) t "
        "ON t.vocabulary_id = b.v AND t.concept_code = b.c"
    )
    seen: set[tuple[str, Key]] = set()
    while batch := rows.fetchmany(_BATCH):
        for name, v, c in batch:
            normalized = normalize_text(name or "")
            if normalized and (normalized, (v, c)) not in seen:
                seen.add((normalized, (v, c)))
                brands[normalized].append(((v, c), frozenset(sets[(v, c)])))
    return brands


def _exists(context: LinkContext, table: str) -> bool:
    return bool(
        context.scalar(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        )
    )
