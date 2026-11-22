"""Code-to-ingredient crosswalk over a built lexicon bundle (v0.2).

Coding attaches every dictionary code a term matches; the crosswalk says, for
each code, which target-terminology ingredients it reaches. It is computed
from the bundle itself — the same edges and keys the Step 1 tier sees — so it
can never disagree with the bundle it describes:

* a concept of the target family (RxNorm, RxNorm Extension) links to itself;
* any other concept links to the family concepts it shares an equivalence key
  with — a direct key (``RxNorm``, ``1191``) or a hub key such as a UNII code
  or UMLS CUI that both carry;
* each linked family concept contributes its ingredients: itself when it is an
  active concept of the requested level, else its targets along the level's
  predicate (``has_ingredient`` for RxNorm ingredients). Callers take the
  classes and predicate from the target spec.

The crosswalk annotates; it decides nothing. ``ingredient_sets`` counts how
many different ingredient sets the linked concepts propose: 1 means they
agree, more means the code is ambiguous at ingredient level.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["CROSSWALK_COLUMNS", "build_crosswalk"]

#: Columns of the crosswalk table, in order.
CROSSWALK_COLUMNS = (
    "vocabulary_id",
    "concept_code",
    "link",
    "linked_concepts",
    "ingredient_codes",
    "ingredient_names",
    "ingredient_sets",
    "all_in_target",
)


def build_crosswalk(
    con: Any,
    *,
    concepts: str,
    output: str,
    schema: str,
    target: str,
    family: Sequence[str],
    equivalence_keys: Sequence[str],
    ingredient_classes: Sequence[str] = ("IN",),
    predicate: str | None = "has_ingredient",
) -> None:
    """Create table ``output`` with one crosswalk row per row of ``concepts``.

    ``concepts`` names a table or query with ``vocabulary_id`` and
    ``concept_code``; ``schema`` is where the bundle's tables live (an
    attached database, or ``main``).
    """
    families = [list(family)]
    keys = list(equivalence_keys)
    classes = list(ingredient_classes)
    con.execute(
        f"CREATE OR REPLACE TABLE xw_input AS SELECT DISTINCT vocabulary_id, concept_code "
        f"FROM {concepts}"
    )
    # Keys of every input concept outside the family: its identity, plus its
    # cross-references of a permitted type.
    con.execute(
        f"CREATE OR REPLACE TABLE xw_input_keys AS "
        f"SELECT vocabulary_id, concept_code, vocabulary_id AS key_type, concept_code AS key_value "
        f"FROM xw_input WHERE NOT list_contains(?::VARCHAR[], vocabulary_id) "
        f"AND list_contains(?::VARCHAR[], vocabulary_id) "
        f"UNION SELECT x.vocabulary_id, x.concept_code, x.key_type, x.key_value "
        f"FROM {schema}.xrefs x JOIN xw_input i USING (vocabulary_id, concept_code) "
        f"WHERE NOT list_contains(?::VARCHAR[], x.vocabulary_id) "
        f"AND list_contains(?::VARCHAR[], x.key_type)",
        [families[0], keys, families[0], keys],
    )
    # Keys family concepts carry, restricted to those the inputs hold: a
    # family concept's identity, or one of its permitted cross-references.
    con.execute(
        f"CREATE OR REPLACE TABLE xw_family_keys AS "
        f"SELECT c.vocabulary_id, c.concept_code, k.key_type, k.key_value "
        f"FROM (SELECT DISTINCT key_type, key_value FROM xw_input_keys) k "
        f"JOIN {schema}.concepts c ON c.vocabulary_id = k.key_type "
        f"AND c.concept_code = k.key_value "
        f"WHERE list_contains(?::VARCHAR[], c.vocabulary_id) "
        f"UNION SELECT x.vocabulary_id, x.concept_code, x.key_type, x.key_value "
        f"FROM {schema}.xrefs x JOIN (SELECT DISTINCT key_type, key_value FROM xw_input_keys) k "
        f"USING (key_type, key_value) WHERE list_contains(?::VARCHAR[], x.vocabulary_id)",
        [families[0], families[0]],
    )
    con.execute(
        "CREATE OR REPLACE TABLE xw_links AS "
        "SELECT vocabulary_id, concept_code, 'self' AS link, vocabulary_id AS linked_vocabulary, "
        "concept_code AS linked_code FROM xw_input WHERE list_contains(?::VARCHAR[], vocabulary_id) "
        "UNION SELECT i.vocabulary_id, i.concept_code, 'key', f.vocabulary_id, f.concept_code "
        "FROM xw_input_keys i JOIN xw_family_keys f USING (key_type, key_value)",
        [families[0]],
    )
    # Targets of every linked family concept (identity match + optional predicate walk).
    class_filter = "list_contains(?::VARCHAR[], c.concept_class) AND " if classes else ""
    self_params: list[Any] = [classes] if classes else []
    if predicate is None:
        con.execute(
            f"CREATE OR REPLACE TABLE xw_ingredients AS "
            f"SELECT l.linked_vocabulary, l.linked_code, c.vocabulary_id AS ingredient_vocabulary, "
            f"c.concept_code AS ingredient_code, c.concept_name AS ingredient_name "
            f"FROM (SELECT DISTINCT linked_vocabulary, linked_code FROM xw_links) l "
            f"JOIN {schema}.concepts c ON c.vocabulary_id = l.linked_vocabulary "
            f"AND c.concept_code = l.linked_code "
            f"WHERE {class_filter}c.status = 'active'",
            self_params,
        )
    else:
        walk_params: list[Any] = [
            *self_params,
            predicate,
            families[0],
            *([classes] if classes else []),
        ]
        con.execute(
            f"CREATE OR REPLACE TABLE xw_ingredients AS "
            f"SELECT l.linked_vocabulary, l.linked_code, c.vocabulary_id AS ingredient_vocabulary, "
            f"c.concept_code AS ingredient_code, c.concept_name AS ingredient_name "
            f"FROM (SELECT DISTINCT linked_vocabulary, linked_code FROM xw_links) l "
            f"JOIN {schema}.concepts c ON c.vocabulary_id = l.linked_vocabulary "
            f"AND c.concept_code = l.linked_code "
            f"WHERE {class_filter}c.status = 'active' "
            f"UNION SELECT l.linked_vocabulary, l.linked_code, c.vocabulary_id, c.concept_code, "
            f"c.concept_name FROM (SELECT DISTINCT linked_vocabulary, linked_code FROM xw_links) l "
            f"JOIN {schema}.relationships r ON r.vocabulary_id_1 = l.linked_vocabulary "
            f"AND r.concept_code_1 = l.linked_code AND r.predicate = ? "
            f"JOIN {schema}.concepts c ON c.vocabulary_id = r.vocabulary_id_2 "
            f"AND c.concept_code = r.concept_code_2 "
            f"WHERE list_contains(?::VARCHAR[], c.vocabulary_id) "
            f"AND {class_filter}c.status = 'active'",
            walk_params,
        )
    # One output row per input. Lists are built unordered and sorted after
    # aggregation: an ordered aggregate cannot spill to disk.
    con.execute(
        "CREATE OR REPLACE TABLE xw_linked AS SELECT vocabulary_id, concept_code, "
        "min(link) AS link, list_sort(list_distinct(list(linked_vocabulary || ':' || "
        "linked_code))) AS linked_concepts FROM xw_links GROUP BY ALL"
    )
    con.execute(
        "CREATE OR REPLACE TABLE xw_union AS SELECT l.vocabulary_id, l.concept_code, "
        "list_sort(list_distinct(list(g.ingredient_vocabulary || ':' || g.ingredient_code))) "
        "AS ingredient_codes, list_sort(list_distinct(list(g.ingredient_name))) AS "
        "ingredient_names, bool_and(g.ingredient_vocabulary = ?) AS all_in_target "
        "FROM xw_links l JOIN xw_ingredients g ON g.linked_vocabulary = l.linked_vocabulary "
        "AND g.linked_code = l.linked_code GROUP BY ALL",
        [target],
    )
    # How many different ingredient sets the linked concepts propose matters
    # only where an input links to more than one family concept.
    con.execute(
        "CREATE OR REPLACE TABLE xw_multi AS SELECT l.vocabulary_id, l.concept_code, "
        "l.linked_vocabulary, l.linked_code FROM xw_links l JOIN (SELECT vocabulary_id, "
        "concept_code FROM xw_linked WHERE len(linked_concepts) > 1) m "
        "USING (vocabulary_id, concept_code)"
    )
    con.execute(
        "CREATE OR REPLACE TABLE xw_sets AS SELECT m.vocabulary_id, m.concept_code, "
        "count(DISTINCT s.signature)::INTEGER AS n FROM xw_multi m JOIN (SELECT "
        "linked_vocabulary, linked_code, array_to_string(list_sort(list(ingredient_vocabulary "
        "|| ':' || ingredient_code)), '|') AS signature FROM xw_ingredients WHERE "
        "(linked_vocabulary, linked_code) IN (SELECT linked_vocabulary, linked_code "
        "FROM xw_multi) GROUP BY ALL) s ON s.linked_vocabulary = m.linked_vocabulary "
        "AND s.linked_code = m.linked_code GROUP BY ALL"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE {output} AS "
        "SELECT i.vocabulary_id, i.concept_code, COALESCE(k.link, 'none') AS link, "
        "COALESCE(k.linked_concepts, []::VARCHAR[]) AS linked_concepts, "
        "COALESCE(g.ingredient_codes, []::VARCHAR[]) AS ingredient_codes, "
        "COALESCE(g.ingredient_names, []::VARCHAR[]) AS ingredient_names, "
        "(CASE WHEN g.ingredient_codes IS NULL THEN 0 ELSE COALESCE(s.n, 1) END)::INTEGER "
        "AS ingredient_sets, COALESCE(g.all_in_target, false) AS all_in_target "
        "FROM xw_input i LEFT JOIN xw_linked k USING (vocabulary_id, concept_code) "
        "LEFT JOIN xw_union g USING (vocabulary_id, concept_code) "
        "LEFT JOIN xw_sets s USING (vocabulary_id, concept_code)"
    )
    for table in (
        "xw_input",
        "xw_input_keys",
        "xw_family_keys",
        "xw_links",
        "xw_ingredients",
        "xw_linked",
        "xw_union",
        "xw_multi",
        "xw_sets",
    ):
        con.execute(f"DROP TABLE IF EXISTS {table}")
