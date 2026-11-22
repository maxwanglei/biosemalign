"""Link rules R1–R3 (inside the target family) and X1–X4 (ID-based, external).

Every rule reads the base tables and the loaded sources, writes an outcome
row for each subject it considered — ``linked`` or the reason it declined —
and inserts what it derived into ``emit_relationships`` or ``emit_xrefs``.
Targets are always active concepts of the dictionary; a rule never invents a
code the dictionary lacks.
"""

from __future__ import annotations

from biosemalign.domains.drug.tools.links.common import (
    BRAND_CLASSES,
    BRANDED_PRODUCT_CLASSES,
    MULTI_INGREDIENT_CLASSES,
    PRECISE_INGREDIENT_CLASSES,
    LinkContext,
    RuleResult,
)

__all__ = [
    "rule_brand_names",
    "rule_mesh_registry_numbers",
    "rule_multiple_ingredients",
    "rule_omop_mappings",
    "rule_precise_ingredients",
    "rule_umls_hub",
    "rule_unii_records",
]

_RELATIONSHIP_EMITTED = "SELECT count(*) FROM emit_relationships WHERE rule = ?"
_XREF_EMITTED = "SELECT count(*) FROM emit_xrefs WHERE rule = ?"


def _subject(v: str, c: str, n: str | None = None) -> str:
    """SQL for a display string naming a concept."""
    label = f" || ' ' || COALESCE({n}, '')" if n else ""
    return f"{v} || ':' || {c}{label}"


# -- internal: R1 to R3 ------------------------------------------------------


def rule_precise_ingredients(context: LinkContext, *, athena: bool) -> RuleResult:
    """R1: a precise ingredient (salt, ester) → its one base ingredient.

    Evidence: Athena's ``Form of`` (or its reverse ``Has form``) and the
    dictionary's own OMOP ``Maps to`` key. When both speak they must agree;
    either alone suffices. Exactly one active ingredient is required.
    """
    result = RuleResult(
        "R1", "Precise ingredient (PIN) → its base ingredient (IN)", overlay="internal"
    )
    con = context.con
    con.execute(
        "CREATE TABLE r1_subjects AS SELECT vocabulary_id AS v, concept_code AS c, "
        "concept_name AS n FROM base_concepts WHERE list_contains(?::VARCHAR[], vocabulary_id) "
        "AND list_contains(?::VARCHAR[], concept_class)",
        [context.family, list(PRECISE_INGREDIENT_CLASSES)],
    )
    if athena:
        con.execute(
            "CREATE TABLE r1_athena AS SELECT DISTINCT x.v, x.c, x.iv, x.ic FROM ("
            "SELECT v1 AS v, c1 AS c, v2 AS iv, c2 AS ic FROM ath_rel "
            "WHERE relationship_id = 'Form of' AND k1 = 'Precise Ingredient' "
            "AND k2 = 'Ingredient' UNION SELECT v2, c2, v1, c1 FROM ath_rel "
            "WHERE relationship_id = 'Has form' AND k1 = 'Ingredient' "
            "AND k2 = 'Precise Ingredient') x JOIN family_ingredients i "
            "ON i.vocabulary_id = x.iv AND i.concept_code = x.ic"
        )
    else:
        con.execute("CREATE TABLE r1_athena (v VARCHAR, c VARCHAR, iv VARCHAR, ic VARCHAR)")
    con.execute(
        "CREATE TABLE r1_maps AS SELECT DISTINCT s.v, s.c, i.vocabulary_id AS iv, "
        "i.concept_code AS ic FROM r1_subjects s "
        "JOIN base_xrefs m ON m.vocabulary_id = s.v AND m.concept_code = s.c "
        "AND m.key_type = 'OMOP_STANDARD' "
        "JOIN base_xrefs own ON own.vocabulary_id = s.v AND own.concept_code = s.c "
        "AND own.key_type = 'OMOP_CONCEPT_ID' AND own.key_value <> m.key_value "
        "JOIN base_xrefs t ON t.key_type = 'OMOP_CONCEPT_ID' AND t.key_value = m.key_value "
        "JOIN family_ingredients i ON i.vocabulary_id = t.vocabulary_id "
        "AND i.concept_code = t.concept_code"
    )
    con.execute(
        "CREATE TABLE r1_out AS WITH "
        "a AS (SELECT v, c, list_sort(list(iv || '|' || ic)) AS s FROM r1_athena GROUP BY ALL), "
        "m AS (SELECT v, c, list_sort(list(iv || '|' || ic)) AS s FROM r1_maps GROUP BY ALL) "
        f"SELECT s.v, s.c, {_subject('s.v', 's.c', 's.n')} AS subject, "
        "CASE WHEN a.s IS NULL AND m.s IS NULL THEN 'no_ingredient_found' "
        "WHEN a.s IS NOT NULL AND m.s IS NOT NULL AND a.s <> m.s THEN 'sources_disagree' "
        "WHEN len(COALESCE(a.s, m.s)) > 1 THEN 'several_ingredients' ELSE 'linked' END "
        "AS outcome, COALESCE(a.s, m.s)[1] AS pick, "
        "CASE WHEN a.s IS NOT NULL AND m.s IS NOT NULL THEN 'OMOP Form of; OMOP Maps to' "
        "WHEN a.s IS NOT NULL THEN 'OMOP Form of' ELSE 'OMOP Maps to' END AS source "
        "FROM r1_subjects s LEFT JOIN a USING (v, c) LEFT JOIN m USING (v, c)"
    )
    # Later rules read precise ingredients through this table.
    con.execute(
        "CREATE TABLE pin_ingredient AS SELECT v, c, split_part(pick, '|', 1) AS iv, "
        "split_part(pick, '|', 2) AS ic FROM r1_out WHERE outcome = 'linked'"
    )
    con.execute(
        "INSERT INTO emit_relationships SELECT p.v, p.c, 'has_ingredient', p.iv, p.ic, "
        "o.source, 'R1' FROM pin_ingredient p JOIN r1_out o USING (v, c)"
    )
    return context.record(
        result, outcomes="r1_out", emitted_sql=_RELATIONSHIP_EMITTED, emitted_params=["R1"]
    )


def rule_multiple_ingredients(context: LinkContext, *, athena: bool) -> RuleResult:
    """R2: a multiple-ingredient concept → each of its ingredients.

    RxNorm composes a MIN's name from its ingredients' names, joined by
    " / ". Every part must name exactly one active ingredient — preferring
    the MIN's own vocabulary — or a precise ingredient resolved by R1. Any
    Athena ingredient edge of the MIN must agree.
    """
    result = RuleResult(
        "R2", "Multiple ingredients (MIN) → each ingredient (IN)", overlay="internal"
    )
    con = context.con
    con.execute(
        "CREATE TABLE r2_subjects AS SELECT vocabulary_id AS v, concept_code AS c, "
        "concept_name AS n FROM base_concepts WHERE list_contains(?::VARCHAR[], vocabulary_id) "
        "AND list_contains(?::VARCHAR[], concept_class)",
        [context.family, list(MULTI_INGREDIENT_CLASSES)],
    )
    con.execute(
        "CREATE TABLE r2_parts AS SELECT v, c, generate_subscripts(parts, 1) AS pos, "
        "trim(unnest(parts)) AS part FROM (SELECT v, c, string_split(lower(n), ' / ') AS parts "
        "FROM r2_subjects)"
    )
    con.execute(
        "CREATE TABLE r2_names AS SELECT lower(concept_name) AS name, vocabulary_id AS iv, "
        "concept_code AS ic, 'IN' AS kind FROM family_ingredients "
        "UNION SELECT lower(b.concept_name), p.iv, p.ic, 'PIN' FROM pin_ingredient p "
        "JOIN base_concepts b ON b.vocabulary_id = p.v AND b.concept_code = p.c "
        "WHERE b.status = 'active'"
    )
    con.execute(
        "CREATE TABLE r2_matches AS SELECT p.v, p.c, p.pos, "
        "COALESCE(list_distinct(list(n.iv || '|' || n.ic) FILTER (WHERE n.kind = 'IN' "
        "AND n.iv = p.v)), []) AS same_in, COALESCE(list_distinct(list(n.iv || '|' || n.ic) "
        "FILTER (WHERE n.kind = 'IN')), []) AS any_in, COALESCE(list_distinct(list(n.iv || "
        "'|' || n.ic) FILTER (WHERE n.kind = 'PIN')), []) AS via_pin FROM r2_parts p LEFT JOIN r2_names n ON n.name = p.part GROUP BY ALL"
    )
    con.execute(
        "CREATE TABLE r2_resolved AS SELECT v, c, pos, CASE "
        "WHEN len(same_in) = 1 THEN same_in[1] "
        "WHEN len(same_in) = 0 AND len(any_in) = 1 THEN any_in[1] "
        "WHEN len(any_in) = 0 AND len(via_pin) = 1 THEN via_pin[1] END AS pick, CASE "
        "WHEN len(same_in) > 1 OR (len(same_in) = 0 AND len(any_in) > 1) "
        "OR (len(any_in) = 0 AND len(via_pin) > 1) THEN 'ambiguous' "
        "WHEN len(any_in) = 0 AND len(via_pin) = 0 THEN 'unresolved' ELSE 'resolved' END "
        "AS state FROM r2_matches"
    )
    if athena:
        con.execute(
            "CREATE TABLE r2_athena AS SELECT x.v, x.c, list_sort(list(DISTINCT x.iv || '|' || "
            "x.ic)) AS s FROM (SELECT v1 AS v, c1 AS c, v2 AS iv, c2 AS ic FROM ath_rel "
            "WHERE k1 = 'Multiple Ingredients' AND k2 = 'Ingredient' "
            "UNION SELECT v2, c2, v1, c1 FROM ath_rel WHERE k2 = 'Multiple Ingredients' "
            "AND k1 = 'Ingredient') x GROUP BY ALL"
        )
    else:
        con.execute("CREATE TABLE r2_athena (v VARCHAR, c VARCHAR, s VARCHAR[])")
    con.execute(
        "CREATE TABLE r2_out AS WITH agg AS (SELECT v, c, count(*) AS parts, "
        "count(DISTINCT pick) AS picks, bool_or(state = 'ambiguous') AS ambiguous, "
        "bool_or(state = 'unresolved') AS unresolved, list_sort(list(DISTINCT pick)) AS s "
        "FROM r2_resolved GROUP BY ALL) "
        f"SELECT s.v, s.c, {_subject('s.v', 's.c', 's.n')} AS subject, CASE "
        "WHEN a.parts IS NULL OR a.parts < 2 THEN 'not_a_combination_name' "
        "WHEN a.ambiguous THEN 'part_ambiguous' WHEN a.unresolved THEN 'part_unresolved' "
        "WHEN a.picks <> a.parts THEN 'repeated_ingredient' "
        "WHEN t.s IS NOT NULL AND t.s <> a.s THEN 'athena_disagrees' ELSE 'linked' END "
        "AS outcome, a.s AS picks, "
        "CASE WHEN t.s IS NOT NULL THEN 'RxNorm MIN name components; OMOP ingredient edges' "
        "ELSE 'RxNorm MIN name components' END AS source "
        "FROM r2_subjects s LEFT JOIN agg a USING (v, c) LEFT JOIN r2_athena t USING (v, c)"
    )
    con.execute(
        "INSERT INTO emit_relationships SELECT v, c, 'has_ingredient', "
        "split_part(pick, '|', 1), split_part(pick, '|', 2), source, 'R2' "
        "FROM (SELECT v, c, source, unnest(picks) AS pick FROM r2_out WHERE outcome = 'linked')"
    )
    return context.record(
        result, outcomes="r2_out", emitted_sql=_RELATIONSHIP_EMITTED, emitted_params=["R2"]
    )


def rule_brand_names(context: LinkContext) -> RuleResult:
    """R3: a brand name → the ingredients every one of its branded products has.

    A brand reaches its products through Athena's ``Brand name of``; each
    product's ingredients are its ``has_ingredient`` edges. Only when every
    product has the same ingredient set is that set the brand's — a brand
    reused across formulations is left for a later step. Athena's
    ``Tradename of`` and the brand's precise ingredients, when present, must
    agree with the set.
    """
    result = RuleResult(
        "R3", "Brand name (BN) → the ingredients all its products share", overlay="internal"
    )
    con = context.con
    con.execute(
        "CREATE TABLE r3_subjects AS SELECT vocabulary_id AS v, concept_code AS c, "
        "concept_name AS n FROM base_concepts WHERE list_contains(?::VARCHAR[], vocabulary_id) "
        "AND list_contains(?::VARCHAR[], concept_class)",
        [context.family, list(BRAND_CLASSES)],
    )
    con.execute(
        "CREATE TABLE r3_products AS SELECT v1 AS v, c1 AS c, v2 AS pv, c2 AS pc FROM ath_rel "
        "WHERE relationship_id = 'Brand name of' AND k1 = 'Brand Name' "
        "AND list_contains(?::VARCHAR[], k2) UNION SELECT v2, c2, v1, c1 FROM ath_rel "
        "WHERE relationship_id = 'Has brand name' AND k2 = 'Brand Name' "
        "AND list_contains(?::VARCHAR[], k1)",
        [list(BRANDED_PRODUCT_CLASSES), list(BRANDED_PRODUCT_CLASSES)],
    )
    con.execute(
        "CREATE TABLE r3_sets AS SELECT p.v, p.c, p.pv, p.pc, "
        "array_to_string(list_sort(list(DISTINCT r.vocabulary_id_2 || '|' || "
        "r.concept_code_2)), ';') AS sig FROM r3_products p JOIN base_relationships r "
        "ON r.vocabulary_id_1 = p.pv AND r.concept_code_1 = p.pc AND r.predicate = "
        "'has_ingredient' JOIN family_ingredients i ON i.vocabulary_id = r.vocabulary_id_2 "
        "AND i.concept_code = r.concept_code_2 GROUP BY ALL"
    )
    con.execute(
        "CREATE TABLE r3_tradename AS SELECT x.v, x.c, array_to_string(list_sort(list(DISTINCT "
        "x.iv || '|' || x.ic)), ';') AS sig FROM (SELECT v1 AS v, c1 AS c, v2 AS iv, c2 AS ic "
        "FROM ath_rel WHERE relationship_id = 'Tradename of' AND k1 = 'Brand Name' "
        "AND k2 = 'Ingredient' UNION SELECT v2, c2, v1, c1 FROM ath_rel "
        "WHERE relationship_id = 'Has tradename' AND k1 = 'Ingredient' AND k2 = 'Brand Name') x "
        "JOIN family_ingredients i ON i.vocabulary_id = x.iv AND i.concept_code = x.ic "
        "GROUP BY ALL"
    )
    con.execute(
        "CREATE TABLE r3_precise AS SELECT r.vocabulary_id_1 AS v, r.concept_code_1 AS c, "
        "list_distinct(list(COALESCE(p.iv || '|' || p.ic, i.vocabulary_id || '|' || "
        "i.concept_code))) AS ins FROM base_relationships r "
        "LEFT JOIN pin_ingredient p ON p.v = r.vocabulary_id_2 AND p.c = r.concept_code_2 "
        "LEFT JOIN family_ingredients i ON i.vocabulary_id = r.vocabulary_id_2 "
        "AND i.concept_code = r.concept_code_2 "
        "WHERE r.predicate = 'has_precise_ingredient' AND (p.v IS NOT NULL OR i.vocabulary_id "
        "IS NOT NULL) GROUP BY ALL"
    )
    con.execute(
        "CREATE TABLE r3_out AS WITH agg AS (SELECT v, c, count(DISTINCT sig) AS sets, "
        "min(sig) AS sig, count(*) AS products FROM r3_sets GROUP BY ALL) "
        f"SELECT s.v, s.c, {_subject('s.v', 's.c', 's.n')} AS subject, a.sig, a.products, CASE "
        "WHEN a.sig IS NULL THEN 'no_products_with_ingredients' "
        "WHEN a.sets > 1 THEN 'products_disagree' "
        "WHEN t.sig IS NOT NULL AND t.sig <> a.sig THEN 'tradename_disagrees' "
        "WHEN q.ins IS NOT NULL AND NOT list_has_all(string_split(a.sig, ';'), q.ins) "
        "THEN 'precise_ingredient_disagrees' ELSE 'linked' END AS outcome, "
        "'OMOP brand products (' || COALESCE(a.products, 0) || ' agree)' || "
        "CASE WHEN t.sig IS NOT NULL THEN '; OMOP Tradename of' ELSE '' END AS source "
        "FROM r3_subjects s LEFT JOIN agg a USING (v, c) LEFT JOIN r3_tradename t USING (v, c) "
        "LEFT JOIN r3_precise q USING (v, c)"
    )
    con.execute(
        "INSERT INTO emit_relationships SELECT v, c, 'has_ingredient', "
        "split_part(pick, '|', 1), split_part(pick, '|', 2), source, 'R3' FROM ("
        "SELECT v, c, source, unnest(string_split(sig, ';')) AS pick FROM r3_out "
        "WHERE outcome = 'linked')"
    )
    return context.record(
        result, outcomes="r3_out", emitted_sql=_RELATIONSHIP_EMITTED, emitted_params=["R3"]
    )


# -- external, ID-based: X1 to X4 ----------------------------------------------


def rule_unii_records(context: LinkContext, *, release: str) -> RuleResult:
    """X1: FDA UNII Records → RxNorm (RXCUI column) and NCIt (NCIT column).

    Keys go on both ends, so a matched UNII name and a matched RxNorm or
    NCIt name join directly, and any other concept carrying the UNII code —
    a DrugBank drug, a MeSH record — joins through it.
    """
    result = RuleResult("X1", "UNII ↔ RxNorm and NCIt (FDA UNII Records)", overlay="external_id")
    unii = context.vocabulary("UNII")
    if unii is None:
        return context.skip(result, "the dictionary has no UNII vocabulary")
    targets = [(name, context.vocabulary(name)) for name in (context.target, "NCIt")]
    con = context.con
    con.execute("CREATE TABLE x1_pairs (unii VARCHAR, column_name VARCHAR, code VARCHAR)")
    for column, vocabulary in targets:
        if vocabulary is None:
            continue
        source_column = "rxcui" if column == context.target else "ncit"
        con.execute(
            f"INSERT INTO x1_pairs SELECT DISTINCT unii, ?, trim(code) FROM (SELECT unii, "
            f"unnest(regexp_split_to_array({source_column}, '[|;, ]+')) AS code "
            f"FROM unii_records WHERE {source_column} IS NOT NULL) WHERE trim(code) <> ''",
            [vocabulary],
        )
    con.execute(
        "CREATE TABLE x1_out AS SELECT p.unii, p.column_name AS vocabulary, p.code, "
        "CASE WHEN t.status = 'active' THEN t.concept_code "
        "WHEN r.status = 'active' THEN r.concept_code END AS resolved, "
        "'UNII:' || p.unii || ' -> ' || p.column_name || ':' || p.code AS subject, CASE "
        "WHEN u.concept_code IS NULL THEN 'unii_not_in_dictionary' "
        "WHEN t.concept_code IS NULL THEN 'code_not_in_dictionary' "
        "WHEN t.status <> 'active' AND (r.concept_code IS NULL OR r.status <> 'active') "
        "THEN 'code_retired_without_replacement' ELSE 'linked' END AS outcome "
        "FROM x1_pairs p LEFT JOIN base_concepts u ON u.vocabulary_id = ? "
        "AND u.concept_code = p.unii LEFT JOIN base_concepts t "
        "ON t.vocabulary_id = p.column_name AND t.concept_code = p.code "
        "LEFT JOIN base_concepts r ON r.vocabulary_id = t.vocabulary_id "
        "AND r.concept_code = t.replaced_by",
        [unii],
    )
    source = f"FDA UNII Records {release}"
    con.execute(
        "INSERT INTO emit_xrefs SELECT ?, unii, vocabulary, resolved, ?, 'X1', 'external_id' "
        "FROM x1_out WHERE outcome = 'linked' UNION ALL SELECT vocabulary, resolved, ?, unii, ?, "
        "'X1', 'external_id' FROM x1_out WHERE outcome = 'linked'",
        [unii, source, unii, source],
    )
    return context.record(
        result, outcomes="x1_out", emitted_sql=_XREF_EMITTED, emitted_params=["X1"]
    )


def rule_mesh_registry_numbers(context: LinkContext) -> RuleResult:
    """X2: a MeSH record's own registry number, when it is a UNII, → that UNII."""
    result = RuleResult("X2", "MeSH → UNII (MeSH registry numbers)", overlay="external_id")
    mesh, unii = context.vocabulary("MeSH"), context.vocabulary("UNII")
    if mesh is None or unii is None:
        return context.skip(result, "the dictionary lacks MeSH or UNII")
    context.con.execute(
        "CREATE TABLE x2_out AS SELECT m.mesh_ui, m.registry_number, "
        "'MeSH:' || m.mesh_ui || ' RN ' || m.registry_number AS subject, CASE "
        "WHEN NOT m.unii_shaped THEN 'not_a_unii (CAS or EC number)' "
        "WHEN c.concept_code IS NULL THEN 'mesh_record_not_in_dictionary' "
        "WHEN u.concept_code IS NULL THEN 'unii_not_in_dictionary' ELSE 'linked' END AS outcome "
        "FROM mesh_rn m LEFT JOIN base_concepts c ON c.vocabulary_id = ? "
        "AND c.concept_code = m.mesh_ui LEFT JOIN base_concepts u ON u.vocabulary_id = ? "
        "AND u.concept_code = m.registry_number",
        [mesh, unii],
    )
    context.con.execute(
        "INSERT INTO emit_xrefs SELECT ?, mesh_ui, ?, registry_number, 'MeSH registry number', "
        "'X2', 'external_id' FROM x2_out WHERE outcome = 'linked'",
        [mesh, unii],
    )
    return context.record(
        result, outcomes="x2_out", emitted_sql=_XREF_EMITTED, emitted_params=["X2"]
    )


def rule_omop_mappings(context: LinkContext) -> RuleResult:
    """X3: OHDSI mappings of other vocabularies into the target family.

    ``Maps to`` (SNOMED substances and products to their standard RxNorm
    concept), ``SNOMED - RxNorm eq``, and ``ATC - RxNorm`` (an ATC fifth level
    to its ingredient). The key names the family concept, so a matched SNOMED
    or ATC name joins the matched RxNorm name it maps to.
    """
    result = RuleResult(
        "X3", "SNOMED and ATC → RxNorm family (OHDSI mappings)", overlay="external_id"
    )
    context.con.execute(
        "CREATE TABLE x3_out AS SELECT r.v1, r.c1, r.relationship_id, r.v2, r.c2, "
        "r.v1 || ':' || r.c1 || ' --' || r.relationship_id || '--> ' || r.v2 || ':' || r.c2 "
        "AS subject, CASE WHEN t.status = 'active' THEN 'linked' ELSE 'target_inactive' END "
        "AS outcome FROM ath_rel r JOIN base_concepts t ON t.vocabulary_id = r.v2 "
        "AND t.concept_code = r.c2 WHERE list_contains(['Maps to', 'SNOMED - RxNorm eq', "
        "'ATC - RxNorm'], r.relationship_id) AND NOT list_contains(?::VARCHAR[], r.v1) "
        "AND list_contains(?::VARCHAR[], r.v2)",
        [context.family, context.family],
    )
    context.con.execute(
        "INSERT INTO emit_xrefs SELECT v1, c1, v2, c2, 'OMOP ' || relationship_id, 'X3', "
        "'external_id' FROM x3_out WHERE outcome = 'linked'"
    )
    return context.record(
        result, outcomes="x3_out", emitted_sql=_XREF_EMITTED, emitted_params=["X3"]
    )


def rule_umls_hub(context: LinkContext, *, release: str) -> RuleResult:
    """X4: give every code a UMLS concept lists that concept's CUI as a key.

    A MeSH record and an RxNorm concept in one CUI then join through the CUI
    even when the UMLS concept itself is not among the matches.
    """
    result = RuleResult("X4", "UMLS CUI as a shared key (UMLS hub)", overlay="external_id")
    umls = context.vocabulary("UMLS")
    if umls is None:
        return context.skip(result, "the dictionary has no UMLS vocabulary")
    others = [v for v in context.served.values() if v != umls]
    context.con.execute(
        "CREATE TABLE x4_out AS SELECT x.concept_code AS cui, x.key_type AS v, x.key_value AS c, "
        "'UMLS:' || x.concept_code || ' lists ' || x.key_type || ':' || x.key_value AS subject, "
        "CASE WHEN b.concept_code IS NULL THEN 'code_not_in_dictionary' ELSE 'linked' END "
        "AS outcome FROM base_xrefs x LEFT JOIN base_concepts b ON b.vocabulary_id = x.key_type "
        "AND b.concept_code = x.key_value WHERE x.vocabulary_id = ? "
        "AND list_contains(?::VARCHAR[], x.key_type)",
        [umls, others],
    )
    context.con.execute(
        "INSERT INTO emit_xrefs SELECT v, c, ?, cui, ?, 'X4', 'external_id' FROM x4_out "
        "WHERE outcome = 'linked'",
        [umls, f"UMLS {release} hub"],
    )
    return context.record(
        result, outcomes="x4_out", emitted_sql=_XREF_EMITTED, emitted_params=["X4"]
    )
