"""Readers for the raw link sources: Athena, FDA UNII Records, and MeSH XML."""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from xml.etree import ElementTree

from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.lexicon.links.common import LinkContext

__all__ = [
    "ATHENA_RELATIONSHIPS",
    "load_athena",
    "load_mesh_registry_numbers",
    "load_unii_records",
]

#: OHDSI relationships the link rules read. Both directions are listed where
#: Athena may hold either.
ATHENA_RELATIONSHIPS = (
    "Form of",
    "Has form",
    "Brand name of",
    "Has brand name",
    "Tradename of",
    "Has tradename",
    "RxNorm has ing",
    "RxNorm ing of",
    "Maps to",
    "SNOMED - RxNorm eq",
    "ATC - RxNorm",
)

_UNII_SHAPE = re.compile(r"^[A-Z0-9]{10}$")


def _sql(path: Path) -> str:
    return str(path).replace("'", "''")


def _athena_csv(path: Path) -> str:
    # Athena files are tab-delimited and unquoted; names may contain quotes.
    return (
        f"read_csv('{_sql(path)}', delim='\\t', header=true, quote='', escape='', all_varchar=true)"
    )


def load_athena(
    context: LinkContext,
    athena_dir: Path,
    *,
    vocabularies: Sequence[str],
    releases: dict[str, str],
    allow_version_mismatch: bool = False,
) -> None:
    """Load the Athena concepts and relationships the rules need.

    Creates ``ath_concept`` (concept_id → vocabulary, code, class) restricted
    to ``vocabularies`` and present in the base, ``ath_rel`` (valid edges of
    :data:`ATHENA_RELATIONSHIPS` touching the target family), and records a
    profile of every RxNorm-family relationship pattern in the context notes.
    """
    con = context.con
    for name in ("CONCEPT", "CONCEPT_RELATIONSHIP", "VOCABULARY"):
        if not (athena_dir / f"{name}.csv").is_file():
            raise LexiconError(f"{name}.csv is missing from {athena_dir}")

    versions = {
        str(vocabulary): str(version or "")
        for vocabulary, version in context.rows(
            "SELECT vocabulary_id, vocabulary_version FROM "
            + _athena_csv(athena_dir / "VOCABULARY.csv")
        )
    }
    mismatches = [
        f"{vocabulary}: dictionary {releases[vocabulary]!r}, Athena {versions.get(vocabulary)!r}"
        for vocabulary in vocabularies
        if vocabulary in releases and versions.get(vocabulary) != releases[vocabulary]
    ]
    if mismatches:
        message = "Athena release differs from the dictionary: " + "; ".join(mismatches)
        if not allow_version_mismatch:
            raise LexiconError(
                message + ". Codes may have changed; use the Athena download the dictionary "
                "was built from, or allow the mismatch explicitly."
            )
        context.notes.append("WARNING: " + message)
    else:
        context.notes.append(
            "Athena releases match the dictionary for " + ", ".join(sorted(vocabularies))
        )

    con.execute(
        "CREATE TABLE ath_concept AS SELECT a.concept_id, a.vocabulary_id, a.concept_code, "
        "a.concept_class_id FROM "
        + _athena_csv(athena_dir / "CONCEPT.csv")
        + " a JOIN base_concepts b ON b.vocabulary_id = a.vocabulary_id "
        "AND b.concept_code = a.concept_code WHERE list_contains(?::VARCHAR[], a.vocabulary_id)",
        [list(vocabularies)],
    )
    con.execute("CREATE UNIQUE INDEX ath_concept_id ON ath_concept (concept_id)")
    relationships = _athena_csv(athena_dir / "CONCEPT_RELATIONSHIP.csv")
    family = list(context.family)
    con.execute(
        "CREATE TABLE ath_rel AS SELECT r.relationship_id, a.vocabulary_id AS v1, "
        "a.concept_code AS c1, a.concept_class_id AS k1, b.vocabulary_id AS v2, "
        "b.concept_code AS c2, b.concept_class_id AS k2 FROM "
        + relationships
        + " r JOIN ath_concept a ON a.concept_id = r.concept_id_1 "
        "JOIN ath_concept b ON b.concept_id = r.concept_id_2 "
        "WHERE list_contains(?::VARCHAR[], r.relationship_id) "
        "AND trim(COALESCE(r.invalid_reason, '')) = '' AND r.concept_id_1 <> r.concept_id_2 "
        "AND (list_contains(?::VARCHAR[], a.vocabulary_id) "
        "OR list_contains(?::VARCHAR[], b.vocabulary_id))",
        [list(ATHENA_RELATIONSHIPS), family, family],
    )
    profile = context.rows(
        "SELECT a.concept_class_id, r.relationship_id, b.concept_class_id, count(*) AS n FROM "
        + relationships
        + " r JOIN ath_concept a ON a.concept_id = r.concept_id_1 "
        "JOIN ath_concept b ON b.concept_id = r.concept_id_2 "
        "WHERE trim(COALESCE(r.invalid_reason, '')) = '' "
        "AND list_contains(?::VARCHAR[], a.vocabulary_id) "
        "AND list_contains(?::VARCHAR[], b.vocabulary_id) "
        "AND list_contains(['Ingredient', 'Precise Ingredient', 'Multiple Ingredients', "
        "'Brand Name'], a.concept_class_id) GROUP BY ALL ORDER BY n DESC",
        [family, family],
    )
    context.notes.append(
        "Athena relationship profile (subject class, relationship, object class: count), "
        "for ingredient and brand-name subjects in the target family:"
    )
    context.notes.extend(f"  {k1} --{rel}--> {k2}: {n:,}" for k1, rel, k2, n in profile)


def load_unii_records(context: LinkContext, path: Path, work: Path) -> int:
    """Load FDA's UNII Records file (or the zip holding it) as ``unii_records``.

    Keeps UNII, RXCUI, and NCIT; fails, naming the header found, when any of
    them is missing. Returns the number of records.
    """
    records = path
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if "unii_records" in Path(name).name.lower()
                and Path(name).suffix.lower() in {".txt", ".tsv"}
            ]
            if len(members) != 1:
                raise LexiconError(
                    f"{path.name} should hold one UNII_Records text file; found {members or 'none'}"
                )
            work.mkdir(parents=True, exist_ok=True)
            records = work / Path(members[0]).name
            with archive.open(members[0]) as source, records.open("wb") as target:
                while chunk := source.read(1 << 20):
                    target.write(chunk)
    relation = (
        f"read_csv('{_sql(records)}', delim='\\t', header=true, quote='', escape='', "
        "all_varchar=true)"
    )
    header = {str(row[0]) for row in context.rows(f"DESCRIBE SELECT * FROM {relation}")}
    missing = [column for column in ("UNII", "RXCUI", "NCIT") if column not in header]
    if missing:
        raise LexiconError(
            f"UNII Records file {records.name} lacks columns {missing}; header: {sorted(header)}"
        )
    context.con.execute(
        f'CREATE TABLE unii_records AS SELECT trim("UNII") AS unii, '
        f"NULLIF(trim(\"RXCUI\"), '') AS rxcui, NULLIF(trim(\"NCIT\"), '') AS ncit "
        f"FROM {relation} WHERE NULLIF(trim(\"UNII\"), '') IS NOT NULL"
    )
    return context.scalar("SELECT count(*) FROM unii_records")


def load_mesh_registry_numbers(context: LinkContext, paths: Sequence[Path]) -> int:
    """Load each MeSH record's preferred-concept registry numbers as ``mesh_rn``.

    Reads descriptor (``desc*.xml``) and supplementary (``supp*.xml``) files,
    streaming. Only the preferred concept's numbers describe the record
    itself; a non-preferred concept may be a related salt or form.
    """
    import pyarrow as pa

    uis: list[str] = []
    numbers: list[str] = []
    shaped: list[bool] = []
    for path in paths:
        for ui, number in _registry_numbers(path):
            uis.append(ui)
            numbers.append(number)
            shaped.append(bool(_UNII_SHAPE.match(number)))
    context.con.register(
        "mesh_batch",
        pa.table(
            {
                "mesh_ui": pa.array(uis, pa.string()),
                "registry_number": pa.array(numbers, pa.string()),
                "unii_shaped": pa.array(shaped, pa.bool_()),
            }
        ),
    )
    context.con.execute("CREATE TABLE mesh_rn AS SELECT * FROM mesh_batch")
    context.con.unregister("mesh_batch")
    return len(uis)


def _registry_numbers(path: Path) -> Iterator[tuple[str, str]]:
    records = {"DescriptorRecord": "DescriptorUI", "SupplementalRecord": "SupplementalRecordUI"}
    for _, element in ElementTree.iterparse(path, events=("end",)):
        tag = element.tag
        if tag not in records:
            continue
        ui = (element.findtext(records[tag]) or "").strip()
        for concept in element.iterfind("ConceptList/Concept"):
            if concept.get("PreferredConceptYN") != "Y":
                continue
            # MeSH nests numbers in RegistryNumberList; older files had one
            # RegistryNumber directly under the concept.
            numbers = [
                *concept.iterfind("RegistryNumberList/RegistryNumber"),
                *concept.iterfind("RegistryNumber"),
            ]
            for number in (n.text.strip() for n in numbers if n.text):
                if ui and number and number != "0":
                    yield ui, number
        element.clear()
