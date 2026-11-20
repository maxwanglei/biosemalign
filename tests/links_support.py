"""An invented dictionary and raw sources that exercise every link rule.

Every code and name here is made up. The dictionary mimics the shape of a
real multi-vocabulary drug dictionary — RxNorm and RxNorm Extension with
OMOP keys, UNII, MeSH, NCIt, SNOMED, ATC, UMLS, and an FDA product list —
with the links a real one lacks left out, so each rule has something to find
and something to decline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

RELEASES = {
    "RxNorm": "RxNorm synthetic-1",
    "RxNorm Extension": "RxNorm Extension synthetic-1",
    "SNOMED": "SNOMED synthetic-1",
    "ATC": "ATC synthetic-1",
    "UNII": "synthetic-unii",
    "MeSH": "synthetic-mesh",
    "NCIt": "synthetic-ncit",
    "UMLS": "synthetic-umls",
    "Drugs@FDA": "synthetic-fda",
}

#: (vocabulary, code, name, class, OMOP concept id or None)
CONCEPTS: list[tuple[str, str, str, str, int | None]] = [
    ("RxNorm", "10", "alphamab", "IN", 110),
    ("RxNorm", "11", "betaprofen", "IN", 111),
    ("RxNorm", "12", "gammacillin", "IN", 112),
    ("RxNorm", "20", "alphamab sodium", "PIN", 120),
    ("RxNorm", "21", "betaprofen lysine", "PIN", 121),
    ("RxNorm", "22", "gammacillin potassium", "PIN", 122),
    ("RxNorm", "30", "alphamab / betaprofen", "MIN", 130),
    ("RxNorm", "31", "alphamab / unknownium", "MIN", 131),
    ("RxNorm", "40", "Alphabrand", "BN", 140),
    ("RxNorm", "41", "Mixbrand", "BN", 141),
    ("RxNorm", "42", "Saltbrand", "BN", 142),
    ("RxNorm", "43", "Tradebrand", "BN", 143),
    ("RxNorm", "44", "Lonebrand", "BN", 144),
    ("RxNorm", "50", "alphamab 10 MG Oral Tablet [Alphabrand]", "SBD", 150),
    ("RxNorm", "51", "alphamab Oral Tablet [Alphabrand]", "SBDF", 151),
    ("RxNorm", "52", "alphamab 5 MG Oral Tablet [Mixbrand]", "SBD", 152),
    ("RxNorm", "53", "alphamab / betaprofen Oral Tablet [Mixbrand]", "SBD", 153),
    ("RxNorm", "54", "alphamab / gammacillin Tablet [Saltbrand]", "SBD", 154),
    ("RxNorm", "55", "betaprofen Tablet [Tradebrand]", "SBD", 155),
    ("RxNorm Extension", "OMOP900", "Extbrand", "BN", 900),
    ("RxNorm Extension", "OMOP901", "alphamab Tablet [Extbrand]", "SBD", 901),
    ("RxNorm Extension", "OMOP902", "deltaxin", "IN", 902),
    ("RxNorm Extension", "OMOP903", "Deltabrand", "BN", 903),
    ("RxNorm Extension", "OMOP904", "deltaxin Tablet [Deltabrand]", "SBD", 904),
    ("UNII", "UNIIALPHA1", "ALPHAMAB", "substance", None),
    ("UNII", "UNIIBETAL1", "BETAPROFEN LYSINE", "substance", None),
    ("UNII", "UNIIPARA01", "PARALPHA", "substance", None),
    ("MeSH", "D000001", "Alphamab", "descriptor", None),
    ("MeSH", "D000002", "Somethingate", "descriptor", None),
    ("MeSH", "C000005", "Betaprofen lysine", "supplementary", None),
    ("NCIt", "C1", "Alphamab", "Pharmacologic Substance", None),
    ("SNOMED", "900001", "Alphamab", "Substance", 9001),
    ("ATC", "Z99ZZ01", "alphamab", "ATC 5th", 9101),
    ("UMLS", "CUI0001", "Alphamab", "cui", None),
    ("UMLS", "CUI0002", "Orphan", "cui", None),
    ("Drugs@FDA", "000001-001", "ALPHABRAND", "FDA drug product", None),
    ("Drugs@FDA", "000002-001", "ALPHAMAB SODIUM", "FDA drug product", None),
    ("Drugs@FDA", "000003-001", "MIXBRAND", "FDA drug product", None),
    ("Drugs@FDA", "000004-001", "OTHERBRAND", "FDA drug product", None),
    ("Drugs@FDA", "000005-001", "ALPHABRAND", "FDA drug product", None),
    ("Drugs@FDA", "000006-001", "ALPHAMAB", "FDA drug product", None),
]

#: Retired concepts: (vocabulary, code, name, class, status, replaced_by).
RETIRED = [
    ("RxNorm", "13", "alphamab (old)", "IN", "remapped", "10"),
    ("RxNorm", "14", "obsoletine", "IN", "obsolete", ""),
    ("UNII", "UNIIRETIRE", "ALPHAMAB OLD", "substance", "active", ""),
    ("UNII", "UNIIOBSOL1", "OBSOLETINE", "substance", "active", ""),
]

#: Product ingredient edges the dictionary already has (from CONCEPT_ANCESTOR).
PRODUCT_INGREDIENTS = [
    ("RxNorm", "50", "RxNorm", "10"),
    ("RxNorm", "51", "RxNorm", "10"),
    ("RxNorm", "52", "RxNorm", "10"),
    ("RxNorm", "53", "RxNorm", "10"),
    ("RxNorm", "53", "RxNorm", "11"),
    ("RxNorm", "54", "RxNorm", "10"),
    ("RxNorm", "54", "RxNorm", "12"),
    ("RxNorm", "55", "RxNorm", "11"),
    ("RxNorm Extension", "OMOP901", "RxNorm", "10"),
    ("RxNorm Extension", "OMOP904", "RxNorm Extension", "OMOP902"),
]

PRODUCT_EVIDENCE = [
    ("Drugs@FDA", "000001-001", "ALPHABRAND", "ALPHAMAB"),
    ("Drugs@FDA", "000002-001", "ALPHAMAB SODIUM", "ALPHAMAB SODIUM"),
    ("Drugs@FDA", "000003-001", "MIXBRAND", "ALPHAMAB"),
    ("Drugs@FDA", "000004-001", "OTHERBRAND", "UNKNOWNIUM"),
    # The same brand, its ingredient named through a UNII whose record has an RXCUI.
    ("Drugs@FDA", "000005-001", "ALPHABRAND", "PARALPHA"),
    # A generic named like both the RxNorm ingredient and a UNII substance.
    ("Drugs@FDA", "000006-001", "ALPHAMAB", "ALPHAMAB"),
]

UNII_RECORDS = [
    ("UNIIALPHA1", "10", "C1"),
    ("UNIIBETAL1", "21", ""),
    ("UNIIPARA01", "10", ""),
    ("UNIIABSENT", "10", ""),
    # An RXCUI since retired: followed to its replacement, or declined without one.
    ("UNIIRETIRE", "13", ""),
    ("UNIIOBSOL1", "14", ""),
]

MESH_DESCRIPTORS = """<?xml version="1.0"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000001</DescriptorUI>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <RegistryNumberList><RegistryNumber>UNIIALPHA1</RegistryNumber></RegistryNumberList>
      </Concept>
      <Concept PreferredConceptYN="N">
        <RegistryNumberList><RegistryNumber>UNIIBETAL1</RegistryNumber></RegistryNumberList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord>
    <DescriptorUI>D000002</DescriptorUI>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <RegistryNumberList><RegistryNumber>50-78-2</RegistryNumber></RegistryNumberList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""

MESH_SUPPLEMENTARY = """<?xml version="1.0"?>
<SupplementalRecordSet>
  <SupplementalRecord>
    <SupplementalRecordUI>C000005</SupplementalRecordUI>
    <ConceptList>
      <Concept PreferredConceptYN="Y"><RegistryNumber>UNIIBETAL1</RegistryNumber></Concept>
      <!-- older files: one RegistryNumber directly under the concept -->
    </ConceptList>
  </SupplementalRecord>
</SupplementalRecordSet>
"""

#: Athena relationships: (relationship_id, concept_id_1, concept_id_2)
ATHENA_RELATIONSHIPS = [
    ("Has form", 111, 121),  # betaprofen has form betaprofen lysine (reverse direction)
    ("Form of", 122, 112),  # gammacillin potassium is a form of gammacillin
    ("Brand name of", 140, 150),
    ("Brand name of", 140, 151),
    ("Has brand name", 152, 141),
    ("Has brand name", 153, 141),
    ("Brand name of", 142, 154),
    ("Brand name of", 143, 155),
    ("Tradename of", 143, 112),  # disagrees with Tradebrand's product
    ("Brand name of", 900, 901),
    ("Brand name of", 903, 904),
    ("Maps to", 9001, 110),
    ("ATC - RxNorm", 9101, 110),
]

_ATHENA_CLASS = {
    "IN": "Ingredient",
    "PIN": "Precise Ingredient",
    "MIN": "Multiple Ingredients",
    "BN": "Brand Name",
    "SBD": "Branded Drug",
    "SBDF": "Branded Drug Form",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dictionary(directory: Path) -> Path:
    """The dictionary's contract tables, plus its declaration and product list."""
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        directory / "concepts.csv",
        [
            {
                "vocabulary_id": v,
                "concept_code": c,
                "concept_name": n,
                "concept_class": k,
                "status": "active",
                "replaced_by": "",
            }
            for v, c, n, k, _ in CONCEPTS
        ]
        + [
            {
                "vocabulary_id": v,
                "concept_code": c,
                "concept_name": n,
                "concept_class": k,
                "status": status,
                "replaced_by": replaced_by,
            }
            for v, c, n, k, status, replaced_by in RETIRED
        ],
    )
    write_csv(
        directory / "synonyms.csv",
        [{"vocabulary_id": "UNII", "concept_code": "UNIIPARA01", "synonym": "Paralpha"}],
    )
    relationships = [
        {
            "vocabulary_id_1": v1,
            "concept_code_1": c1,
            "predicate": "has_ingredient",
            "vocabulary_id_2": v2,
            "concept_code_2": c2,
        }
        for v1, c1, v2, c2 in PRODUCT_INGREDIENTS
    ]
    relationships.append(
        {
            "vocabulary_id_1": "RxNorm",
            "concept_code_1": "42",
            "predicate": "has_precise_ingredient",
            "vocabulary_id_2": "RxNorm",
            "concept_code_2": "20",
        }
    )
    write_csv(directory / "relationships.csv", relationships)
    xrefs: list[dict[str, Any]] = []
    for v, c, _, _, omop in CONCEPTS:
        if omop is not None:
            xrefs.append(_xref(v, c, "OMOP_CONCEPT_ID", str(omop), "OMOP"))
    # alphamab sodium maps to alphamab in OMOP.
    xrefs.append(_xref("RxNorm", "20", "OMOP_STANDARD", "110", "OMOP"))
    xrefs.append(_xref("UMLS", "CUI0001", "MeSH", "D000001", "UMLS"))
    xrefs.append(_xref("UMLS", "CUI0001", "RxNorm", "10", "UMLS"))
    xrefs.append(_xref("UMLS", "CUI0002", "NCIt", "C999", "UMLS"))
    write_csv(directory / "xrefs.csv", xrefs)
    write_csv(
        directory / "product_evidence.csv",
        [
            {"vocabulary_id": v, "concept_code": c, "product_name": n, "ingredient_text": i}
            for v, c, n, i in PRODUCT_EVIDENCE
        ],
    )
    (directory / "vocabularies.yaml").write_text(
        yaml.safe_dump(
            {
                "vocabularies": [
                    {"vocabulary_id": v, "release": r, "license": "synthetic"}
                    for v, r in RELEASES.items()
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return directory


def _xref(v: str, c: str, key_type: str, key_value: str, source: str) -> dict[str, Any]:
    return {
        "vocabulary_id": v,
        "concept_code": c,
        "key_type": key_type,
        "key_value": key_value,
        "assertion_source": source,
    }


def write_athena(directory: Path, *, rxnorm_release: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    vocabularies = [
        ["vocabulary_id", "vocabulary_name", "vocabulary_reference", "vocabulary_version"]
    ] + [
        [v, v, "x", rxnorm_release if (v == "RxNorm" and rxnorm_release) else r]
        for v, r in RELEASES.items()
        if v in {"RxNorm", "RxNorm Extension", "SNOMED", "ATC"}
    ]
    concepts = [
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ]
    ] + [
        [str(omop), n, "Drug", v, _ATHENA_CLASS.get(k, k), "S", c, "19700101", "20991231", ""]
        for v, c, n, k, omop in CONCEPTS
        if omop is not None
    ]
    relationships = [
        [
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ]
    ] + [[str(a), str(b), rel, "19700101", "20991231", ""] for rel, a, b in ATHENA_RELATIONSHIPS]
    for name, rows in (
        ("VOCABULARY", vocabularies),
        ("CONCEPT", concepts),
        ("CONCEPT_RELATIONSHIP", relationships),
    ):
        (directory / f"{name}.csv").write_text(
            "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
        )
    return directory


def write_unii_zip(path: Path) -> Path:
    import zipfile

    header = "UNII\tDisplay Name\tRN\tNCIT\tRXCUI\tINCHIKEY"
    lines = [header] + [f"{unii}\tX\t\t{ncit}\t{rxcui}\t" for unii, rxcui, ncit in UNII_RECORDS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("UNII_Records_01Jan2026.txt", "\n".join(lines) + "\n")
        archive.writestr("UNII_Names_01Jan2026.txt", "Name\tTYPE\tUNII\n")
    return path


def write_mesh(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    desc = directory / "desc.xml"
    supp = directory / "supp.xml"
    desc.write_text(MESH_DESCRIPTORS, encoding="utf-8")
    supp.write_text(MESH_SUPPLEMENTARY, encoding="utf-8")
    return [desc, supp]
