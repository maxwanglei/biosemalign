"""Context-envelope and contextual-attribute construction (§5.4, §5.6).

Known context keys are lifted into typed envelope fields; everything else is
kept verbatim under ``neighboring_fields``. Nothing supplied by the caller is
discarded — context that looks irrelevant to drug normalization may be exactly
what a reviewer needs to adjudicate an ambiguous product name.
"""

from __future__ import annotations

from typing import Any

from biosemalign.schemas.sau import ContextEnvelope, ContextualAttributes
from biosemalign.schemas.source import SourceRecord

__all__ = ["build_attributes", "build_envelope"]

#: Context keys recognised as envelope fields, with the aliases real source
#: systems actually use. FAERS, for instance, names the drug role ``role_cod``.
_ENVELOPE_ALIASES: dict[str, str] = {
    "source_system": "source_system",
    "table_name": "table_name",
    "table": "table_name",
    "column_name": "column_name",
    "column": "column_name",
    "field_description": "field_description",
    "data_dictionary_definition": "data_dictionary_definition",
    "record_id": "record_id",
    "primaryid": "record_id",
    "dose": "dose",
    "dose_amt": "dose",
    "unit": "unit",
    "dose_unit": "unit",
    "route": "route",
    "route_cod": "route",
    "drug_role": "drug_role",
    "role_cod": "drug_role",
    "indication": "indication",
    "indi_pt": "indication",
    "date": "date",
    "species": "species",
    "sentence": "sentence",
    "paragraph": "paragraph",
    "section_heading": "section_heading",
    "document_type": "document_type",
    "table_caption": "table_caption",
    "footnote": "footnote",
    "study_design": "study_design",
    "evidence_block_id": "evidence_block_id",
}

#: Context keys that describe the mention itself rather than its surroundings.
_ATTRIBUTE_ALIASES: dict[str, str] = {
    "negation": "negation",
    "certainty": "certainty",
    "assertion_status": "assertion_status",
    "temporality": "temporality",
    "severity": "severity",
    "species": "species",
    "anatomical_location": "anatomical_location",
    "dose": "dose",
    "dose_amt": "dose",
    "strength": "strength",
    "route": "route",
    "route_cod": "route",
    "formulation": "formulation",
    "dose_form": "formulation",
    "dosage_form": "formulation",
    "specimen": "specimen",
    "assay_method": "assay_method",
    "unit": "unit",
    "dose_unit": "unit",
    "experimental_system": "experimental_system",
}


def _normalize_key(key: str) -> str:
    return key.strip().casefold().replace(" ", "_")


def build_envelope(record: SourceRecord) -> ContextEnvelope:
    """Assemble the context surrounding the mention."""
    known: dict[str, Any] = {}
    leftover: dict[str, Any] = {}

    for raw_key, value in record.row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        field = _ENVELOPE_ALIASES.get(_normalize_key(raw_key))
        if field is not None and field not in known:
            known[field] = value if field in {"neighboring_fields"} else str(value)
        else:
            leftover[raw_key] = value

    # Explicit record fields win over anything inferred from the row.
    known["source_system"] = record.source_system or known.get("source_system")
    known["table_name"] = record.table_name or known.get("table_name")
    known["column_name"] = record.column_name or known.get("column_name")
    known["record_id"] = record.record_id or known.get("record_id")
    known = {k: v for k, v in known.items() if v is not None}

    return ContextEnvelope(neighboring_fields=leftover, **known)


def build_attributes(record: SourceRecord) -> ContextualAttributes:
    """Assemble modifiers asserted about the mention.

    Deliberately deterministic: v0.1 lifts attributes that the source system
    already supplies as structured fields. Inferring negation or certainty from
    free text is Module 1 work deferred to v0.2 along with mention detection.
    """
    values: dict[str, Any] = {}
    for raw_key, value in record.row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        field = _ATTRIBUTE_ALIASES.get(_normalize_key(raw_key))
        if field is not None and field not in values:
            values[field] = value if field == "negation" else str(value)
    return ContextualAttributes(**values)
