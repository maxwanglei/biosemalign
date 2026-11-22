"""Context-envelope, contextual-attribute, and purpose-scoped view (§5.2, WP3).

Declared fields (from the profile and its domain pack) route into named schema
slots or ``extensions``. Structural provenance keys route to their named
``ContextEnvelope`` slots. Everything undeclared lands in
``context.neighboring_fields`` — visible to reviewers, invisible to modules.

Every module reads context through :func:`context_view`, which exposes only
fields whose ``use`` includes that purpose (and never blinded fields during
resolution or selection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from biosemalign.schemas.sau import ContextEnvelope, ContextualAttributes, SemanticAlignmentUnit
from biosemalign.schemas.source import SourceRecord

if TYPE_CHECKING:
    from biosemalign.profiles.schema import ContextFieldConfig, ContextUse, TaskProfile

__all__ = [
    "build_attributes",
    "build_envelope",
    "context_view",
    "context_view_from_parts",
    "normalize_context_key",
]

#: Structural provenance keys recognised regardless of domain pack.
_STRUCTURAL_ENVELOPE_ALIASES: dict[str, str] = {
    "source_system": "source_system",
    "table_name": "table_name",
    "table": "table_name",
    "column_name": "column_name",
    "column": "column_name",
    "field_description": "field_description",
    "data_dictionary_definition": "data_dictionary_definition",
    "record_id": "record_id",
    "date": "date",
    "sentence": "sentence",
    "paragraph": "paragraph",
    "section_heading": "section_heading",
    "document_type": "document_type",
    "table_caption": "table_caption",
    "footnote": "footnote",
    "study_design": "study_design",
    "evidence_block_id": "evidence_block_id",
}

_NAMED_ENVELOPE_FIELDS: frozenset[str] = frozenset(ContextEnvelope.model_fields) - {
    "extensions",
    "neighboring_fields",
}
_NAMED_ATTRIBUTE_FIELDS: frozenset[str] = frozenset(ContextualAttributes.model_fields) - {
    "extensions",
}


def normalize_context_key(key: str) -> str:
    """Normalize a source/context field key for case- and space-insensitive matching."""
    return key.strip().casefold().replace(" ", "_")


_normalize_key = normalize_context_key


def _declared_alias_map(profile: TaskProfile | None) -> dict[str, ContextFieldConfig]:
    if profile is None:
        return {}
    aliases: dict[str, ContextFieldConfig] = {}
    for field in profile.effective_context_fields():
        aliases.setdefault(_normalize_key(field.name), field)
        for alias in field.aliases:
            aliases.setdefault(_normalize_key(alias), field)
    return aliases


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def build_envelope(record: SourceRecord, profile: TaskProfile | None = None) -> ContextEnvelope:
    """Assemble the context surrounding the mention from declared and provenance fields."""
    alias_map = _declared_alias_map(profile)
    known: dict[str, Any] = {}
    extensions: dict[str, Any] = {}
    leftover: dict[str, Any] = {}

    for raw_key, value in record.row.items():
        if _is_missing_value(value):
            continue
        norm = _normalize_key(raw_key)
        declared = alias_map.get(norm)
        if declared is not None:
            name = declared.name
            if declared.kind == "attribute":
                # Attribute fields live on ContextualAttributes; only mirror onto
                # shared ContextEnvelope slots when the schema carries that slot.
                if name in _NAMED_ENVELOPE_FIELDS:
                    if name not in known:
                        known[name] = str(value)
                    else:
                        leftover[raw_key] = value
                continue
            if name in _NAMED_ENVELOPE_FIELDS:
                if name not in known:
                    known[name] = str(value)
                else:
                    leftover[raw_key] = value
            elif name not in extensions:
                extensions[name] = value
            else:
                leftover[raw_key] = value
            continue

        structural = _STRUCTURAL_ENVELOPE_ALIASES.get(norm)
        if structural is not None:
            if structural not in known:
                known[structural] = str(value)
            else:
                leftover[raw_key] = value
            continue

        if profile is None and norm in _NAMED_ENVELOPE_FIELDS:
            if norm not in known:
                known[norm] = str(value)
            else:
                leftover[raw_key] = value
            continue

        if profile is None and norm in _NAMED_ATTRIBUTE_FIELDS:
            continue

        leftover[raw_key] = value

    # Explicit record fields win over anything inferred from the row.
    known["source_system"] = record.source_system or known.get("source_system")
    known["table_name"] = record.table_name or known.get("table_name")
    known["column_name"] = record.column_name or known.get("column_name")
    known["record_id"] = record.record_id or known.get("record_id")
    known = {k: v for k, v in known.items() if v is not None}

    return ContextEnvelope(neighboring_fields=leftover, extensions=extensions, **known)


def build_attributes(
    record: SourceRecord, profile: TaskProfile | None = None
) -> ContextualAttributes:
    """Assemble modifiers asserted about the mention."""
    alias_map = _declared_alias_map(profile)
    values: dict[str, Any] = {}
    extensions: dict[str, Any] = {}

    for raw_key, value in record.row.items():
        if _is_missing_value(value):
            continue
        norm = _normalize_key(raw_key)
        declared = alias_map.get(norm)
        if declared is not None:
            if declared.kind != "attribute":
                continue
            name = declared.name
            if name in _NAMED_ATTRIBUTE_FIELDS:
                if name not in values:
                    values[name] = value if name == "negation" else str(value)
            elif name not in extensions:
                extensions[name] = value
            continue

        if profile is None and norm in _NAMED_ATTRIBUTE_FIELDS and norm not in values:
            values[norm] = value if norm == "negation" else str(value)

    return ContextualAttributes(extensions=extensions, **values)


def _read_declared_field(
    envelope: ContextEnvelope,
    attributes: ContextualAttributes,
    field: ContextFieldConfig,
) -> Any:
    """Read a declared field's value from named slots or ``extensions`` (never neighboring_fields)."""
    name = field.name
    order = ("attributes", "context") if field.kind == "attribute" else ("context", "attributes")
    for source_kind in order:
        if source_kind == "attributes":
            if name in _NAMED_ATTRIBUTE_FIELDS:
                val = getattr(attributes, name, None)
                if val is not None and val != "":
                    return val
            elif name in attributes.extensions:
                val = attributes.extensions[name]
                if val is not None and val != "":
                    return val
        else:
            if name in _NAMED_ENVELOPE_FIELDS:
                val = getattr(envelope, name, None)
                if val is not None and val != "":
                    return val
            elif name in envelope.extensions:
                val = envelope.extensions[name]
                if val is not None and val != "":
                    return val
    return None


def context_view_from_parts(
    envelope: ContextEnvelope,
    attributes: ContextualAttributes,
    purpose: ContextUse,
    profile: TaskProfile,
) -> dict[str, Any]:
    """Return declared context/attribute values visible for ``purpose``."""
    fields = profile.effective_context_fields()
    if purpose == "key":
        key_names = set(profile.effective_mapping_key_fields())
        selected = [field for field in fields if field.name in key_names]
    else:
        selected = [
            field
            for field in fields
            if purpose in field.use and not (field.blinded and purpose in {"resolve", "select"})
        ]

    view: dict[str, Any] = {}
    for field in selected:
        value = _read_declared_field(envelope, attributes, field)
        if value is not None:
            view[field.name] = value
    return view


def context_view(
    sau: SemanticAlignmentUnit, purpose: ContextUse, profile: TaskProfile
) -> dict[str, Any]:
    """Return declared context/attribute fields on ``sau`` permitted for ``purpose``."""
    return context_view_from_parts(sau.context, sau.attributes, purpose, profile)
