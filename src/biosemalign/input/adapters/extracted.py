"""Source adapter for terms already extracted from free text (``extracted_terms``)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from biosemalign.enums import MentionFidelity, SemanticType, SourceType
from biosemalign.exceptions import UnsupportedInput
from biosemalign.input.adapters.base import ADAPTER_VERSION
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.source import SourceRecord
from biosemalign.util import normalize_text

__all__ = ["ExtractedTermsAdapter", "compute_mention_fidelity"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _nfkc_normalized(text: str) -> str:
    return normalize_text(unicodedata.normalize("NFKC", text))


def compute_mention_fidelity(
    mention_text: str,
    source_text: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> MentionFidelity:
    """Classify whether ``mention_text`` occurs verbatim or normalized in ``source_text``."""
    if not mention_text:
        return MentionFidelity.NOT_FOUND
    if start is not None and end is not None and 0 <= start <= end <= len(source_text):
        span = source_text[start:end]
        if span == mention_text:
            return MentionFidelity.VERBATIM
        if _nfkc_normalized(span) == _nfkc_normalized(mention_text):
            return MentionFidelity.NORMALIZED_MATCH
    if mention_text in source_text:
        return MentionFidelity.VERBATIM
    if _nfkc_normalized(mention_text) in _nfkc_normalized(source_text):
        return MentionFidelity.NORMALIZED_MATCH
    return MentionFidelity.NOT_FOUND


def _containing_sentence(
    source_text: str,
    mention_text: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> str:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(source_text) if s.strip()]
    if not sentences:
        return source_text
    if start is not None and end is not None and 0 <= start <= end <= len(source_text):
        cursor = 0
        for sentence in sentences:
            pos = source_text.find(sentence, cursor)
            if pos != -1:
                cursor = pos + len(sentence)
                if pos <= start and end <= cursor:
                    return sentence
    for sentence in sentences:
        if mention_text in sentence:
            return sentence
    folded = _nfkc_normalized(mention_text)
    for sentence in sentences:
        if folded in _nfkc_normalized(sentence):
            return sentence
    return source_text


def _coerce_semantic_types(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, Sequence) else None
    if items is None:
        raise UnsupportedInput(
            f"term semantic type must be a string or list, got {type(raw).__name__}"
        )
    resolved: list[str] = []
    for item in items:
        try:
            resolved.append(SemanticType(str(item)).value)
        except ValueError as exc:
            raise UnsupportedInput(f"unknown semantic type {item!r} on extracted term") from exc
    return resolved


class ExtractedTermsAdapter:
    """Translate document text + pre-extracted term spans into one SourceRecord per term."""

    adapter_name = "extracted_terms"
    adapter_version = ADAPTER_VERSION

    def to_source_records(
        self, request: AlignmentRequest, profile: TaskProfile
    ) -> list[SourceRecord]:
        del profile
        if not isinstance(request.value, Mapping):
            raise UnsupportedInput(
                f"{self.adapter_name} expects a mapping with 'text' and 'terms', "
                f"got {type(request.value).__name__}"
            )
        source_text = request.value.get("text")
        if not isinstance(source_text, str) or not source_text.strip():
            raise UnsupportedInput(f"{self.adapter_name} requires a non-empty 'text' string")
        raw_terms = request.value.get("terms")
        if not isinstance(raw_terms, list) or not raw_terms:
            raise UnsupportedInput(f"{self.adapter_name} requires a non-empty 'terms' list")

        doc_context: dict[str, Any] = {
            k: v for k, v in request.value.items() if k not in {"text", "terms", "extractor"}
        }
        doc_context.update(request.context)
        doc_extractor = request.value.get("extractor") or request.context.get("extractor")

        term_texts: list[str] = []
        parsed_terms: list[Mapping[str, Any]] = []
        for idx, item in enumerate(raw_terms):
            if isinstance(item, str):
                entry: Mapping[str, Any] = {"text": item}
            elif isinstance(item, Mapping):
                entry = item
            else:
                raise UnsupportedInput(
                    f"{self.adapter_name} term {idx} must be a mapping or string, "
                    f"got {type(item).__name__}"
                )
            text_val = entry.get("text")
            if not isinstance(text_val, str) or not text_val.strip():
                raise UnsupportedInput(f"{self.adapter_name} term {idx} requires non-empty 'text'")
            term_texts.append(text_val)
            parsed_terms.append(entry)

        records: list[SourceRecord] = []
        for idx, entry in enumerate(parsed_terms):
            mention_text = term_texts[idx]
            start = entry.get("start")
            end = entry.get("end")
            if start is not None and not isinstance(start, int):
                raise UnsupportedInput(f"{self.adapter_name} term {idx} 'start' must be an int")
            if end is not None and not isinstance(end, int):
                raise UnsupportedInput(f"{self.adapter_name} term {idx} 'end' must be an int")

            fidelity = compute_mention_fidelity(mention_text, source_text, start=start, end=end)
            extractor = str(entry.get("extractor") or doc_extractor or self.adapter_name)
            siblings = [t for j, t in enumerate(term_texts) if j != idx]

            row: dict[str, Any] = dict(doc_context)
            row.setdefault("paragraph", source_text)
            row["sentence"] = str(
                entry.get("sentence")
                or _containing_sentence(source_text, mention_text, start=start, end=end)
            )
            if siblings:
                row["extracted_terms"] = siblings
            if entry.get("role") is not None:
                row["role"] = entry["role"]
            for nested_key in ("context", "attributes"):
                nested = entry.get(nested_key)
                if isinstance(nested, Mapping):
                    row.update(nested)

            record_id = row.get("record_id") or request.context.get("record_id")
            extensions: dict[str, Any] = {
                "term_index": idx,
                "_mention_extractor": extractor,
                "_mention_fidelity": fidelity.value,
                "_mention_detection_method": self.adapter_name,
            }
            if start is not None:
                extensions["_mention_start"] = start
            if end is not None:
                extensions["_mention_end"] = end
            sem_types = _coerce_semantic_types(entry.get("semantic_type") or entry.get("type"))
            if sem_types:
                extensions["_mention_semantic_types"] = sem_types

            records.append(
                SourceRecord(
                    source_type=SourceType.EXTRACTED_TERMS,
                    raw_value=mention_text,
                    source_system=request.source_system,
                    record_id=None if record_id is None else str(record_id),
                    row=row,
                    extensions=extensions,
                )
            )
        return records
