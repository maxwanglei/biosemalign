"""Built-in clean operators: non-destructive normalization and slot extraction."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import OpContext
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = [
    "AdapterCleaner",
    "CleanerResult",
    "UnicodeNormalizeCleaner",
    "with_retrieval_text",
]


class UnicodeNormalizeCleaner:
    """Query with NFKC-normalized, whitespace-collapsed text.

    Full-width characters, ligatures, and non-breaking spaces from PDF or
    spreadsheet exports otherwise defeat exact lookup.
    """

    op_id = "unicode_normalize"
    version = "0.1.0"
    uses_model = False
    options: ClassVar[dict[str, Any]] = {}

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        base = sau.mention.retrieval_text or sau.mention.text
        canonical = " ".join(unicodedata.normalize("NFKC", base).split()).strip()
        return with_retrieval_text(sau, canonical)


@dataclass(frozen=True, slots=True)
class CleanerResult:
    """What an external cleaner returns for one string."""

    canonical_text: str | None
    slots: Mapping[str, str] = field(default_factory=dict)


class AdapterCleaner:
    """Wrap an external cleaner (Aioli, DiAna, a lab regex set) as a Cleaner.

    The cleaner's canonical string becomes the retrieval query and its
    extracted slots fill empty attributes. The kernel's non-destructiveness
    check still applies, so a cleaner cannot rewrite the source mention or an
    attribute the source asserted.
    """

    def __init__(
        self,
        cleaner: Callable[[str], CleanerResult],
        *,
        op_id: str,
        version: str,
        uses_model: bool = False,
    ) -> None:
        self.cleaner = cleaner
        self.op_id = op_id
        self.version = version
        self.uses_model = uses_model
        self.options: dict[str, Any] = {}

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        result = self.cleaner(sau.mention.retrieval_text or sau.mention.text)
        updated = sau
        if result.canonical_text and result.canonical_text.strip():
            updated = with_retrieval_text(updated, result.canonical_text.strip())
        fills = {
            name: value
            for name, value in result.slots.items()
            if name in type(updated.attributes).model_fields
            and name != "extensions"
            and getattr(updated.attributes, name) is None
        }
        if fills:
            updated = updated.model_copy(
                update={"attributes": updated.attributes.model_copy(update=fills)}
            )
        return updated


def with_retrieval_text(sau: SemanticAlignmentUnit, text: str) -> SemanticAlignmentUnit:
    if text == sau.mention.text:
        return sau
    return sau.model_copy(
        update={"mention": sau.mention.model_copy(update={"retrieval_text": text})}
    )


def _no_options(name: str, options: Mapping[str, Any]) -> None:
    if options:
        raise ConfigurationError(f"operator {name!r} takes no options, got {sorted(options)}")


def _unicode_normalize(context: OpContext, options: Mapping[str, Any]) -> UnicodeNormalizeCleaner:
    del context
    _no_options("unicode_normalize", options)
    return UnicodeNormalizeCleaner()
