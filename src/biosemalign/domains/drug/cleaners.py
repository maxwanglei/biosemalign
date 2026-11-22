"""Drug-domain non-destructive cleaners (`drug.extract_strength`)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import OpContext
from biosemalign.modules.clean.builtin import with_retrieval_text
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["ExtractStrengthCleaner", "extract_strength_factory"]

_WHITESPACE = re.compile(r"\s+")
_STRENGTH = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|l|unit|units|iu|%)(?:\s*/\s*\d*(?:\.\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|ml|l|actuat|hr|h))?(?=\W|$)",
    re.IGNORECASE,
)


class ExtractStrengthCleaner:
    """Move a stated strength out of the query and into the ``strength`` slot.

    "TOPROL XL 50 MG" is queried as "TOPROL XL" while ``strength='50 MG'`` is
    kept as an attribute, so the decision records the strength rather than the
    cleaner erasing it. A strength the source already asserted is never
    overwritten.
    """

    op_id = "drug.extract_strength"
    version = "0.1.0"
    uses_model = False
    options: ClassVar[dict[str, Any]] = {}

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        base = sau.mention.retrieval_text or sau.mention.text
        matches = list(_STRENGTH.finditer(base))
        if not matches:
            return sau
        remainder = _WHITESPACE.sub(" ", _STRENGTH.sub(" ", base)).strip()
        if not remainder:
            return sau
        updated = with_retrieval_text(sau, remainder)
        if updated.attributes.strength is None:
            strength = ", ".join(_WHITESPACE.sub(" ", m.group(0)).strip() for m in matches)
            updated = updated.model_copy(
                update={"attributes": updated.attributes.model_copy(update={"strength": strength})}
            )
        return updated


def extract_strength_factory(
    context: OpContext, options: Mapping[str, Any]
) -> ExtractStrengthCleaner:
    del context
    if options:
        raise ConfigurationError(
            f"operator 'drug.extract_strength' takes no options, got {sorted(options)}"
        )
    return ExtractStrengthCleaner()
