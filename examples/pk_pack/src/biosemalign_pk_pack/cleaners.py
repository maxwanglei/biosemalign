"""Non-destructive PK parameter cleaner (`pk.normalize_parameter`)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import OpContext
from biosemalign.modules.clean.builtin import with_retrieval_text
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["NormalizePKParameterCleaner", "normalize_pk_parameter_factory"]

_TRAILING_SPECIMEN = re.compile(r"\s*[\(\[]([^()\[\]]+)[\)\]]\s*$")


class NormalizePKParameterCleaner:
    """Strip a parenthetical specimen qualifier into the ``specimen`` attribute slot."""

    op_id = "pk.normalize_parameter"
    version = "0.1.0"
    uses_model = False
    options: ClassVar[dict[str, Any]] = {}

    def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
        base = sau.mention.retrieval_text or sau.mention.text
        match = _TRAILING_SPECIMEN.search(base)
        if not match:
            return sau
        head = base[: match.start()].strip()
        specimen = match.group(1).strip()
        if not head or not specimen:
            return sau
        updated = with_retrieval_text(sau, head)
        if updated.attributes.specimen is None:
            updated = updated.model_copy(
                update={"attributes": updated.attributes.model_copy(update={"specimen": specimen})}
            )
        return updated


def normalize_pk_parameter_factory(
    context: OpContext, options: Mapping[str, Any]
) -> NormalizePKParameterCleaner:
    del context
    if options:
        raise ConfigurationError(
            f"operator 'pk.normalize_parameter' takes no options, got {sorted(options)}"
        )
    return NormalizePKParameterCleaner()
