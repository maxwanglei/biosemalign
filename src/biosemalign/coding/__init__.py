"""Coding: attach every code a term matches, deciding nothing (v0.3).

Two faces of one module:

* per SAU, a :class:`Coder` — a registered operator named under a tier's
  ``coding:`` key (``exact``, ``lexicon``, ``dense``, ``rxnav``) — returns a
  :data:`CodeSet`: candidates with how each was found;
* in bulk, :func:`code_terms` (``biosemalign code``) codes millions of terms
  in one set-based pass over a lexicon bundle.
"""

from typing import Any

from biosemalign.coding.bulk import CODER_VERSION, CodingReport, code_terms, term_key

__all__ = ["CODER_VERSION", "CodeSet", "Coder", "CodingReport", "code_terms", "term_key"]


def __getattr__(name: str) -> Any:
    # The protocols live with the other operator protocols; importing them
    # eagerly would import every registered operator.
    if name in {"Coder", "CodeSet"}:
        from biosemalign.ops import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
