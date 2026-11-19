"""Linking: where codes lead in the target terminology (v0.3).

A linker builds the evidence package — the codes, their equivalence groups,
and the edges toward the target — and the kernel derives from it a typed
:class:`~biosemalign.schemas.links.LinkSet`: each matched code's links to
target codes at the requested granularity, as ``EQUIVALENCE`` or
``PROJECTION``, with ambiguity reported rather than resolved.

Built-in linkers are registered operators: ``cross_vocab_concordance`` (alias
``concordance``) over a lexicon bundle, and ``knowledge_package`` (RxNav and
default enrichment). Profiles name them under a tier's ``linking:`` key.
"""

from typing import Any

from biosemalign.linking.core import Reached, derive_link_set, targets_at
from biosemalign.schemas.links import CodeLink, LinkSet

__all__ = ["CodeLink", "LinkSet", "Linker", "Reached", "derive_link_set", "targets_at"]


def __getattr__(name: str) -> Any:
    # The protocol lives with the other operator protocols; importing it
    # eagerly would import every registered operator, which imports this.
    if name == "Linker":
        from biosemalign.ops.base import Linker

        return Linker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
