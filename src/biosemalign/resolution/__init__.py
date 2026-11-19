"""Resolution: decide from codes and links, or pass to the next tier (v0.3).

A :class:`Resolver` — named under a tier's ``resolution:`` key — reads the
package's codes and the kernel's typed links and returns a proposal or a
:class:`PassToNext` with its reason. Policy lives here: MPRINT's "done only
when the codes agree on an RxNorm ingredient" is the ``concordant`` resolver.
Built-ins: ``concordant`` (and ``concordant_exact``, its v0.2 name with
drug-specific pass reasons), ``adjudicator`` (deterministic rules), ``llm``.

Resolution never mints a decision: the kernel validates and routes every
proposal.
"""

from typing import Any

__all__ = ["PassToNext", "Resolver"]


def __getattr__(name: str) -> Any:
    if name in {"Resolver", "PassToNext"}:
        from biosemalign.ops import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
