"""Extension module contracts, operator registry, and built-in operators (§5.3)."""

# Subpackages register their built-in operators on import.
from biosemalign.modules import check as _check
from biosemalign.modules import clean as _clean
from biosemalign.modules import expand as _expand
from biosemalign.modules import link as _link
from biosemalign.modules import match as _match
from biosemalign.modules import resolve as _resolve
from biosemalign.modules import select as _select
from biosemalign.modules.base import (
    CandidateSet,
    Check,
    Cleaner,
    Expander,
    Linker,
    Matcher,
    OpContext,
    Operator,
    PassToNext,
    Resolver,
    Selector,
)
from biosemalign.modules.registry import (
    REGISTRY,
    OperatorRegistry,
    OpFactory,
    OpKind,
    available,
    build,
    register,
)

del _check, _clean, _expand, _link, _match, _resolve, _select

__all__ = [
    "REGISTRY",
    "CandidateSet",
    "Check",
    "Cleaner",
    "Expander",
    "Linker",
    "Matcher",
    "OpContext",
    "OpFactory",
    "OpKind",
    "Operator",
    "OperatorRegistry",
    "PassToNext",
    "Resolver",
    "Selector",
    "available",
    "build",
    "register",
]
