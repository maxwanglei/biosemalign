"""Match module: built-in local lexicon matchers."""

from biosemalign.modules.match.lexicon import (
    DenseMatcher,
    ExactMatcher,
    LexicalMatcher,
    LexicalMethod,
    _dense_factory,
    _exact_factory,
    _lexical_factory,
)
from biosemalign.modules.registry import register

__all__ = ["DenseMatcher", "ExactMatcher", "LexicalMatcher", "LexicalMethod"]

register("match", "exact", _exact_factory)
register("match", "lexical", _lexical_factory)
register("match", "dense", _dense_factory)
