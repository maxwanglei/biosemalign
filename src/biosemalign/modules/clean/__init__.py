"""Clean module: built-in non-destructive text and slot cleaners."""

from biosemalign.modules.clean.builtin import (
    AdapterCleaner,
    CleanerResult,
    UnicodeNormalizeCleaner,
    _unicode_normalize,
)
from biosemalign.modules.registry import register

__all__ = [
    "AdapterCleaner",
    "CleanerResult",
    "UnicodeNormalizeCleaner",
]

register("clean", "unicode_normalize", _unicode_normalize)
