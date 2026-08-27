"""Source adapters: structure translation only, no alignment logic (§5.3)."""

from biosemalign.input.adapters.base import SourceAdapter, get_adapter, register_adapter
from biosemalign.input.adapters.structured import (
    StructuredBatchAdapter,
    StructuredFieldAdapter,
    StructuredRecordAdapter,
    register_default_adapters,
)

__all__ = [
    "SourceAdapter",
    "StructuredBatchAdapter",
    "StructuredFieldAdapter",
    "StructuredRecordAdapter",
    "get_adapter",
    "register_adapter",
    "register_default_adapters",
]
