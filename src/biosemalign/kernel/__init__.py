"""Alignment kernel, guards, and cascade combinators (§5.5)."""

from biosemalign.kernel.guards import (
    isolated_view,
    require_non_destructive,
    resolver_view,
    withheld_fields,
)
from biosemalign.kernel.kernel import KERNEL_VERSION, AlignmentKernel, PackageCache, Resolution
from biosemalign.kernel.tiers import Cascade, ParallelUnion, Tier

__all__ = [
    "KERNEL_VERSION",
    "AlignmentKernel",
    "Cascade",
    "PackageCache",
    "ParallelUnion",
    "Resolution",
    "Tier",
    "isolated_view",
    "require_non_destructive",
    "resolver_view",
    "withheld_fields",
]
