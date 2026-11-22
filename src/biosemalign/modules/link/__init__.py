"""Link module: merging, enrichment, package assembly, concordance, and target projection."""

from biosemalign.modules.link.concordance import EXACT_CHANNELS, KeyLookup, assess_concordance
from biosemalign.modules.link.enrichment import enrich_candidates, supported_predicates
from biosemalign.modules.link.fusion import (
    CHANNEL_PRIORITY,
    RRF_K,
    channel_agreement,
    fuse_candidates,
    fuse_ranked,
    rank_key,
)
from biosemalign.modules.link.package_builder import (
    build_knowledge_package,
    derive_coding_policies,
    enforce_relationship_policy,
)
from biosemalign.modules.link.standard import MergeStrategy, StandardLinker, _standard_factory
from biosemalign.modules.link.targets import Reached, derive_link_set, targets_at
from biosemalign.modules.registry import register

__all__ = [
    "CHANNEL_PRIORITY",
    "EXACT_CHANNELS",
    "RRF_K",
    "KeyLookup",
    "MergeStrategy",
    "Reached",
    "StandardLinker",
    "assess_concordance",
    "build_knowledge_package",
    "channel_agreement",
    "derive_coding_policies",
    "derive_link_set",
    "enforce_relationship_policy",
    "enrich_candidates",
    "fuse_candidates",
    "fuse_ranked",
    "rank_key",
    "supported_predicates",
    "targets_at",
]

register("link", "standard", _standard_factory)
