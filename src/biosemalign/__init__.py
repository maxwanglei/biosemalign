"""BioSemAlign — context-aware, knowledge-guided biomedical semantic alignment.

Converts structured and unstructured biomedical inputs into context-preserving
Semantic Alignment Units, retrieves candidate concepts and task-specific
guidance from authoritative terminology resources, adjudicates the mapping
against that retrieved evidence, and reports the semantic relationship —
including what was lost — with full provenance.

Typical use::

    from biosemalign import BioSemAlign
    from biosemalign.schemas import AlignmentRequest

    pipeline = BioSemAlign.from_profile("faers_drug_normalization")
    result = pipeline.align_value("TOPROL XL 50 MG")
"""

from __future__ import annotations

from biosemalign._version import __version__
from biosemalign.enums import (
    DecisionType,
    EgressDataClass,
    RelationshipType,
    ResourceMode,
    RoutingStatus,
    SourceType,
    ValidationStatus,
)
from biosemalign.exceptions import BioSemAlignError
from biosemalign.pipelines.base import BioSemAlign

__all__ = [
    "BioSemAlign",
    "BioSemAlignError",
    "DecisionType",
    "EgressDataClass",
    "RelationshipType",
    "ResourceMode",
    "RoutingStatus",
    "SourceType",
    "ValidationStatus",
    "__version__",
]
