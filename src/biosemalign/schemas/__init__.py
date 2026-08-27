"""BioSemAlign core data contracts (§11).

These schemas are the stable surface of the framework. Everything else —
adapters, providers, adjudicators, validators — exists to produce or consume
them, so they are versioned (:data:`SCHEMA_VERSION`) and their generated JSON
Schema is committed under ``docs/schemas/`` to make any change reviewable.
"""

from biosemalign.schemas.artifact import AlignmentArtifact, BatchAlignmentRecord, ReviewArtifact
from biosemalign.schemas.base import (
    SCHEMA_VERSION,
    BioSemAlignModel,
    ExtensibleModel,
    VersionedModel,
)
from biosemalign.schemas.decision import (
    AlignmentDecision,
    AlignmentProposal,
    ConceptValidation,
    ReviewDecision,
    ReviewQueueItem,
    RoutingFeatures,
    SelectedConcept,
    SemanticLoss,
    ValidationFinding,
    ValidationResult,
)
from biosemalign.schemas.evaluation import GoldMapping
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    CodingPolicy,
    Definition,
    KnowledgeConflict,
    KnowledgePackage,
    KnowledgeStatement,
    RetrievalEvidence,
)
from biosemalign.schemas.manifest import ResearchRunManifest
from biosemalign.schemas.provenance import (
    ModelProvenance,
    Provenance,
    ResourceProvenance,
    RunContext,
)
from biosemalign.schemas.request import AlignmentRequest, MappingTask
from biosemalign.schemas.sau import (
    ContextEnvelope,
    ContextualAttributes,
    Mention,
    ReadinessAssessment,
    SemanticAlignmentUnit,
)
from biosemalign.schemas.source import SourceRecord

#: Top-level contracts, exported to JSON Schema by ``biosemalign.schemas.export``.
ROOT_CONTRACTS: tuple[type[VersionedModel], ...] = (
    AlignmentRequest,
    SemanticAlignmentUnit,
    KnowledgePackage,
    AlignmentProposal,
    AlignmentDecision,
    AlignmentArtifact,
    BatchAlignmentRecord,
    ResearchRunManifest,
    ReviewArtifact,
    ReviewDecision,
    Provenance,
    GoldMapping,
)

__all__ = [
    "ROOT_CONTRACTS",
    "SCHEMA_VERSION",
    "AlignmentArtifact",
    "AlignmentDecision",
    "AlignmentProposal",
    "AlignmentRequest",
    "BatchAlignmentRecord",
    "BioSemAlignModel",
    "CandidateConcept",
    "CodingPolicy",
    "ConceptValidation",
    "ContextEnvelope",
    "ContextualAttributes",
    "Definition",
    "ExtensibleModel",
    "GoldMapping",
    "KnowledgeConflict",
    "KnowledgePackage",
    "KnowledgeStatement",
    "MappingTask",
    "Mention",
    "ModelProvenance",
    "Provenance",
    "ReadinessAssessment",
    "ResearchRunManifest",
    "ResourceProvenance",
    "RetrievalEvidence",
    "ReviewArtifact",
    "ReviewDecision",
    "ReviewQueueItem",
    "RoutingFeatures",
    "RunContext",
    "SelectedConcept",
    "SemanticAlignmentUnit",
    "SemanticLoss",
    "SourceRecord",
    "ValidationFinding",
    "ValidationResult",
    "VersionedModel",
]
