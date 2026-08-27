"""Controlled vocabularies for BioSemAlign.

Every enumerated value the framework can emit is defined here so there is a
single authority for the terms that appear in serialized output, in evaluation
reports, and in the AL-MedLit review contract. Section references point at
``docs/BioSemAlign System Design and Implementation Plan.md``.
"""

from enum import StrEnum

__all__ = [
    "Authority",
    "ConceptStatus",
    "DecisionType",
    "EgressDataClass",
    "ErrorCategory",
    "MentionMode",
    "ReadinessStatus",
    "RelationshipType",
    "ResourceMode",
    "RetrievalChannel",
    "RoutingStatus",
    "SemanticLossKind",
    "SemanticType",
    "Severity",
    "SourceType",
    "ValidationStatus",
]


class Severity(StrEnum):
    """Severity of a single validator finding.

    Distinct from :class:`ValidationStatus`, which is the verdict on the whole
    proposal: one ERROR finding forces the verdict to ``REJECTED``, while any
    number of INFO findings does not.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SourceType(StrEnum):
    """How the caller is handing us the thing to be mapped (§5.2)."""

    STRUCTURED_FIELD = "structured_field"
    STRUCTURED_RECORD = "structured_record"
    STRUCTURED_BATCH = "structured_batch"
    UNSTRUCTURED_TEXT = "unstructured_text"


class MentionMode(StrEnum):
    """Whether the caller identified the span or we must find it (§5.2)."""

    USER_SPECIFIED = "user_specified"
    AUTOMATIC_DETECTION = "automatic_detection"


class SemanticType(StrEnum):
    """Provisional entity type assigned by the SAU Builder (§5.5).

    This is a *provisional* typing decision, deliberately separate from concept
    identity. Module 3 performs the final concept selection.
    """

    DRUG = "drug"
    DRUG_PRODUCT = "drug_product"
    DRUG_INGREDIENT = "drug_ingredient"
    METABOLITE = "metabolite"
    DISEASE = "disease"
    PHENOTYPE = "phenotype"
    CLINICAL_FINDING = "clinical_finding"
    LABORATORY_TEST = "laboratory_test"
    LABORATORY_RESULT = "laboratory_result"
    GENE = "gene"
    PROTEIN = "protein"
    ENZYME = "enzyme"
    ASSAY = "assay"
    BIOLOGICAL_SYSTEM = "biological_system"
    ANATOMICAL_STRUCTURE = "anatomical_structure"
    PROCEDURE = "procedure"
    SPECIES = "species"
    DOSE = "dose"
    UNIT = "unit"
    TEMPORAL_EXPRESSION = "temporal_expression"
    UNKNOWN = "unknown"


class ReadinessStatus(StrEnum):
    """Whether an SAU carries enough information to be mapped (§5.8)."""

    READY_FOR_MAPPING = "READY_FOR_MAPPING"
    READY_WITH_LIMITED_CONTEXT = "READY_WITH_LIMITED_CONTEXT"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    NEEDS_DECOMPOSITION = "NEEDS_DECOMPOSITION"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"


class RelationshipType(StrEnum):
    """How the selected target concept relates to the source (§7.3).

    The values below the divider are reserved for later phenotype and
    cross-species work and are rejected by the v0.1 FAERS profile.
    """

    EXACT = "EXACT"
    TARGET_BROADER_THAN_SOURCE = "TARGET_BROADER_THAN_SOURCE"
    TARGET_NARROWER_THAN_SOURCE = "TARGET_NARROWER_THAN_SOURCE"
    CROSS_VOCABULARY_EQUIVALENT = "CROSS_VOCABULARY_EQUIVALENT"
    COMPONENT_OF = "COMPONENT_OF"
    MULTIPLE_COMPONENTS_REQUIRED = "MULTIPLE_COMPONENTS_REQUIRED"
    RELATED_NOT_EQUIVALENT = "RELATED_NOT_EQUIVALENT"
    UNSUPPORTED_MAPPING = "UNSUPPORTED_MAPPING"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

    # Reserved — not permitted in the first FAERS implementation.
    POTENTIAL_CLINICAL_MANIFESTATION = "POTENTIAL_CLINICAL_MANIFESTATION"
    MECHANISTICALLY_RELATED = "MECHANISTICALLY_RELATED"
    CROSS_SPECIES_RELATED = "CROSS_SPECIES_RELATED"
    SPECIES_SPECIFIC = "SPECIES_SPECIFIC"


class DecisionType(StrEnum):
    """Top-level outcome of adjudication (§11.5)."""

    MAPPED = "MAPPED"
    MULTI_CONCEPT_MAPPED = "MULTI_CONCEPT_MAPPED"
    NO_VALID_MAPPING = "NO_VALID_MAPPING"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    ABSTAINED = "ABSTAINED"


class ValidationStatus(StrEnum):
    """Deterministic validator verdict (§8.3)."""

    VALID = "VALID"
    VALID_WITH_SEMANTIC_LOSS = "VALID_WITH_SEMANTIC_LOSS"
    VALID_WITH_WARNING = "VALID_WITH_WARNING"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class RoutingStatus(StrEnum):
    """Where the decision goes next (§9.2)."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ABSTAIN = "ABSTAIN"


class Authority(StrEnum):
    """Knowledge authority hierarchy, highest first (§6.8).

    Lower-authority information must not be used to silently contradict
    higher-authority evidence. ``rank`` gives the numeric precedence.
    """

    SOURCE_INPUT = "source_input"
    TARGET_TERMINOLOGY = "target_terminology"
    OFFICIAL_CODING_RELATIONSHIP = "official_coding_relationship"
    OFFICIAL_CROSSWALK = "official_crosswalk"
    ONTOLOGY_DEFINITION = "ontology_definition"
    UMLS_LEXICAL_SEMANTIC = "umls_lexical_semantic"
    EXISTING_KG_RELATIONSHIP = "existing_kg_relationship"
    PROJECT_CODING_POLICY = "project_coding_policy"
    PRIOR_EXPERT_EXAMPLE = "prior_expert_example"
    LLM_INTERNAL_KNOWLEDGE = "llm_internal_knowledge"

    @property
    def rank(self) -> int:
        """1 is the most authoritative; 10 the least."""
        return _AUTHORITY_RANK[self]

    def outranks(self, other: "Authority") -> bool:
        return self.rank < other.rank


_AUTHORITY_RANK: dict[Authority, int] = {
    Authority.SOURCE_INPUT: 1,
    Authority.TARGET_TERMINOLOGY: 2,
    Authority.OFFICIAL_CODING_RELATIONSHIP: 3,
    Authority.OFFICIAL_CROSSWALK: 4,
    Authority.ONTOLOGY_DEFINITION: 5,
    Authority.UMLS_LEXICAL_SEMANTIC: 6,
    Authority.EXISTING_KG_RELATIONSHIP: 7,
    Authority.PROJECT_CODING_POLICY: 8,
    Authority.PRIOR_EXPERT_EXAMPLE: 9,
    Authority.LLM_INTERNAL_KNOWLEDGE: 10,
}


class RetrievalChannel(StrEnum):
    """Which retriever produced a candidate (§6.3).

    Retrieval failure and adjudication failure must be separable, so every
    candidate records the channel that surfaced it. The split between
    ``TERMINOLOGY_APPROXIMATE_CURRENT`` and ``TERMINOLOGY_APPROXIMATE_HISTORICAL``
    exists because RxNav's approximate matcher searches obsolete and
    non-RxNorm-source concepts by default; conflating the two silently produces
    dead-end candidates that cannot be enriched.
    """

    EXACT_LOOKUP = "exact_lookup"
    NORMALIZED_LOOKUP = "normalized_lookup"
    IDENTIFIER_LOOKUP = "identifier_lookup"
    OFFICIAL_CROSSWALK = "official_crosswalk"
    TERMINOLOGY_APPROXIMATE_CURRENT = "terminology_approximate_current"
    TERMINOLOGY_APPROXIMATE_HISTORICAL = "terminology_approximate_historical"
    SPELLING_SUGGESTION = "spelling_suggestion"
    LEXICAL = "lexical"
    EMBEDDING = "embedding"
    ONTOLOGY_NEIGHBORHOOD = "ontology_neighborhood"


class ConceptStatus(StrEnum):
    """Lifecycle state of a terminology concept (§8.1).

    Values mirror the states RxNav's ``historystatus`` endpoint reports so the
    provider layer does not have to lose information on the way in.
    """

    ACTIVE = "active"
    NOT_CURRENT = "not_current"
    REMAPPED = "remapped"
    QUANTIFIED = "quantified"
    OBSOLETE = "obsolete"
    NEVER_ACTIVE = "never_active"
    UNKNOWN = "unknown"

    @property
    def is_usable(self) -> bool:
        """Whether a concept in this state may be selected as a final mapping."""
        return self is ConceptStatus.ACTIVE


class SemanticLossKind(StrEnum):
    """What was dropped when a source was mapped to a coarser concept (§8.2).

    The design document types ``semantic_loss`` as ``list[str]``. It is
    structured here instead so that "how often does ingredient-level analysis
    discard route?" is a query rather than a string search.
    """

    STRENGTH = "strength"
    DOSE_FORM = "dose_form"
    ROUTE = "route"
    SALT_FORM = "salt_form"
    BRAND_IDENTITY = "brand_identity"
    COMBINATION_COMPONENT = "combination_component"
    SPECIFICITY = "specificity"
    ASSERTION_STATUS = "assertion_status"
    TEMPORALITY = "temporality"
    ANATOMICAL_SITE = "anatomical_site"
    SPECIMEN = "specimen"
    UNIT = "unit"
    OTHER = "other"


class ErrorCategory(StrEnum):
    """Failure taxonomy for evaluation and reviewer error capture (§22.2)."""

    INPUT_ERROR = "INPUT_ERROR"
    MENTION_DETECTION_ERROR = "MENTION_DETECTION_ERROR"
    SEMANTIC_TYPING_ERROR = "SEMANTIC_TYPING_ERROR"
    ATTRIBUTE_EXTRACTION_ERROR = "ATTRIBUTE_EXTRACTION_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    KNOWLEDGE_SELECTION_ERROR = "KNOWLEDGE_SELECTION_ERROR"
    ADJUDICATION_ERROR = "ADJUDICATION_ERROR"
    UNSUPPORTED_SPECIFICITY = "UNSUPPORTED_SPECIFICITY"
    RELATIONSHIP_CLASSIFICATION_ERROR = "RELATIONSHIP_CLASSIFICATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ROUTING_ERROR = "ROUTING_ERROR"
    RESOURCE_COVERAGE_GAP = "RESOURCE_COVERAGE_GAP"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class ResourceMode(StrEnum):
    """External-resource access policy for a run (§20.3)."""

    ONLINE = "ONLINE"
    """Always query live services and replace the matching cache entry."""

    CACHE_PREFER = "CACHE_PREFER"
    """Replay a valid cache entry first; query live services only on a miss."""

    FROZEN = "FROZEN"
    """Serve only a release-bound cache snapshot; never use the network."""

    OFFLINE = "OFFLINE"
    """Deprecated strict cache-only compatibility mode for legacy fixtures."""


class EgressDataClass(StrEnum):
    """Privacy classification applied before an external request is attempted."""

    PUBLIC_TERMINOLOGY = "public_terminology"
    """Public release metadata, concept identifiers, and ontology relationships."""

    SENSITIVE_TEXT = "sensitive_text"
    """Free-text or lexical source values that may contain confidential information."""

    DIRECT_IDENTIFIER = "direct_identifier"
    """Person, patient, report, or other directly identifying values."""
