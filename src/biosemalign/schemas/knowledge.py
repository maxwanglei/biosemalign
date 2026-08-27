"""Candidate concepts and the Knowledge Package handed to the adjudicator."""

from __future__ import annotations

from pydantic import Field, field_validator

from biosemalign.enums import Authority, ConceptStatus, RetrievalChannel
from biosemalign.schemas.base import BioSemAlignModel, ExtensibleModel, VersionedModel
from biosemalign.schemas.provenance import ResourceProvenance, require_unique_resources

__all__ = [
    "CandidateConcept",
    "CodingPolicy",
    "ConceptEvidence",
    "Definition",
    "KnowledgeConflict",
    "KnowledgePackage",
    "KnowledgeStatement",
    "RetrievalEvidence",
]


class Definition(BioSemAlignModel):
    """A textual definition, with the resource that asserted it."""

    text: str
    source: str = Field(description="Asserting resource, e.g. 'NCI', 'MSH', 'RxNorm'.")
    authority: Authority = Authority.ONTOLOGY_DEFINITION


class RetrievalEvidence(BioSemAlignModel):
    """How one retrieval channel found this candidate (§6.3).

    Kept per-channel rather than collapsed into a single score, because
    agreement between independent channels is one of the routing features
    (§9.1) and because retrieval failure must stay measurable separately from
    adjudication failure.
    """

    channel: RetrievalChannel
    query: str = Field(description="The string or identifier actually sent to the provider.")
    provider: str
    score: float | None = None
    rank: int | None = None
    source_atom_id: str | None = Field(
        default=None,
        description="Atom-level identifier that matched, e.g. an RxNorm RXAUI.",
    )
    source_vocabulary: str | None = Field(
        default=None,
        description=(
            "The source vocabulary whose atom matched, e.g. 'RXNORM', 'MMSL', 'GS', "
            "'NDDF'. A match against a non-RxNorm source atom is a weaker signal "
            "than a match against RXNORM itself and must stay distinguishable."
        ),
    )


class KnowledgeStatement(ExtensibleModel):
    """One subject–predicate–object assertion retrieved from a resource (§6.7)."""

    subject_id: str
    predicate: str
    object_id: str
    subject_label: str | None = None
    object_label: str | None = None
    source_resource: str = Field(description="Resource that asserted this, e.g. 'RxNorm'.")
    authority: Authority
    direct: bool = Field(
        default=True,
        description="False when the statement was inferred rather than directly asserted.",
    )
    evidence_type: str | None = None
    confidence: float | None = None
    provenance_note: str | None = None


class CandidateConcept(ExtensibleModel):
    """One enriched candidate the adjudicator may choose from (§6.4, §11.3).

    Several fields here are absent from the design document and were added
    after probing RxNav showed they are load-bearing. ``status``,
    ``remapped_to`` and ``source_vocabularies`` exist because the approximate
    matcher happily returns obsolete concepts from non-RxNorm source
    vocabularies: without them a ``NotCurrent`` MMSL hit is indistinguishable
    from a current RxNorm concept right up until enrichment silently returns
    nothing.
    """

    candidate_id: str = Field(description="Identifier within the vocabulary, e.g. an RxCUI.")
    preferred_label: str
    vocabulary: str
    vocabulary_version: str | None = None
    concept_class: str | None = Field(
        default=None, description="Vocabulary-native class, e.g. an RxNorm TTY such as 'IN'."
    )
    semantic_types: list[str] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    relationships: list[KnowledgeStatement] = Field(default_factory=list)
    retrieval_evidence: list[RetrievalEvidence] = Field(default_factory=list)
    status: ConceptStatus = ConceptStatus.UNKNOWN
    remapped_to: list[str] = Field(
        default_factory=list,
        description="Replacement identifiers when this concept was remapped.",
    )
    source_vocabularies: list[str] = Field(
        default_factory=list,
        description="Source vocabularies asserting this concept, e.g. ['RXNORM', 'MMSL'].",
    )

    @property
    def best_score(self) -> float:
        """Highest score across retrieval channels; 0.0 when unscored."""
        scores = [e.score for e in self.retrieval_evidence if e.score is not None]
        return max(scores) if scores else 0.0

    @property
    def channels(self) -> set[RetrievalChannel]:
        """Distinct channels that surfaced this candidate."""
        return {e.channel for e in self.retrieval_evidence}

    def related_ids(self, predicate: str) -> list[str]:
        """Object identifiers of relationships with the given predicate."""
        return [s.object_id for s in self.relationships if s.predicate == predicate]


class ConceptEvidence(BioSemAlignModel):
    """Canonical package evidence for one identifier.

    Adjudicators are allowed to choose identifiers, not to redefine their label,
    vocabulary, class, or lifecycle status.  This derived index gives validation
    one authoritative view over both directly retrieved candidates and concepts
    reached through permitted terminology relationships.
    """

    concept_id: str
    label: str | None = None
    vocabulary: str
    concept_class: str | None = None
    status: ConceptStatus = ConceptStatus.UNKNOWN
    retrieved_directly: bool = False
    source_candidate_ids: list[str] = Field(default_factory=list)
    relationship_predicates: list[str] = Field(default_factory=list)


class CodingPolicy(BioSemAlignModel):
    """A project rule the adjudicator must follow (§6.2, §14)."""

    policy_id: str
    description: str
    rule: str = Field(description="The instruction as it should be shown to the adjudicator.")
    applies_to: list[str] = Field(
        default_factory=list, description="Tasks or semantic types this policy governs."
    )
    authority: Authority = Authority.PROJECT_CODING_POLICY
    source: str | None = None


class KnowledgeConflict(BioSemAlignModel):
    """Two resources disagreeing, preserved rather than silently resolved (§6.8)."""

    description: str
    statements: list[KnowledgeStatement] = Field(default_factory=list)
    higher_authority: Authority
    lower_authority: Authority


class KnowledgePackage(VersionedModel):
    """The bounded, provenance-linked evidence bundle for one SAU (§11.4)."""

    sau_id: str
    candidates: list[CandidateConcept] = Field(default_factory=list)
    ontology_guidance: list[KnowledgeStatement] = Field(default_factory=list)
    umls_guidance: list[KnowledgeStatement] = Field(default_factory=list)
    kg_guidance: list[KnowledgeStatement] = Field(default_factory=list)
    coding_policies: list[CodingPolicy] = Field(default_factory=list)
    conflicts: list[KnowledgeConflict] = Field(default_factory=list)
    resource_provenance: list[ResourceProvenance] = Field(default_factory=list)

    @field_validator("resource_provenance")
    @classmethod
    def _resources_are_unique(cls, resources: list[ResourceProvenance]) -> list[ResourceProvenance]:
        require_unique_resources(resources)
        return resources

    @property
    def candidate_ids(self) -> set[str]:
        """Every identifier the adjudicator is permitted to select.

        The validator uses this to reject identifiers that were never
        retrieved, which is the structural guard against invented codes
        (§7.2, §28 criterion 6).
        """
        return {
            evidence.concept_id for evidence in self.qualified_concept_evidence_index().values()
        }

    def concept_evidence_index(self) -> dict[str, ConceptEvidence]:
        """Compatibility ID-only index, omitting vocabulary-ambiguous IDs.

        Correctness-sensitive callers should use
        :meth:`qualified_concept_evidence_index` or :meth:`evidence_for`.
        """
        index: dict[str, ConceptEvidence] = {}
        ambiguous: set[str] = set()
        for evidence in self.qualified_concept_evidence_index().values():
            if evidence.concept_id in index:
                ambiguous.add(evidence.concept_id)
                index.pop(evidence.concept_id, None)
            elif evidence.concept_id not in ambiguous:
                index[evidence.concept_id] = evidence
        return index

    def qualified_concept_evidence_index(
        self,
    ) -> dict[tuple[str, str], ConceptEvidence]:
        """Index every evidenced identifier without trusting proposal metadata.

        Direct candidates take precedence over relationship-only cards because
        their own lifecycle and concept metadata were fetched. Relationship
        provenance is still merged so validation can distinguish an ingredient
        target from, for example, a reachable dose form that is evidence but is
        not a valid analysis-level selection.
        """
        index: dict[tuple[str, str], ConceptEvidence] = {}

        for candidate in self.candidates:
            key = (candidate.vocabulary.casefold(), candidate.candidate_id)
            index[key] = ConceptEvidence(
                concept_id=candidate.candidate_id,
                label=candidate.preferred_label,
                vocabulary=candidate.vocabulary,
                concept_class=candidate.concept_class,
                status=candidate.status,
                retrieved_directly=True,
                source_candidate_ids=[candidate.candidate_id],
            )

        for candidate in self.candidates:
            for statement in candidate.relationships:
                if statement.subject_id != candidate.candidate_id:
                    continue
                key = (candidate.vocabulary.casefold(), statement.object_id)
                evidence = index.get(key)
                if evidence is None:
                    evidence = ConceptEvidence(
                        concept_id=statement.object_id,
                        label=statement.object_label,
                        vocabulary=candidate.vocabulary,
                        concept_class=_target_concept_class(candidate, statement),
                        status=_relationship_target_status(candidate, statement),
                        source_candidate_ids=[candidate.candidate_id],
                        relationship_predicates=[statement.predicate],
                    )
                    index[key] = evidence
                    continue

                if candidate.candidate_id not in evidence.source_candidate_ids:
                    evidence.source_candidate_ids.append(candidate.candidate_id)
                if statement.predicate not in evidence.relationship_predicates:
                    evidence.relationship_predicates.append(statement.predicate)
                if evidence.label is None and statement.object_label is not None:
                    evidence.label = statement.object_label
                if evidence.concept_class is None:
                    evidence.concept_class = _target_concept_class(candidate, statement)

        for evidence in index.values():
            evidence.source_candidate_ids.sort()
            evidence.relationship_predicates.sort()
        return index

    def evidence_for_id(self, concept_id: str) -> ConceptEvidence | None:
        """Return evidence only when an ID is unique across package vocabularies."""
        return self.concept_evidence_index().get(concept_id)

    def evidence_for(self, vocabulary: str, concept_id: str) -> ConceptEvidence | None:
        """Return canonical evidence for a vocabulary-qualified identifier."""
        return self.qualified_concept_evidence_index().get((vocabulary.casefold(), concept_id))

    def candidate_by_id(self, candidate_id: str) -> CandidateConcept | None:
        """Return a candidate only when its bare identifier is unambiguous."""
        matches = [c for c in self.candidates if c.candidate_id == candidate_id]
        vocabularies = {candidate.vocabulary.casefold() for candidate in matches}
        if len(vocabularies) != 1:
            return None
        return matches[0] if matches else None

    def candidate_by_qualified_id(
        self, vocabulary: str, candidate_id: str
    ) -> CandidateConcept | None:
        """Return a directly retrieved candidate by qualified identity."""
        wanted = vocabulary.casefold()
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == candidate_id
                and candidate.vocabulary.casefold() == wanted
            ),
            None,
        )


def _target_concept_class(candidate: CandidateConcept, statement: KnowledgeStatement) -> str | None:
    if _is_release_bound_target(candidate, statement):
        inferred = {
            "has_ingredient": "IN",
            "has_precise_ingredient": "PIN",
        }.get(statement.predicate)
        if inferred is not None:
            return inferred
    value = statement.extensions.get("target_tty")
    return str(value) if value is not None else None


_RELEASE_BOUND_TARGET_PREDICATES = frozenset(
    {"has_ingredient", "has_precise_ingredient", "has_boss"}
)


def _is_release_bound_target(candidate: CandidateConcept, statement: KnowledgeStatement) -> bool:
    """Whether a relationship is canonical terminology evidence for its target."""
    return (
        candidate.vocabulary.casefold() == "rxnorm"
        and statement.predicate in _RELEASE_BOUND_TARGET_PREDICATES
        and statement.source_resource.casefold() == "rxnorm"
        and statement.authority is Authority.OFFICIAL_CODING_RELATIONSHIP
    )


def _relationship_target_status(
    candidate: CandidateConcept, statement: KnowledgeStatement
) -> ConceptStatus:
    if candidate.status is ConceptStatus.ACTIVE and _is_release_bound_target(candidate, statement):
        return ConceptStatus.ACTIVE
    return ConceptStatus.UNKNOWN
