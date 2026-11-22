"""Coding workflow contracts: every matched code with its evidence, no decision (§5.4)."""

from __future__ import annotations

from pydantic import Field, model_validator

from biosemalign.enums import ConceptStatus, RetrievalChannel
from biosemalign.schemas.base import BioSemAlignModel, VersionedModel, canonical_id
from biosemalign.schemas.knowledge import RetrievalEvidence
from biosemalign.schemas.links import CodeLink, LinkSet
from biosemalign.schemas.provenance import Provenance
from biosemalign.schemas.sau import SemanticAlignmentUnit

__all__ = ["CodedConcept", "CodingArtifact", "CodingTierAttempt"]


class CodedConcept(BioSemAlignModel):
    """One terminology concept attached to a term by coding."""

    vocabulary: str = Field(min_length=1)
    concept_code: str = Field(min_length=1)
    label: str
    concept_class: str | None = None
    status: ConceptStatus = ConceptStatus.UNKNOWN
    channels: list[RetrievalChannel] = Field(default_factory=list)
    retrieval_evidence: list[RetrievalEvidence] = Field(default_factory=list)
    target_links: list[CodeLink] = Field(default_factory=list)
    from_model_suggestion: bool = Field(
        default=False,
        description=(
            "True when found via model-written query text or a model-backed matcher (M4, G10)."
        ),
    )


class CodingTierAttempt(BioSemAlignModel):
    """What one coding tier queried and how many codes it matched."""

    tier: str = Field(min_length=1)
    query_text: str
    cleaners_applied: list[str] = Field(default_factory=list)
    matched_codes: int = Field(ge=0)
    coded: bool = False


class CodingArtifact(VersionedModel):
    """Auditable coding output for one SAU: every matched code and no decision."""

    artifact_id: str = Field(description="Unique content-addressed identifier for this artifact.")
    run_id: str = Field(min_length=1)
    mapping_key: str = Field(min_length=1)
    sau: SemanticAlignmentUnit
    codes: list[CodedConcept] = Field(default_factory=list)
    uncoded: bool
    coded_by_tier: str | None = None
    cleaned_by: list[str] = Field(default_factory=list)
    attempts: list[CodingTierAttempt] = Field(min_length=1)
    links: LinkSet | None = None
    provenance: Provenance

    @staticmethod
    def compute_artifact_id(
        *,
        run_id: str,
        sau: SemanticAlignmentUnit,
        codes: list[CodedConcept],
        uncoded: bool,
        coded_by_tier: str | None,
        cleaned_by: list[str],
        attempts: list[CodingTierAttempt],
        links: LinkSet | None,
        provenance: Provenance,
    ) -> str:
        """Compute the deterministic content-addressed identifier for a CodingArtifact."""
        payload = {
            "run_id": run_id,
            "sau": sau.model_dump(mode="json"),
            "codes": [code.model_dump(mode="json") for code in codes],
            "uncoded": uncoded,
            "coded_by_tier": coded_by_tier,
            "cleaned_by": list(cleaned_by),
            "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
            "links": links.model_dump(mode="json") if links is not None else None,
            "provenance": provenance.model_dump(mode="json"),
        }
        return canonical_id("COD", payload, length=32)

    @model_validator(mode="after")
    def _validate_coding_artifact(self) -> CodingArtifact:
        if self.mapping_key != self.sau.mapping_key:
            raise ValueError("coding artifact mapping_key does not match its SAU")
        if self.run_id != self.provenance.run_id:
            raise ValueError("coding artifact run_id does not match provenance")
        if self.run_id != self.sau.provenance.run_id:
            raise ValueError("coding artifact run_id does not match SAU provenance")
        for field_name in (
            "package_version",
            "profile_name",
            "profile_version",
            "source_adapter",
            "source_adapter_version",
            "identifier_hmac_key_fingerprint",
        ):
            if getattr(self.sau.provenance, field_name) != getattr(self.provenance, field_name):
                raise ValueError(f"coding provenance {field_name} does not match SAU provenance")
        expected_uncoded = len(self.codes) == 0
        if self.uncoded is not expected_uncoded:
            raise ValueError("uncoded flag must agree with whether codes is empty")

        seen_codes: set[tuple[str, str]] = set()
        for code in self.codes:
            key = (code.vocabulary.casefold(), code.concept_code)
            if key in seen_codes:
                raise ValueError(f"duplicate coded concept {code.vocabulary}:{code.concept_code}")
            seen_codes.add(key)
            requires_model_mark = self.sau.mention.retrieval_text_from_model or any(
                ev.from_model_suggestion for ev in code.retrieval_evidence
            )
            if requires_model_mark and not code.from_model_suggestion:
                raise ValueError(
                    "coded concept from model-written text or model retrieval must set "
                    "from_model_suggestion=True"
                )

        if self.provenance.workflow is not None:
            configured = [tier.name for tier in self.provenance.workflow.tiers]
            attempted = [attempt.tier for attempt in self.attempts]
            expected_order = configured if self.uncoded else configured[: len(attempted)]
            if attempted != expected_order:
                raise ValueError("coding attempts do not follow the recorded workflow order")

        if self.uncoded:
            if self.coded_by_tier is not None:
                raise ValueError("uncoded artifact must not name coded_by_tier")
            if self.cleaned_by:
                raise ValueError("uncoded artifact must not name cleaned_by cleaners")
            if any(attempt.coded or attempt.matched_codes != 0 for attempt in self.attempts):
                raise ValueError("uncoded artifact cannot record a coded tier attempt")
        else:
            *earlier, last = self.attempts
            if any(attempt.coded or attempt.matched_codes != 0 for attempt in earlier):
                raise ValueError("only the final tier attempt may code an SAU")
            if not last.coded or last.matched_codes != len(self.codes):
                raise ValueError("final coding attempt must reflect the matched codes count")
            if self.coded_by_tier != last.tier:
                raise ValueError("coded_by_tier must name the tier that coded the SAU")
            if self.cleaned_by != last.cleaners_applied:
                raise ValueError("cleaned_by must match the final tier's cleaners_applied")

        if self.links is None:
            if any(code.target_links for code in self.codes):
                raise ValueError("coded concepts carry target_links without an artifact LinkSet")
        else:
            allowed_links = self.links.links
            for code in self.codes:
                code_key = (code.vocabulary.casefold(), code.concept_code)
                for link in code.target_links:
                    if (link.source.vocabulary.casefold(), link.source.concept_code) != code_key:
                        raise ValueError(
                            "coded concept target_link source does not match the concept"
                        )
                    if link not in allowed_links:
                        raise ValueError(
                            "coded concept target_link is absent from the artifact LinkSet"
                        )

        expected_id = CodingArtifact.compute_artifact_id(
            run_id=self.run_id,
            sau=self.sau,
            codes=self.codes,
            uncoded=self.uncoded,
            coded_by_tier=self.coded_by_tier,
            cleaned_by=self.cleaned_by,
            attempts=self.attempts,
            links=self.links,
            provenance=self.provenance,
        )
        if self.artifact_id != expected_id:
            raise ValueError("coding artifact_id does not match its canonical payload")
        return self
