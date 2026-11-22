"""Consumer-side verification of persisted alignment artifacts (v0.2).

A downstream system should not have to trust that an artifact came out of the
kernel. Re-validating it from its serialized form re-derives every consistency
rule the artifact contract encodes — identities, canonical evidence for each
selected concept, the validation verdict, routing features, tier attribution,
and the content hash. With the profile that produced it, verification also
confirms the profile identity and that no prohibited relationship reached the
evidence the adjudicator saw.

Verification cannot re-query terminology providers; it proves the artifact is
internally consistent and policy-conformant, not that the terminology was
right.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.coding import CodingArtifact
from biosemalign.schemas.phenotyping import ConceptSet, ConceptSetArtifact
from biosemalign.storage.jsonl import read_jsonl

__all__ = ["VerificationResult", "verify_artifact", "verify_jsonl"]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The outcome of verifying one artifact."""

    artifact_id: str | None
    errors: list[str] = field(default_factory=list)
    row_number: int | None = None

    @property
    def valid(self) -> bool:
        return not self.errors


def verify_artifact(
    artifact: (
        AlignmentArtifact | CodingArtifact | ConceptSetArtifact | ConceptSet | Mapping[str, Any]
    ),
    *,
    profile: TaskProfile | None = None,
) -> VerificationResult:
    """Re-validate one artifact, optionally against the profile that produced it."""
    if isinstance(artifact, AlignmentArtifact | CodingArtifact | ConceptSetArtifact | ConceptSet):
        payload: Any = artifact.model_dump(mode="json")
    else:
        payload = artifact
    artifact_id = payload.get("artifact_id") if isinstance(payload, Mapping) else None
    is_concept_set = isinstance(artifact, ConceptSet) or (
        isinstance(payload, Mapping)
        and "concept_set_id" in payload
        and "artifact_digest" in payload
    )
    if is_concept_set:
        cs_id = payload.get("concept_set_id") if isinstance(payload, Mapping) else None
        try:
            validated_set = ConceptSet.model_validate(payload)
        except ValidationError as exc:
            return VerificationResult(
                artifact_id=cs_id if isinstance(cs_id, str) else None,
                errors=[_describe(error) for error in exc.errors()],
            )
        errors = _exported_set_profile_errors(validated_set, profile) if profile is not None else []
        return VerificationResult(artifact_id=validated_set.concept_set_id, errors=errors)

    is_concept_set_artifact = isinstance(artifact, ConceptSetArtifact) or (
        isinstance(payload, Mapping) and "expansion" in payload and "coding" in payload
    )
    if is_concept_set_artifact:
        try:
            validated_cs = ConceptSetArtifact.model_validate(payload)
        except ValidationError as exc:
            return VerificationResult(
                artifact_id=artifact_id if isinstance(artifact_id, str) else None,
                errors=[_describe(error) for error in exc.errors()],
            )
        errors = _concept_set_profile_errors(validated_cs, profile) if profile is not None else []
        return VerificationResult(artifact_id=validated_cs.artifact_id, errors=errors)

    is_coding = isinstance(artifact, CodingArtifact) or (
        isinstance(payload, Mapping)
        and "decision" not in payload
        and ("codes" in payload or "coded_by_tier" in payload)
    )
    if is_coding:
        try:
            validated_coding = CodingArtifact.model_validate(payload)
        except ValidationError as exc:
            return VerificationResult(
                artifact_id=artifact_id if isinstance(artifact_id, str) else None,
                errors=[_describe(error) for error in exc.errors()],
            )
        errors = _coding_profile_errors(validated_coding, profile) if profile is not None else []
        return VerificationResult(artifact_id=validated_coding.artifact_id, errors=errors)
    try:
        validated = AlignmentArtifact.model_validate(payload)
    except ValidationError as exc:
        return VerificationResult(
            artifact_id=artifact_id if isinstance(artifact_id, str) else None,
            errors=[_describe(error) for error in exc.errors()],
        )
    errors = _profile_errors(validated, profile) if profile is not None else []
    return VerificationResult(artifact_id=validated.artifact_id, errors=errors)


def verify_jsonl(
    path: str | Path, *, profile: TaskProfile | None = None
) -> Iterator[VerificationResult | None]:
    """Verify every artifact in an artifact or batch-record JSONL file.

    Yields ``None`` for batch records that carry no artifact (skipped or failed
    rows), so callers can account for every line.
    """
    for payload in read_jsonl(path):
        if "status" in payload and "row_number" in payload:
            row_number = payload.get("row_number")
            artifact = payload.get("artifact")
            if artifact is None:
                yield None
                continue
            result = verify_artifact(artifact, profile=profile)
            yield VerificationResult(
                artifact_id=result.artifact_id,
                errors=result.errors,
                row_number=row_number if isinstance(row_number, int) else None,
            )
            continue
        yield verify_artifact(payload, profile=profile)


def _profile_errors(artifact: AlignmentArtifact, profile: TaskProfile) -> list[str]:
    errors: list[str] = []
    provenance = artifact.decision.provenance
    if (provenance.profile_name, provenance.profile_version) != (
        profile.profile_name,
        profile.profile_version,
    ):
        errors.append(
            f"artifact was produced by profile {provenance.profile_name}@"
            f"{provenance.profile_version}, not {profile.profile_name}@{profile.profile_version}"
        )
    package = artifact.knowledge_package
    statements = [
        *(s for candidate in package.candidates for s in candidate.relationships),
        *package.ontology_guidance,
        *package.umls_guidance,
        *package.kg_guidance,
        *(s for conflict in package.conflicts for s in conflict.statements),
    ]
    leaked = sorted({s.predicate for s in statements if not profile.knowledge.permits(s.predicate)})
    if leaked:
        errors.append(f"knowledge package contains predicates the profile prohibits: {leaked}")
    return errors


def _coding_profile_errors(artifact: CodingArtifact, profile: TaskProfile) -> list[str]:
    errors: list[str] = []
    provenance = artifact.provenance
    if (provenance.profile_name, provenance.profile_version) != (
        profile.profile_name,
        profile.profile_version,
    ):
        errors.append(
            f"artifact was produced by profile {provenance.profile_name}@"
            f"{provenance.profile_version}, not {profile.profile_name}@{profile.profile_version}"
        )
    if artifact.links is not None:
        leaked = sorted(
            {
                pred
                for link in artifact.links.links
                for pred in link.predicates
                if not profile.knowledge.permits(pred)
            }
        )
        if leaked:
            errors.append(f"coding links contain predicates the profile prohibits: {leaked}")
    return errors


def _concept_set_profile_errors(artifact: ConceptSetArtifact, profile: TaskProfile) -> list[str]:
    errors: list[str] = []
    provenance = artifact.decision.provenance
    if (provenance.profile_name, provenance.profile_version) != (
        profile.profile_name,
        profile.profile_version,
    ):
        errors.append(
            f"artifact was produced by profile {provenance.profile_name}@"
            f"{provenance.profile_version}, not {profile.profile_name}@{profile.profile_version}"
        )
    leaked = sorted(
        {
            step.predicate
            for member in artifact.expansion.members
            for step in member.path
            if (
                step.predicate in profile.knowledge.prohibited_relationships
                if step.direction == "lexical"
                else not profile.knowledge.permits(step.predicate)
            )
        }
    )
    if leaked:
        errors.append(f"expansion contains predicates the profile prohibits: {leaked}")
    return errors


def _exported_set_profile_errors(concept_set: ConceptSet, profile: TaskProfile) -> list[str]:
    errors: list[str] = []
    if concept_set.terminology.casefold() != profile.target.terminology.casefold():
        errors.append(
            f"concept set targets terminology {concept_set.terminology!r}, not "
            f"{profile.target.terminology!r}"
        )
    return errors


def _describe(error: Mapping[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid"))
    return f"{location}: {message}" if location else message
