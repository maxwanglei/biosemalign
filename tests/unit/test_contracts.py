"""Persisted evidence must reject internally inconsistent or forged decisions."""

import json

import pytest
from pydantic import ValidationError
from tests.support import ALPHA, make_pipeline

from biosemalign.enums import (
    Authority,
    ConceptStatus,
    DecisionType,
    RelationshipType,
    RoutingStatus,
    Severity,
    ValidationStatus,
)
from biosemalign.routing.router import to_review_artifact
from biosemalign.schemas import ROOT_CONTRACTS, SCHEMA_VERSION
from biosemalign.schemas.artifact import AlignmentArtifact, BatchAlignmentRecord, ReviewArtifact
from biosemalign.schemas.base import BioSemAlignModel, ExtensibleModel, FrozenDict, deep_freeze
from biosemalign.schemas.decision import (
    AlignmentProposal,
    SelectedConcept,
    ValidationFinding,
    ValidationResult,
)
from biosemalign.schemas.export import export_schemas, main
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    KnowledgePackage,
    KnowledgeStatement,
    RetrievalEvidence,
)
from biosemalign.schemas.provenance import (
    ModelProvenance,
    Provenance,
    ResourceProvenance,
    RunContext,
    require_unique_resources,
)
from biosemalign.schemas.request import AlignmentRequest


def mutate(payload, path, value):
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return payload


def test_strict_contracts_keep_explicit_extensions_and_source_whitespace():
    with pytest.raises(ValidationError, match="Extra inputs"):
        BioSemAlignModel(unexpected=True)
    model = ExtensibleModel(extensions={"host": {"measurement": "  unchanged "}})
    assert ExtensibleModel.model_validate_json(model.model_dump_json()) == model
    assert model.extensions["host"]["measurement"] == "  unchanged "
    request = AlignmentRequest(
        source_type="structured_field",
        value="  Alpha \n",
        task="map",
        mapping_intent="normalize",
        target_terminology="RxNorm",
    )
    assert request.value == "  Alpha \n" and request.schema_version == SCHEMA_VERSION
    assert request.to_task().target_terminology == "RxNorm"
    for value in [" ", {}]:
        with pytest.raises(ValidationError, match="must not"):
            AlignmentRequest(
                source_type="structured_field",
                value=value,
                task="map",
                mapping_intent="normalize",
                target_terminology="RxNorm",
            )


def test_deep_freeze_prevents_nested_mutation_and_preserves_json():
    frozen = deep_freeze({"items": [{"name": "alpha"}], "set": {1, 2}})
    assert isinstance(frozen, FrozenDict)
    assert frozen["set"] == frozenset({1, 2})
    assert frozen["items"][0]["name"] == "alpha"
    for mutation in [
        lambda: frozen.update(other=1),
        lambda: frozen.pop("items"),
        lambda: frozen.clear(),
        lambda: frozen.setdefault("new", 1),
        lambda: frozen.__setitem__("new", 1),
        lambda: frozen.__delitem__("items"),
        lambda: frozen.popitem(),
        lambda: frozen.__ior__({"new": 1}),
    ]:
        with pytest.raises(TypeError, match="immutable"):
            mutation()
    model = ModelProvenance(backend="mock", generation_parameters={"nested": [{"temperature": 0}]})
    with pytest.raises(TypeError):
        model.generation_parameters["nested"][0]["temperature"] = 1
    assert json.loads(model.model_dump_json())["generation_parameters"] == {
        "nested": [{"temperature": 0}]
    }
    with pytest.raises(ValidationError, match="frozen"):
        model.backend = "changed"


def test_resource_identity_and_run_context_counting():
    resource = ResourceProvenance(
        resource_name="RxNorm", provider_name="rxnav", provider_version="test"
    )
    duplicate = resource.model_copy(update={"resource_name": "RXNORM", "provider_name": "RXNAV"})
    with pytest.raises(ValueError, match="duplicate resource"):
        require_unique_resources([resource, duplicate])
    with pytest.raises(ValidationError, match="duplicate resource"):
        Provenance(
            run_id="r",
            package_version="test",
            profile_name="p",
            profile_version="1",
            resources=[resource, duplicate],
        )
    run = RunContext(run_id="fixed", package_version="test", profile_name="p", profile_version="1")
    run.record_resource(resource_name="RxNorm", provider_name="rxnav", provider_version="test")
    run.record_resource(
        resource_name="RxNorm",
        provider_name="rxnav",
        provider_version="test",
        terminology_release="release",
        cache_hit=True,
    )
    run.record_resource(
        resource_name="Local", provider_name="local", provider_version="test", cache_hit=True
    )
    snapshot = run.freeze()
    assert [r.resource_name for r in snapshot.resources] == ["Local", "RxNorm"]
    rxnorm = snapshot.resources[1]
    assert rxnorm.cache_hits == rxnorm.network_calls == 1
    assert rxnorm.terminology_release == "release"
    run.record_resource(
        resource_name="RxNorm", provider_name="rxnav", provider_version="test", cache_hit=True
    )
    assert rxnorm.cache_hits == 1 and run.freeze().resources[1].cache_hits == 2


def test_schema_export_is_deterministic_and_check_fails_for_missing_or_stale(tmp_path, capsys):
    assert main(["--output-dir", str(tmp_path), "--check"]) == 1
    assert "out of date" in capsys.readouterr().err
    paths = export_schemas(tmp_path)
    assert len(paths) == len(ROOT_CONTRACTS) == 12
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["$schema"].endswith("2020-12/schema")
        assert payload["x-biosemalign-schema-version"] == SCHEMA_VERSION
    assert main(["--output-dir", str(tmp_path), "--check"]) == 0
    paths[0].write_text("{}")
    assert main(["--output-dir", str(tmp_path), "--check"]) == 1
    assert main(["--output-dir", str(tmp_path)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert main(["--output-dir", str(tmp_path), "--check"]) == 0


def test_complete_artifact_round_trip_preserves_every_stage(artifact):
    round_trip = AlignmentArtifact.model_validate_json(artifact.model_dump_json())
    assert round_trip == artifact
    assert round_trip.decision.validation_status is ValidationStatus.VALID_WITH_SEMANTIC_LOSS
    assert round_trip.decision.selected_concepts[0].concept_id == "1001"
    assert round_trip.knowledge_package.candidates[0].candidate_id == "2001"
    assert round_trip.proposal.source_level_concept.concept_id == "2001"


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("knowledge_package", "sau_id"), "wrong", "share SAU identity"),
        (("proposal", "sau_id"), "wrong", "share SAU identity"),
        (("decision", "sau_id"), "wrong", "share SAU identity"),
        (("mapping_key",), "wrong", "mapping_key"),
        (("run_id",), "wrong", "decision provenance"),
        (("sau", "provenance", "run_id"), "wrong", "SAU provenance"),
        (("validation", "rules_version"), "wrong", "routed decision"),
        (("decision", "provenance", "profile_version"), "wrong", "provenance profile_version"),
        (("decision", "provenance", "validation_rules_version"), "wrong", "validation rules"),
        (("decision", "provenance", "resources"), [], "omits a Knowledge Package"),
        (
            ("decision", "provenance", "resources", 0, "terminology_release"),
            "wrong",
            "resource provenance",
        ),
        (("decision", "provenance", "resources", 0, "cache_hits"), 0, "counters precede"),
        (("decision", "source_value"), "wrong", "source value"),
        (("decision", "record_id"), "wrong", "record ID"),
        (("decision", "preserved_attributes"), {}, "preserved attributes"),
        (("proposal", "preserved_attributes"), {}, "preserved attributes"),
        (("decision", "selected_concepts"), [], "selections"),
        (("decision", "decision"), "ABSTAINED", "outcome"),
        (("decision", "relationship_to_source"), "EXACT", "relationship"),
        (("decision", "source_level_concept"), None, "source_level_concept"),
        (("decision", "analysis_level_concept"), None, "analysis_level_concept"),
        (("decision", "rationale"), "forged rationale", "rationale"),
        (("decision", "routing_features"), None, "routing features"),
        (("decision", "routing_features", "channel_agreement"), 999, "routing features"),
        (("artifact_id",), "forged", "artifact_id"),
    ],
)
def test_artifact_rejects_inconsistent_persisted_evidence(artifact, path, value, message):
    payload = mutate(artifact.model_dump(mode="json"), path, value)
    with pytest.raises(ValidationError, match=message):
        AlignmentArtifact.model_validate(payload)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("vocabulary", "Other", "vocabulary"),
        ("label", "forged", "label"),
        ("concept_class", "OTHER", "class"),
        ("status", "obsolete", "status"),
    ],
)
def test_forged_proposal_metadata_cannot_become_canonical(artifact, field, value, message):
    payload = artifact.model_dump(mode="json")
    payload["proposal"]["selected_concepts"][0][field] = value
    payload["decision"]["selected_concepts"][0][field] = value
    with pytest.raises(ValidationError, match=message):
        AlignmentArtifact.model_validate(payload)


def test_unresolvable_evidence_reference_is_rejected(artifact):
    payload = artifact.model_dump(mode="json")
    payload["proposal"]["evidence_references"] = ["nonexistent"]
    payload["decision"]["evidence_references"] = ["nonexistent"]
    with pytest.raises(ValidationError, match="evidence references are absent"):
        AlignmentArtifact.model_validate(payload)


def test_artifact_rejects_forged_canonical_metadata_and_verdicts(artifact):
    # Persisted proposals cannot claim evidence that disappeared from the package.
    payload = artifact.model_dump(mode="json")
    candidate = payload["knowledge_package"]["candidates"][0]
    candidate["relationships"] = [
        item for item in candidate["relationships"] if item["object_id"] != "1001"
    ]
    with pytest.raises(ValidationError, match="absent from canonical package evidence"):
        AlignmentArtifact.model_validate(payload)

    for field, message in [
        ("selected_concepts", "selected concept metadata"),
        ("source_level_concept", "source-level concept metadata"),
        ("analysis_level_concept", "analysis-level concept metadata"),
    ]:
        payload = artifact.model_dump(mode="json")
        selected = payload["decision"][field]
        if isinstance(selected, list):
            selected = selected[0]
        selected["label"] = "forged label"
        with pytest.raises(ValidationError, match=message):
            AlignmentArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["proposal"]["requested_human_review"] = True
    with pytest.raises(ValidationError, match="validation status does not match"):
        AlignmentArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["validation"]["status"] = "VALID"
    payload["decision"]["validation"]["status"] = "VALID"
    with pytest.raises(ValidationError, match="validation status does not match"):
        AlignmentArtifact.model_validate(payload)


def test_artifact_warning_requires_review_even_when_json_is_valid(artifact):
    payload = artifact.model_dump(mode="json")
    warning = {"code": "SYNTHETIC_WARNING", "severity": "warning", "message": "review required"}
    for validation in [payload["validation"], payload["decision"]["validation"]]:
        validation["findings"].append(warning)
        validation["status"] = "VALID_WITH_WARNING"
    # Recompute the audit vector, leaving the inappropriate AUTO_ACCEPT routing untouched.
    from biosemalign.routing.features import extract_features
    from biosemalign.schemas.sau import SemanticAlignmentUnit

    payload["decision"]["routing_features"] = extract_features(
        sau=SemanticAlignmentUnit.model_validate(payload["sau"]),
        package=KnowledgePackage.model_validate(payload["knowledge_package"]),
        proposal=AlignmentProposal.model_validate(payload["proposal"]),
        validation=ValidationResult.model_validate(payload["validation"]),
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="must route to human review"):
        AlignmentArtifact.model_validate(payload)


def test_invalid_selection_is_preserved_only_in_rejected_proposal(tmp_path):
    class InvalidAdjudicator:
        def provenance(self):
            return ModelProvenance(backend="mock")

        def adjudicate(self, sau, package):
            selected = SelectedConcept(
                concept_id="never-retrieved", label="invented", vocabulary="RxNorm"
            )
            return AlignmentProposal(
                sau_id=sau.sau_id,
                decision=DecisionType.MAPPED,
                selected_concepts=[selected],
                relationship_to_source=RelationshipType.EXACT,
            )

    with make_pipeline(tmp_path / "rejected", adjudicator=InvalidAdjudicator()) as pipeline:
        rejected = pipeline.align_artifact(pipeline.make_request(ALPHA))
    assert rejected.validation.status is ValidationStatus.REJECTED
    assert rejected.proposal.selected_concepts[0].concept_id == "never-retrieved"
    assert rejected.decision.selected_concepts == []
    assert rejected.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert AlignmentArtifact.model_validate_json(rejected.model_dump_json()) == rejected
    payload = rejected.model_dump(mode="json")
    payload["decision"]["routing_status"] = "AUTO_ACCEPT"
    with pytest.raises(ValidationError, match="sanitized"):
        AlignmentArtifact.model_validate(payload)


def test_review_artifact_preserves_alternatives_and_rejects_forged_review_fields(artifact):
    review = to_review_artifact(artifact)
    assert ReviewArtifact.model_validate_json(review.model_dump_json()) == review
    assert review.review_item.candidates_considered
    for field, value, match in [
        ("sau_id", "wrong", "artifact SAU"),
        ("source_value", "wrong", "source value"),
        ("mention_text", "wrong", "mention text"),
        ("candidates_considered", [], "candidates"),
        ("validator_messages", ["forged"], "validator messages"),
        ("reason_for_review", ["forged"], "review reasons"),
        ("priority", 999, "priority"),
    ]:
        payload = review.model_dump(mode="json")
        payload["review_item"][field] = value
        with pytest.raises(ValidationError, match=match):
            ReviewArtifact.model_validate(payload)
    payload = review.model_dump(mode="json")
    payload["review_item"]["decision"]["rationale"] = "forged"
    with pytest.raises(ValidationError, match="decision does not match"):
        ReviewArtifact.model_validate(payload)


def test_batch_record_status_and_occurrence_integrity(synthetic_pipeline):
    records = list(
        synthetic_pipeline.align_batch_records(
            [{"drug": ALPHA, "id": "row-1"}], value_column="drug", id_columns=["id"]
        )
    )
    record = records[0]
    assert record.status == "success"
    assert BatchAlignmentRecord.model_validate_json(record.model_dump_json()) == record
    payload = record.model_dump(mode="json")
    payload["resource_provenance"] = []
    assert BatchAlignmentRecord.model_validate(payload).resource_provenance
    for field, value, match in [
        ("status", "error", "only successful"),
        ("error_code", "forged", "error fields"),
        ("source_value", "wrong", "source value"),
        ("record_id", "wrong", "record ID"),
        ("row_number", 2, "row number"),
    ]:
        payload = record.model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError, match=match):
            BatchAlignmentRecord.model_validate(payload)
    for status in ["error", "skipped"]:
        with pytest.raises(ValidationError, match="error_code"):
            BatchAlignmentRecord(row_number=1, status=status)
    with pytest.raises(ValidationError, match="require an artifact"):
        BatchAlignmentRecord(row_number=1, status="success")
    payload = record.model_dump(mode="json")
    payload["resource_provenance"][0]["resource_name"] = "Other"
    with pytest.raises(ValidationError, match="omits a decision resource"):
        BatchAlignmentRecord.model_validate(payload)
    payload = record.model_dump(mode="json")
    payload["resource_provenance"][0]["terminology_release"] = "different"
    with pytest.raises(ValidationError, match="contradicts decision provenance"):
        BatchAlignmentRecord.model_validate(payload)


def test_knowledge_index_keeps_qualified_ids_and_merges_relationship_evidence():
    def statement(subject, predicate, target, **updates):
        values = {
            "subject_id": subject,
            "predicate": predicate,
            "object_id": target,
            "source_resource": "RxNorm",
            "authority": Authority.OFFICIAL_CODING_RELATIONSHIP,
        }
        values.update(updates)
        return KnowledgeStatement(**values)

    direct = CandidateConcept(
        candidate_id="same",
        preferred_label="canonical",
        vocabulary="RxNorm",
        status=ConceptStatus.ACTIVE,
        relationships=[
            statement("same", "has_ingredient", "target", object_label="ingredient"),
            statement("wrong-subject", "has_ingredient", "ignored"),
            statement("same", "has_dose_form", "form", extensions={"target_tty": "DF"}),
        ],
    )
    other = CandidateConcept(
        candidate_id="same", preferred_label="other vocabulary", vocabulary="Other"
    )
    second = CandidateConcept(
        candidate_id="second",
        preferred_label="second",
        vocabulary="RxNorm",
        status=ConceptStatus.ACTIVE,
        relationships=[
            statement("second", "has_ingredient", "target"),
            statement("second", "has_precise_ingredient", "precise"),
            statement("second", "has_boss", "target"),
        ],
    )
    package = KnowledgePackage(sau_id="s", candidates=[direct, other, second])
    assert package.evidence_for_id("same") is None and package.candidate_by_id("same") is None
    assert package.candidate_by_id("missing") is None
    assert package.candidate_by_qualified_id("RXNORM", "same") is direct
    target = package.evidence_for("RxNorm", "target")
    assert target.source_candidate_ids == ["same", "second"]
    assert target.relationship_predicates == ["has_boss", "has_ingredient"]
    assert target.status is ConceptStatus.ACTIVE and target.concept_class == "IN"
    assert package.evidence_for("RxNorm", "precise").concept_class == "PIN"
    assert package.evidence_for("RxNorm", "form").status is ConceptStatus.UNKNOWN
    assert "ignored" not in package.candidate_ids
    assert direct.related_ids("has_ingredient") == ["target", "ignored"]
    assert direct.best_score == 0 and direct.channels == set()
    direct.retrieval_evidence = [
        RetrievalEvidence(channel="exact_lookup", query="x", provider="synthetic", score=None)
    ]
    assert direct.best_score == 0
    assert len(direct.channels) == 1


def test_direct_candidate_metadata_takes_precedence_over_relationship_only_cards():
    from biosemalign.enums import RetrievalChannel

    target = CandidateConcept(
        candidate_id="target",
        preferred_label="direct",
        vocabulary="RxNorm",
        concept_class="IN",
        status=ConceptStatus.NOT_CURRENT,
    )
    source = CandidateConcept(
        candidate_id="source",
        preferred_label="source",
        vocabulary="RxNorm",
        status=ConceptStatus.ACTIVE,
        relationships=[
            KnowledgeStatement(
                subject_id="source",
                predicate="has_ingredient",
                object_id="target",
                object_label="relationship label",
                source_resource="RxNorm",
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
            )
        ],
        retrieval_evidence=[
            RetrievalEvidence(
                channel=RetrievalChannel.EXACT_LOOKUP, query="source", provider="fixture", score=100
            ),
            RetrievalEvidence(
                channel=RetrievalChannel.LEXICAL, query="source", provider="fixture", score=90
            ),
        ],
    )
    package = KnowledgePackage(sau_id="s", candidates=[source, target])
    evidence = package.evidence_for_id("target")
    assert evidence.label == "direct" and evidence.status is ConceptStatus.NOT_CURRENT
    assert evidence.retrieved_directly and evidence.source_candidate_ids == ["source", "target"]
    assert package.candidate_by_id("target") is target
    assert source.best_score == 100 and len(source.channels) == 2


def test_validation_findings_keep_warning_and_error_separate():
    error = ValidationFinding(code="E", severity=Severity.ERROR, message="error")
    warning = ValidationFinding(code="W", severity=Severity.WARNING, message="warning")
    finding = ValidationFinding(code="I", severity=Severity.INFO, message="info")
    result = ValidationResult(
        status=ValidationStatus.REJECTED, rules_version="test", findings=[error, warning, finding]
    )
    assert result.errors == [error] and result.warnings == [warning]
