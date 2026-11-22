"""Public pipeline and deterministic baseline behavior, including failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.support import (
    ALPHA,
    AMBIGUOUS,
    BRAND,
    COMBINATION,
    HISTORICAL,
    MISSPELLING,
    NO_MATCH,
    PRODUCT,
    REJECTED,
    SALT,
    UNSUPPORTED,
    canonicalize,
    make_pipeline,
)

from biosemalign import BioSemAlign
from biosemalign.enums import (
    DecisionType,
    RelationshipType,
    RoutingStatus,
    SourceType,
    ValidationStatus,
)
from biosemalign.exceptions import (
    AdjudicationFailure,
    ConfigurationError,
    ProviderError,
    UnsupportedInput,
)
from biosemalign.modules import OpContext, build
from biosemalign.modules.resolve.backend import GenerationConfig
from biosemalign.modules.resolve.deterministic import DeterministicResolver
from biosemalign.modules.resolve.mock_backend import MockLLMBackend, MockResolver
from biosemalign.profiles.loader import load_profile
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import KnowledgePackage


@pytest.mark.parametrize(
    ("value", "decision", "relation", "selected", "route"),
    [
        (ALPHA, DecisionType.MAPPED, RelationshipType.EXACT, ["1001"], RoutingStatus.AUTO_ACCEPT),
        (
            PRODUCT,
            DecisionType.MAPPED,
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            ["1001"],
            RoutingStatus.AUTO_ACCEPT,
        ),
        (
            COMBINATION,
            DecisionType.MULTI_CONCEPT_MAPPED,
            RelationshipType.MULTIPLE_COMPONENTS_REQUIRED,
            ["1002", "1001"],
            RoutingStatus.AUTO_ACCEPT,
        ),
        (
            SALT,
            DecisionType.MAPPED,
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            ["1001"],
            RoutingStatus.AUTO_ACCEPT,
        ),
        (
            BRAND,
            DecisionType.MAPPED,
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            ["1001"],
            RoutingStatus.AUTO_ACCEPT,
        ),
        (
            AMBIGUOUS,
            DecisionType.MAPPED,
            RelationshipType.TARGET_BROADER_THAN_SOURCE,
            ["1001"],
            RoutingStatus.HUMAN_REVIEW,
        ),
        (
            MISSPELLING,
            DecisionType.MAPPED,
            RelationshipType.EXACT,
            ["1001"],
            RoutingStatus.AUTO_ACCEPT,
        ),
        (
            NO_MATCH,
            DecisionType.NO_VALID_MAPPING,
            RelationshipType.UNSUPPORTED_MAPPING,
            [],
            RoutingStatus.ABSTAIN,
        ),
        (
            REJECTED,
            DecisionType.NO_VALID_MAPPING,
            RelationshipType.UNSUPPORTED_MAPPING,
            [],
            RoutingStatus.ABSTAIN,
        ),
    ],
)
def test_current_deterministic_drug_cases(
    synthetic_pipeline: BioSemAlign,
    value: str,
    decision: DecisionType,
    relation: RelationshipType,
    selected: list[str],
    route: RoutingStatus,
) -> None:
    result = synthetic_pipeline.align_value(value)
    assert result.decision is decision
    assert result.relationship_to_source is relation
    assert [concept.concept_id for concept in result.selected_concepts] == selected
    assert result.routing_status is route
    assert all(resource.network_calls == 0 for resource in result.provenance.resources)


def test_product_projection_reports_exact_supported_losses(artifact: AlignmentArtifact) -> None:
    decision = artifact.decision
    assert {loss.kind.value for loss in decision.semantic_loss} == {
        "strength",
        "dose_form",
        "salt_form",
        "brand_identity",
        "route",
    }
    assert decision.source_level_concept.concept_id == "2001"
    assert decision.analysis_level_concept.concept_id == "1001"
    assert decision.preserved_attributes == {"route": "oral"}
    assert artifact.sau.source.raw_value == PRODUCT


def test_abstention_preserves_source_attributes(synthetic_pipeline: BioSemAlign) -> None:
    # M0 baseline limitation, repaired in v0.2: an abstention used to drop the
    # source's attributes and was then rejected as PRESERVED_ATTRIBUTES_MISMATCH.
    artifact = synthetic_pipeline.align_artifact(
        synthetic_pipeline.make_request(NO_MATCH, route="oral")
    )
    assert artifact.proposal.decision is DecisionType.NO_VALID_MAPPING
    assert artifact.proposal.preserved_attributes == {"route": "oral"}
    assert artifact.validation.status is ValidationStatus.VALID
    assert artifact.decision.decision is DecisionType.NO_VALID_MAPPING
    assert artifact.decision.routing_status is RoutingStatus.ABSTAIN
    assert artifact.decision.preserved_attributes == {"route": "oral"}


@pytest.mark.parametrize("route", [None, "oral"])
@pytest.mark.parametrize("value", [UNSUPPORTED, HISTORICAL])
def test_review_requested_abstention_routes_to_human_review(
    synthetic_pipeline: BioSemAlign, value: str, route: str | None
) -> None:
    # M0 baseline limitation, repaired in v0.2: the router sent requested-review
    # abstentions to ABSTAIN, which the artifact contract rejected.
    context = {} if route is None else {"route": route}
    artifact = synthetic_pipeline.align_artifact(synthetic_pipeline.make_request(value, **context))
    assert artifact.decision.decision is DecisionType.ABSTAINED
    assert artifact.validation.status is ValidationStatus.HUMAN_REVIEW_REQUIRED
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert "the adjudicator requested review" in artifact.decision.routing_reasons
    record = next(synthetic_pipeline.align_batch_records([{"drug": value}], value_column="drug"))
    assert record.status == "success"


@pytest.mark.parametrize(
    ("granularity", "value", "expected"),
    [
        ("precise_ingredient", PRODUCT, "1101"),
        ("precise_ingredient", SALT, "1101"),
        ("active_moiety", ALPHA, "1001"),
        ("active_moiety", PRODUCT, "1001"),
    ],
)
def test_existing_supported_projection_branches(
    tmp_path: Path, granularity: str, value: str, expected: str
) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.target.level = granularity
    with make_pipeline(tmp_path, profile=profile) as pipeline:
        decision = pipeline.align_value(value)
    assert decision.selected_concepts[0].concept_id == expected
    # This freezes current active_moiety/has_boss behavior; it is not a clinical
    # endorsement of that unfinished policy.


def test_unresolved_granularity_can_be_explicitly_represented_by_existing_profile(
    tmp_path: Path,
) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.task.relation_types.append(RelationshipType.RELATED_NOT_EQUIVALENT)
    with make_pipeline(tmp_path, profile=profile) as pipeline:
        result = pipeline.align_value(UNSUPPORTED)
        historic = pipeline.align_value(HISTORICAL)
    assert result.routing_status is RoutingStatus.HUMAN_REVIEW
    assert result.analysis_level_concept is None
    assert historic.routing_status is RoutingStatus.HUMAN_REVIEW
    assert historic.validation.status is ValidationStatus.REJECTED


def test_unsupported_granularity_abstains_instead_of_defaulting(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.target.level = "clinical_drug"
    with make_pipeline(tmp_path, profile=profile) as pipeline:
        result = pipeline.align_artifact(pipeline.make_request(ALPHA))
    assert result.decision.decision is DecisionType.ABSTAINED
    assert "no policy" in result.proposal.rationale
    assert result.decision.rationale is None
    assert {finding.code for finding in result.validation.errors} == {"UNSUPPORTED_GRANULARITY"}


def test_required_context_requests_review_without_manufacturing_attributes(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.context.fields = [
        f.model_copy(update={"required": True}) if f.name == "route" else f
        for f in profile.context.fields
    ]
    with make_pipeline(tmp_path, profile=profile) as pipeline:
        result = pipeline.align_value(ALPHA)
    assert result.decision is DecisionType.INSUFFICIENT_CONTEXT
    assert result.routing_status is RoutingStatus.HUMAN_REVIEW
    assert result.missing_information == ["route"]


def test_multiple_units_have_explicit_collection_api(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.input.source_type = SourceType.STRUCTURED_BATCH
    profile.input.value_field = "drugname"
    with make_pipeline(tmp_path, profile=profile) as pipeline:
        request = pipeline.make_request(ALPHA).model_copy(
            update={"value": {"rows": [{"drugname": ALPHA}, {"drugname": SALT}]}}
        )
        with pytest.raises(UnsupportedInput, match="requires exactly one"):
            pipeline.align(request)
        assert len(pipeline.align_all(request)) == 2
        assert len(pipeline.align_all_artifacts(request)) == 2


def test_pipeline_rejects_profile_mismatch_and_empty_unit_output(
    synthetic_pipeline: BioSemAlign, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = synthetic_pipeline.make_request(ALPHA)
    request.target_terminology = "different"
    with pytest.raises(UnsupportedInput, match="contradicts profile"):
        synthetic_pipeline.build_sau(request)
    request.target_terminology = "RxNorm"
    monkeypatch.setattr(synthetic_pipeline.sau_builder, "build", lambda *_: [])
    with pytest.raises(ValueError, match="no semantic alignment units"):
        synthetic_pipeline.align(request)
    records = list(synthetic_pipeline.align_batch_records([{"drug": ALPHA}], value_column="drug"))
    assert records[0].error_code == "UnsupportedInput"


def test_batch_preserves_occurrences_and_reuses_only_knowledge(
    synthetic_pipeline: BioSemAlign,
) -> None:
    rows = [{"id": "one", "drug": ALPHA}, {"id": "two", "drug": ALPHA}]
    records = list(
        synthetic_pipeline.align_batch_records(
            rows, value_column="drug", id_columns=["id"], start_row=7
        )
    )
    assert [record.row_number for record in records] == [7, 8]
    first, second = [record.artifact for record in records]
    assert first.mapping_key == second.mapping_key
    assert first.sau.sau_id != second.sau.sau_id
    assert [record.record_id for record in records] == ["one", "two"]
    assert (
        records[0].resource_provenance[0].cache_hits == records[1].resource_provenance[0].cache_hits
    )
    assert (
        len(
            list(
                synthetic_pipeline.align_batch([{"drug": ""}, {"drug": ALPHA}], value_column="drug")
            )
        )
        == 1
    )


def test_composite_source_identifiers_are_unambiguous(synthetic_pipeline: BioSemAlign) -> None:
    rows = [{"report": "a|b", "sequence": "c\\d", "drug": ALPHA}]
    record = next(
        synthetic_pipeline.align_batch_records(
            rows, value_column="drug", id_columns=["report", "sequence"]
        )
    )
    assert record.record_id == "a\\|b|c\\\\d"
    assert record.artifact.sau.source.row["report"] == "a|b"


def test_provider_failure_is_not_a_negative_mapping(
    synthetic_pipeline: BioSemAlign, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(_: str) -> Any:
        raise ProviderError("synthetic outage")

    monkeypatch.setattr(synthetic_pipeline.knowledge.primary_provider(), "find_by_name", failed)
    with pytest.raises(ProviderError):
        synthetic_pipeline.align_value(ALPHA)
    record = next(synthetic_pipeline.align_batch_records([{"drug": ALPHA}], value_column="drug"))
    assert record.status == "error"
    assert record.error_code == "ProviderError"
    assert record.artifact is None


def test_resolver_cannot_mutate_canonical_source_or_candidates(
    tmp_path: Path,
) -> None:
    class MutatingResolver(DeterministicResolver):
        def resolve(self, sau: Any, package: Any) -> AlignmentProposal:
            proposal = super().resolve(sau, package)
            sau.source.raw_value = "mutated"
            package.candidates.clear()
            return proposal

    profile = load_profile("faers_drug_normalization")
    with make_pipeline(tmp_path, resolver=MutatingResolver(profile)) as pipeline:
        result = pipeline.align_artifact(pipeline.make_request(ALPHA))
    assert result.sau.source.raw_value == ALPHA
    assert result.knowledge_package.candidates
    assert result.decision.routing_status is RoutingStatus.AUTO_ACCEPT


def test_bad_knowledge_identity_fails_before_resolution(
    synthetic_pipeline: BioSemAlign,
) -> None:
    run = synthetic_pipeline.new_run(run_id="fixed")
    sau = synthetic_pipeline.build_sau(synthetic_pipeline.make_request(ALPHA), run)[0]
    package = synthetic_pipeline.build_package(sau)
    package.sau_id = "different"
    with pytest.raises(ValueError, match="identity does not match"):
        synthetic_pipeline._align_with_package(sau, package, run)


@pytest.mark.parametrize("name", ["llm", "unknown"])
def test_unconfigured_or_unknown_resolvers_fail_explicitly(
    synthetic_pipeline: BioSemAlign, name: str
) -> None:
    context = OpContext(
        profile=synthetic_pipeline.profile,
        settings=synthetic_pipeline.settings,
        knowledge=synthetic_pipeline.knowledge,
    )
    with pytest.raises(ConfigurationError):
        build("resolve", name, context)


def test_mock_backend_preserves_schema_and_exhaustion_contract(artifact: AlignmentArtifact) -> None:
    backend = MockLLMBackend([artifact.proposal])
    options = {
        "messages": [{"role": "user", "content": "synthetic"}],
        "output_schema": AlignmentProposal,
        "generation_config": GenerationConfig(),
    }
    assert backend.generate_structured(**options) == artifact.proposal
    assert backend.calls[0]["output_schema"] == "AlignmentProposal"
    with pytest.raises(AdjudicationFailure, match="exhausted"):
        backend.generate_structured(**options)
    wrong = MockLLMBackend([KnowledgePackage(sau_id="test")])
    with pytest.raises(AdjudicationFailure, match="expected AlignmentProposal"):
        wrong.generate_structured(**options)
    mock = MockResolver()
    assert mock.provenance().backend == "mock"
    assert (
        mock.resolve(artifact.sau, artifact.knowledge_package).selected_concepts[0].concept_id
        == "2001"
    )
    assert (
        mock.resolve(artifact.sau, KnowledgePackage(sau_id=artifact.sau.sau_id)).decision
        is DecisionType.NO_VALID_MAPPING
    )


def test_disallowed_required_relationship_fails_closed(artifact: AlignmentArtifact) -> None:
    profile = load_profile("faers_drug_normalization")
    profile.task.relation_types = [RelationshipType.EXACT]
    resolver = DeterministicResolver(profile)
    with pytest.raises(AdjudicationFailure, match="cannot represent abstention"):
        resolver.resolve(artifact.sau, artifact.knowledge_package)


def test_collection_vs_single_outputs_are_consistent(synthetic_pipeline: BioSemAlign) -> None:
    request = synthetic_pipeline.make_request(ALPHA)
    assert canonicalize(synthetic_pipeline.align(request)) == canonicalize(
        synthetic_pipeline.align_all(request)[0]
    )
