"""Source context and occurrence identity are preserved before retrieval."""

import hashlib
import hmac
from datetime import UTC

import pytest

from biosemalign.enums import ReadinessStatus, SourceType
from biosemalign.exceptions import UnsupportedInput
from biosemalign.input.adapters.base import get_adapter
from biosemalign.input.adapters.structured import (
    StructuredBatchAdapter,
    StructuredFieldAdapter,
    StructuredRecordAdapter,
)
from biosemalign.input.context import build_attributes, build_envelope
from biosemalign.input.readiness import assess_readiness
from biosemalign.input.sau_builder import SAUBuilder
from biosemalign.profiles.loader import load_profile
from biosemalign.schemas.provenance import RunContext
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.sau import ContextEnvelope, ContextualAttributes
from biosemalign.schemas.source import SourceRecord
from biosemalign.util import (
    lexical_similarity,
    normalize_text,
    stable_digest,
    stable_id,
    tokenize,
    utc_now,
)


def request(value=" ExampleAlpha ", **updates):
    fields = {
        "source_type": "structured_field",
        "value": value,
        "task": "drug_normalization",
        "mapping_intent": "ingredient_level_analysis",
        "target_terminology": "RxNorm",
        "desired_granularity": "ingredient",
        "source_system": "synthetic",
    }
    fields.update(updates)
    return AlignmentRequest(**fields)


def run_context():
    return RunContext(
        run_id="fixed",
        package_version="test",
        profile_name="faers_drug_normalization",
        profile_version="0.1.0",
    )


def test_context_aliases_preserve_conflict_and_source_metadata():
    source = SourceRecord(
        source_type=SourceType.STRUCTURED_FIELD,
        raw_value="  ExampleAlpha\n",
        source_system="explicit",
        record_id="row-1",
        table_name="DRUG",
        column_name="prod_ai",
        row={
            "SOURCE SYSTEM": "inferred",
            "table": "other",
            "column": "other",
            "primaryid": "other",
            "route_cod": "oral",
            "route": "intravenous",
            "role_cod": "PS",
            "indi_pt": "unknown",
            "dose_amt": 10,
            "dose_unit": "MG",
            "dose_form": "tablet",
            "negation": False,
            "certainty": "suspected",
            "custom": {"keep": [1]},
            "empty": " ",
            "null": None,
        },
    )
    envelope = build_envelope(source)
    assert (
        envelope.source_system,
        envelope.record_id,
        envelope.table_name,
        envelope.column_name,
    ) == ("explicit", "row-1", "DRUG", "prod_ai")
    assert (
        envelope.route,
        envelope.drug_role,
        envelope.indication,
        envelope.dose,
        envelope.unit,
    ) == ("oral", "PS", "unknown", "10", "MG")
    assert envelope.neighboring_fields["route"] == "intravenous"
    assert envelope.neighboring_fields["custom"] == {"keep": [1]}
    assert "empty" not in envelope.neighboring_fields
    attributes = build_attributes(source)
    assert attributes.negation is False
    assert attributes.certainty == "suspected"
    assert attributes.formulation == "tablet"
    assert attributes.route == "oral"
    assert source.raw_value == "  ExampleAlpha\n"


def test_readiness_requires_context_without_treating_false_as_missing():
    profile = load_profile("faers_drug_normalization")
    profile.input.required_context = ["route", "negation", "custom"]
    profile.input.optional_context = ["dose"]

    def assess(text, envelope, attributes):
        return assess_readiness(
            mention_text=text, envelope=envelope, attributes=attributes, profile=profile
        )

    assert (
        assess(" ", ContextEnvelope(), ContextualAttributes()).status
        is ReadinessStatus.UNSUPPORTED_INPUT
    )
    missing = assess("ExampleAlpha", ContextEnvelope(), ContextualAttributes())
    assert missing.status is ReadinessStatus.INSUFFICIENT_CONTEXT
    assert missing.missing_context == ["custom", "negation", "route"]
    envelope = ContextEnvelope(route="oral", neighboring_fields={"custom": 0, "unused": None})
    limited = assess("ExampleAlpha", envelope, ContextualAttributes(negation=False))
    assert limited.status is ReadinessStatus.READY_WITH_LIMITED_CONTEXT
    assert limited.is_mappable
    complete = assess("ExampleAlpha", envelope, ContextualAttributes(negation=False, dose="10"))
    assert complete.status is ReadinessStatus.READY_FOR_MAPPING
    assert complete.warnings == []
    assert not missing.is_mappable


def test_source_adapters_preserve_rows_and_validate_shapes():
    profile = load_profile("faers_drug_normalization")
    field = StructuredFieldAdapter()
    source = field.to_source_records(request(context={"record_id": 7}), profile)[0]
    assert source.raw_value == " ExampleAlpha " and source.record_id == "7"
    with pytest.raises(UnsupportedInput, match="expects a string"):
        field.to_source_records(request({"drug": "x"}), profile)
    adapter = StructuredRecordAdapter()
    with pytest.raises(UnsupportedInput, match="expects a mapping"):
        adapter.to_source_records(request(), profile)
    with pytest.raises(UnsupportedInput, match="value_field"):
        adapter.to_source_records(request({"drug": "x"}), profile)
    profile.input.value_field = "drug"
    for row, error in [
        ({"missing": "x"}, "does not contain"),
        ({"drug": None}, "empty"),
        ({"drug": " "}, "empty"),
    ]:
        with pytest.raises(UnsupportedInput, match=error):
            adapter.to_source_records(request(row), profile)
    source = adapter.to_source_records(
        request({"drug": "  X ", "record_id": 8, "route": "IV"}, context={"route": "oral"}), profile
    )[0]
    assert source.raw_value == "  X " and source.record_id == "8"
    assert source.row == {"record_id": 8, "route": "oral"}
    batch = StructuredBatchAdapter()
    with pytest.raises(UnsupportedInput, match="'rows' key"):
        batch.to_source_records(request({"bad": 1}), profile)
    with pytest.raises(UnsupportedInput, match="must be a list"):
        batch.to_source_records(request({"rows": "bad"}), profile)
    records = batch.to_source_records(request({"rows": [{"drug": "X"}, {"drug": "Y"}]}), profile)
    assert [record.extensions["batch_index"] for record in records] == [1, 2]
    assert all(record.source_type is SourceType.STRUCTURED_BATCH for record in records)
    with pytest.raises(UnsupportedInput, match="no adapter registered"):
        get_adapter(SourceType.UNSTRUCTURED_TEXT)


def test_sau_identity_changes_with_occurrence_but_mapping_key_reuses_equivalent_context():
    profile = load_profile("faers_drug_normalization")
    builder = SAUBuilder(profile)
    first = builder.build(
        request(context={"record_id": "one", "route": "oral", "role_cod": "PS"}), run_context()
    )[0]
    second = builder.build(
        request("examplealpha", context={"record_id": "two", "route": "oral", "role_cod": "SS"}),
        run_context(),
    )[0]
    changed = builder.build(request(context={"record_id": "one", "route": "IV"}), run_context())[0]
    repeat = builder.build(
        request(context={"record_id": "one", "route": "oral", "role_cod": "PS"}), run_context()
    )[0]
    assert first.mapping_key == second.mapping_key
    assert first.sau_id != second.sau_id
    assert first.mapping_key != changed.mapping_key
    assert first.sau_id == repeat.sau_id
    assert (
        first.mention.text == " ExampleAlpha " and first.mention.normalized_text == "examplealpha"
    )
    assert first.context.drug_role == "PS"
    keyed = SAUBuilder(profile, identifier_key="x" * 32).build(
        request(context={"record_id": "one", "route": "oral", "role_cod": "PS"}), run_context()
    )[0]
    assert keyed.sau_id != first.sau_id and keyed.mapping_key != first.mapping_key


def test_digest_matches_standard_sha_and_hmac_and_normalization_is_conservative():
    assert stable_digest({"b": 2, "a": 1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    expected = hmac.new(b"secret", b'{"a":1}', hashlib.sha256).hexdigest()
    assert stable_digest({"a": 1}, key="secret") == expected
    assert stable_digest({"a": 1}, key=b"secret", length=8) == expected[:8]
    assert stable_id("ID", {"a": 1}, length=8).startswith("ID-")
    assert normalize_text("  Alpha-Beta\t50 MG ") == "alpha-beta 50 mg"
    assert tokenize("A B Alpha 50 MG") == ["alpha", "50", "mg"]
    assert lexical_similarity("", "alpha") == 0
    assert lexical_similarity("Alpha", "alpha") == 1
    assert 0.9 < lexical_similarity("amoxicilin", "amoxicillin") < 1
    assert lexical_similarity("!", "?") == 0
    assert lexical_similarity("alpha 10 mg", "long alpha 10 mg tablet") == 1
    assert utc_now().tzinfo is UTC
