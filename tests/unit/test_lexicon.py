"""Local lexicon bundles: contract, integrity, provider, concordance, and OMOP import."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.lexicon_support import (
    ACRONYM,
    COLLISION,
    CONCORDANT,
    CONFLICT,
    DECLARATION,
    MESH_ONLY,
    REDISTRIBUTED,
    TYPO,
    build_bundle,
    concordance_profile,
    lexicon_pipeline,
    lexicon_profile,
    source_rows,
    write_sources,
)
from tests.support import (
    ALPHA,
    BRAND,
    COMBINATION,
    HISTORICAL,
    MISSPELLING,
    PRODUCT,
    SALT,
    make_pipeline,
)
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.enums import ConceptStatus, ConcordanceState, RetrievalChannel, RoutingStatus
from biosemalign.exceptions import (
    ConfigurationError,
    LexiconError,
    ProviderError,
    ResourceNotPermitted,
)
from biosemalign.knowledge.concordance import assess_concordance
from biosemalign.knowledge.lexicon.build import build_lexicon
from biosemalign.knowledge.lexicon.contract import LexiconManifest, bundle_digest
from biosemalign.knowledge.lexicon.omop import import_omop
from biosemalign.knowledge.providers.base import KnowledgeProvider
from biosemalign.knowledge.providers.lexicon import LexiconProvider, LexiconStore
from biosemalign.ops import ConcordanceFuseOp, LexiconRetrieveOp
from biosemalign.profiles.loader import load_profile
from biosemalign.schemas.knowledge import CandidateConcept, RetrievalEvidence
from biosemalign.util import stable_id


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, LexiconManifest]:
    root = tmp_path_factory.mktemp("lexicon")
    bundle, manifest = build_bundle(root)
    return root, bundle, manifest


@pytest.fixture
def store(built: tuple[Path, Path, LexiconManifest]) -> Iterator[LexiconStore]:
    _, bundle, manifest = built
    opened = LexiconStore(bundle, expected_sha256=manifest.bundle_sha256)
    yield opened
    opened.close()


@pytest.fixture
def pipeline(built: tuple[Path, Path, LexiconManifest], tmp_path: Path) -> Iterator[BioSemAlign]:
    _, bundle, manifest = built
    with lexicon_pipeline(tmp_path, bundle, manifest) as configured:
        yield configured


@pytest.fixture
def cascade(built: tuple[Path, Path, LexiconManifest], tmp_path: Path) -> Iterator[BioSemAlign]:
    _, bundle, manifest = built
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=concordance_profile()) as configured:
        yield configured


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    payload["artifact_id"] = stable_id(
        "ART",
        {
            name: payload[name]
            for name in ("run_id", "sau", "knowledge_package", "proposal", "validation", "decision")
        },
        length=32,
    )
    return payload


# -- build and contract ------------------------------------------------------


def test_bundle_manifest_pins_its_content(store: LexiconStore) -> None:
    manifest = store.manifest
    assert manifest.indexes == ["exact", "bm25", "trigram"]
    assert manifest.row_counts["concepts:RxNorm"] == 14
    assert manifest.bundle_sha256 == manifest.expected_bundle_sha256()
    assert manifest.vocabulary("rxnorm") is not None
    assert store.vocabulary_id("unii") == "UNII"
    assert store.release("MeSH") == "synthetic-mesh-v1"


def _with(table: str, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {table: rows}


def _concepts_plus(**row: Any) -> list[dict[str, Any]]:
    concepts = copy.deepcopy(source_rows()["concepts"])
    template = dict.fromkeys(concepts[0], "")
    concepts.append({**template, **row})
    return concepts


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            _with("concepts", [{**r, "extra": "x"} for r in source_rows()["concepts"]]),
            "unknown columns",
        ),
        (
            _with("synonyms", [{"vocabulary_id": "RxNorm", "concept_code": "1001"}]),
            "missing required columns",
        ),
        (
            _with(
                "synonyms",
                [{"vocabulary_id": "RxNorm", "concept_code": "nope", "synonym": "x", "source": ""}],
            ),
            "reference concepts absent",
        ),
        (
            _with(
                "concepts",
                _concepts_plus(vocabulary_id="Other", concept_code="1", concept_name="x"),
            ),
            "not declared",
        ),
        (
            _with(
                "concepts",
                _concepts_plus(
                    vocabulary_id="RxNorm", concept_code="X1", concept_name="x", status="alive"
                ),
            ),
            "unknown values",
        ),
        (
            _with(
                "concepts",
                _concepts_plus(
                    vocabulary_id="RxNorm",
                    concept_code="X1",
                    concept_name="x",
                    valid_end_date="31/12/2020",
                ),
            ),
            "not ISO dates",
        ),
        (
            _with(
                "concepts",
                _concepts_plus(vocabulary_id="RxNorm", concept_code="1001", concept_name="dup"),
            ),
            "duplicated",
        ),
        (
            _with(
                "concepts",
                _concepts_plus(
                    vocabulary_id="RxNorm", concept_code="X1", concept_name="x", replaced_by="nope"
                ),
            ),
            "replaced_by",
        ),
        (
            _with(
                "concepts",
                _concepts_plus(vocabulary_id="RxNorm", concept_code=" X1", concept_name="x"),
            ),
            "whitespace",
        ),
        ({"relationships": [{"vocabulary_id_1": "RxNorm"}]}, "missing required columns"),
    ],
)
def test_builder_lists_contract_violations(
    tmp_path: Path, overrides: dict[str, list[dict[str, Any]]], message: str
) -> None:
    with pytest.raises(LexiconError, match=message):
        build_bundle(tmp_path, **overrides)
    assert not (tmp_path / "bundle" / "manifest.json").exists()


def test_builder_requires_concepts_and_declared_vocabularies(tmp_path: Path) -> None:
    source = write_sources(tmp_path / "source", concepts=None)
    with pytest.raises(LexiconError, match="required table 'concepts'"):
        build_lexicon(source, tmp_path / "bundle", declaration=DECLARATION)

    source = write_sources(tmp_path / "source2")
    declaration = DECLARATION.model_copy(
        update={
            "vocabularies": [
                *DECLARATION.vocabularies,
                DECLARATION.vocabularies[0].model_copy(update={"vocabulary_id": "Empty"}),
            ]
        }
    )
    with pytest.raises(LexiconError, match="have no concepts"):
        build_lexicon(source, tmp_path / "bundle2", declaration=declaration)
    with pytest.raises(LexiconError, match="does not exist"):
        build_lexicon(tmp_path / "missing", tmp_path / "bundle3", declaration=DECLARATION)
    with pytest.raises(ValidationError, match="more than once"):
        DECLARATION.model_validate({"vocabularies": [{"vocabulary_id": "A", "release": "1"}] * 2})


def test_existing_bundle_is_only_replaced_on_request(tmp_path: Path) -> None:
    bundle, first = build_bundle(tmp_path)
    source = tmp_path / "source"
    with pytest.raises(LexiconError, match="already holds"):
        build_lexicon(source, bundle, declaration=DECLARATION)
    second = build_lexicon(source, bundle, declaration=DECLARATION, overwrite=True)
    assert second.row_counts == first.row_counts
    assert LexiconStore(bundle, expected_sha256=second.bundle_sha256).manifest == second


def test_parquet_and_tsv_sources_are_accepted(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = write_sources(tmp_path / "source")
    rows = list(csv.DictReader((source / "concepts.csv").open(encoding="utf-8")))
    pq.write_table(pa.Table.from_pylist(rows), source / "concepts.parquet")
    (source / "concepts.csv").unlink()
    synonyms = (source / "synonyms.csv").read_text(encoding="utf-8").replace(",", "\t")
    (source / "synonyms.tsv").write_text(synonyms, encoding="utf-8")
    (source / "synonyms.csv").unlink()
    manifest = build_lexicon(source, tmp_path / "bundle", declaration=DECLARATION)
    assert manifest.row_counts["concepts"] == 21
    assert set(manifest.inputs_sha256) == {
        "concepts.parquet",
        "synonyms.tsv",
        "relationships.csv",
        "xrefs.csv",
    }

    (source / "concepts.csv").write_text("vocabulary_id\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="supplied more than once"):
        build_lexicon(source, tmp_path / "bundle2", declaration=DECLARATION)


# -- integrity ---------------------------------------------------------------


def test_tampered_index_is_refused(tmp_path: Path) -> None:
    bundle, manifest = build_bundle(tmp_path)
    index = bundle / manifest.index_file
    data = bytearray(index.read_bytes())
    data[-1] ^= 0xFF
    index.write_bytes(bytes(data))
    with pytest.raises(LexiconError, match="modified or corrupted"):
        LexiconStore(bundle)
    index.unlink()
    with pytest.raises(LexiconError, match="is missing"):
        LexiconStore(bundle)


def test_edited_manifest_is_refused(tmp_path: Path) -> None:
    bundle, _ = build_bundle(tmp_path)
    path = bundle / "manifest.json"
    payload = json.loads(path.read_text())
    payload["vocabularies"][0]["release"] = "edited"
    path.write_text(json.dumps(payload))
    with pytest.raises(LexiconError, match="does not match its own contents"):
        LexiconStore(bundle)
    with pytest.raises(LexiconError, match="not a lexicon bundle"):
        LexiconStore(tmp_path / "nowhere")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("normalizer_version", "0", "Rebuild the bundle"),
        ("contract_version", "9.0", "Rebuild the bundle"),
    ],
)
def test_incompatible_bundles_require_a_rebuild(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    bundle, manifest = build_bundle(tmp_path)
    edited = manifest.model_copy(update={field: value})
    payload = edited.model_dump(mode="json")
    payload["bundle_sha256"] = bundle_digest(
        contract_version=edited.contract_version,
        normalizer_version=edited.normalizer_version,
        index_sha256=edited.index_sha256,
        vocabularies=edited.vocabularies,
        source_aliases=edited.source_aliases,
    )
    (bundle / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(LexiconError, match=message):
        LexiconStore(bundle)


def test_frozen_mode_requires_a_matching_pin(
    built: tuple[Path, Path, LexiconManifest], tmp_path: Path
) -> None:
    _, bundle, manifest = built
    with pytest.raises(ConfigurationError, match="requires the lexicon bundle to be pinned"):
        lexicon_pipeline(tmp_path, bundle, manifest, lexicon_sha256=None)
    with pytest.raises(LexiconError, match="does not match the pinned"):
        lexicon_pipeline(tmp_path, bundle, manifest, lexicon_sha256="0" * 64)


# -- store and provider -----------------------------------------------------


def test_exact_lookup_uses_the_mention_normalization(store: LexiconStore) -> None:
    hits = store.exact("  EXAMPLEALPHA \t", ["RxNorm", "UNII"])
    assert {(hit.vocabulary_id, hit.concept_code) for hit in hits} == {
        ("RxNorm", "1001"),
        ("UNII", "U-ALPHA"),
    }
    assert {hit.source for hit in hits if hit.vocabulary_id == "UNII"} == {"UNII"}
    assert store.exact("   ", ["RxNorm"]) == []
    assert store.bm25("!!", ["RxNorm"], 5) == []


def test_ranked_retrieval_finds_partial_and_misspelled_names(store: LexiconStore) -> None:
    partial = store.bm25("alpha sodium", ["RxNorm"], 3)
    assert partial[0].concept_code == "1101"
    misspelled = store.bm25(MISSPELLING, ["RxNorm"], 3, kind="trigram")
    assert misspelled[0].concept_code == "1001"
    assert all(hit.score > 0 for hit in misspelled)


def test_bundle_without_trigrams_rejects_trigram_retrieval(tmp_path: Path) -> None:
    source = write_sources(tmp_path / "source")
    manifest = build_lexicon(
        source, tmp_path / "bundle", declaration=DECLARATION, trigram_index=False
    )
    assert manifest.indexes == ["exact", "bm25"]
    store = LexiconStore(tmp_path / "bundle")
    with pytest.raises(LexiconError, match="without a 'trigram' index"):
        store.bm25("x", ["RxNorm"], 3, kind="trigram")
    provider = LexiconProvider(store, "RxNorm", mode=manifest_mode())
    assert provider.search("alpha sodium", top_k=2)[0].candidate_id == "1101"
    with (
        pytest.raises(ConfigurationError, match="no index for"),
        lexicon_pipeline(tmp_path, tmp_path / "bundle", manifest),
    ):
        pass
    store.close()


def manifest_mode() -> Any:
    from biosemalign.enums import ResourceMode

    return ResourceMode.FROZEN


def test_provider_implements_the_knowledge_provider_protocol(store: LexiconStore) -> None:
    provider = LexiconProvider(store, "rxnorm", mode=manifest_mode())
    assert isinstance(provider, KnowledgeProvider)
    assert provider.vocabulary == "RxNorm"
    assert provider.terminology_release() == "synthetic-rxnorm-v1"

    product = provider.lookup("2001")
    assert product is not None
    assert (product.concept_class, product.status) == ("SBD", ConceptStatus.ACTIVE)
    assert product.source_vocabularies == ["RxNorm"]
    remapped = provider.lookup("9001")
    assert remapped is not None and remapped.remapped_to == ["2001"]
    assert provider.lookup("missing") is None

    retired = provider.validate_concept("6001")
    assert retired.exists and retired.status is ConceptStatus.NOT_CURRENT
    assert not provider.validate_concept("missing").exists

    direct = provider.get_relationships("2001", predicates=["has_ingredient", "has_tradename"])
    assert {(s.predicate, s.object_id, s.extensions["target_tty"]) for s in direct} == {
        ("has_ingredient", "1001", "IN"),
        ("has_tradename", "3001", "BN"),
    }
    transitive = provider.get_relationships(
        "3001", predicates=["has_precise_ingredient", "has_ingredient"], max_hops=2
    )
    second_hop = [s for s in transitive if not s.direct]
    assert [(s.subject_id, s.object_id) for s in second_hop] == [("3001", "1001")]
    assert provider.available_predicates() >= {"has_ingredient", "has_dose_form"}

    with pytest.raises(ProviderError, match="serves"):
        provider.lookup("2001", version="another-release")
    with pytest.raises(ConfigurationError, match="does not serve"):
        LexiconProvider(store, "SNOMED", mode=manifest_mode())

    searched = provider.search("alpha sodium", top_k=2)
    assert searched[0].candidate_id == "1101"
    assert searched[0].channels == {RetrievalChannel.LEXICAL}
    assert provider.search(ALPHA, top_k=2)[0].channels == {RetrievalChannel.NORMALIZED_LOOKUP}
    assert provider.cache_hits > 0
    provider.reset_counters()
    assert provider.cache_hits == 0


@pytest.mark.parametrize("value", [ALPHA, PRODUCT, COMBINATION, SALT, BRAND, MISSPELLING])
def test_lexicon_served_rxnorm_matches_the_rxnav_outcome(
    value: str, pipeline: BioSemAlign, synthetic_pipeline: BioSemAlign
) -> None:
    local = pipeline.align_value(value)
    remote = synthetic_pipeline.align_value(value)
    assert local.decision is remote.decision
    assert local.relationship_to_source is remote.relationship_to_source
    # Combination components carry no order; RxNav lists them as its API does.
    assert sorted(c.concept_id for c in local.selected_concepts) == sorted(
        c.concept_id for c in remote.selected_concepts
    )
    assert local.routing_status is remote.routing_status
    assert {loss.kind for loss in local.semantic_loss} == {
        loss.kind for loss in remote.semantic_loss
    }
    [resource] = local.provenance.resources
    assert (resource.provider_name, resource.network_calls) == ("lexicon", 0)
    assert resource.content_sha256 == pipeline.knowledge.lexicon.bundle_sha256


def test_default_workflow_queries_the_lexicon_when_it_serves_the_target(
    pipeline: BioSemAlign,
) -> None:
    [tier] = pipeline.workflow.tiers
    assert [op.op_id for op in tier.retrievers] == ["lexicon"]
    assert pipeline.align_value(HISTORICAL).routing_status is RoutingStatus.HUMAN_REVIEW


def test_validity_dates_filter_when_requested(pipeline: BioSemAlign) -> None:
    op = LexiconRetrieveOp(
        pipeline.knowledge,
        pipeline.profile,
        vocabularies=["RxNorm"],
        methods=["exact"],
        respect_validity_dates=True,
    )
    assert op.options["respect_validity_dates"] is True

    def codes(**context: Any) -> list[str]:
        sau = pipeline.build_sau(pipeline.make_request(HISTORICAL, **context))[0]
        return [c.candidate_id for c in op.retrieve(sau).candidates]

    assert codes(date="2023-05-01") == []
    assert codes(date="20190101") == ["6001"]
    assert codes() == ["6001"]
    assert codes(date="not a date") == ["6001"]
    assert codes(date="2019-02-30") == ["6001"]


def test_rxnav_retriever_refuses_lexicon_served_rxnorm(
    built: tuple[Path, Path, LexiconManifest], tmp_path: Path
) -> None:
    _, bundle, manifest = built
    profile = lexicon_profile(workflow={"tiers": [{"name": "remote"}]})
    with pytest.raises(ConfigurationError, match="use the 'lexicon' retriever"):
        lexicon_pipeline(tmp_path, bundle, manifest, profile=profile)


def test_unserved_resources_fail_with_an_explanation(
    synthetic_pipeline: BioSemAlign, tmp_path: Path
) -> None:
    with pytest.raises(ResourceNotPermitted, match="neither served"):
        synthetic_pipeline.knowledge.provider_for("UNII")
    with pytest.raises(ResourceNotPermitted, match="served only from a local lexicon"):
        synthetic_pipeline.knowledge.provider_for("LocalLexicon")
    profile = lexicon_profile(secondary_resources=["UNII"])
    with pytest.raises(ConfigurationError, match="no workflow retriever queries them"):
        make_pipeline(tmp_path, profile=profile)
    with pytest.raises(ConfigurationError, match="needs a lexicon bundle"):
        LexiconRetrieveOp(synthetic_pipeline.knowledge, synthetic_pipeline.profile)


def test_lexicon_retriever_needs_a_served_vocabulary(pipeline: BioSemAlign) -> None:
    with pytest.raises(ConfigurationError, match="no vocabulary to query"):
        LexiconRetrieveOp(
            pipeline.knowledge,
            load_profile("faers_drug_normalization").model_copy(
                update={
                    "knowledge": pipeline.profile.knowledge.model_copy(
                        update={"primary_resources": []}
                    )
                }
            ),
        )


# -- concordance --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mention", "state", "reason"),
    [
        (CONCORDANT, ConcordanceState.MULTI_SOURCE_CONCORDANT, None),
        (REDISTRIBUTED, ConcordanceState.SINGLE_SOURCE_EXACT, "single_source"),
        (COLLISION, ConcordanceState.CROSS_VOCAB_COLLISION, "lexical_collision"),
        (CONFLICT, ConcordanceState.EQUIVALENCE_CONFLICT, "equivalence_conflict"),
        (ACRONYM, ConcordanceState.SINGLE_SOURCE_EXACT, "short_mention"),
        (MESH_ONLY, ConcordanceState.SINGLE_SOURCE_EXACT, "single_source"),
        (TYPO, ConcordanceState.NO_EXACT_MATCH, "no_exact_match"),
    ],
)
def test_concordance_outcomes_drive_the_cascade(
    cascade: BioSemAlign, mention: str, state: ConcordanceState, reason: str | None
) -> None:
    sau = cascade.build_sau(cascade.make_request(mention))[0]
    package = cascade.retrieve_knowledge(sau)
    assert package.concordance is not None
    assert package.concordance.state is state

    artifact = cascade.align_artifact(cascade.make_request(mention))
    trace = artifact.decision.provenance.resolution
    assert trace is not None
    if reason is None:
        assert trace.resolved_by_tier == "concordant"
    else:
        assert trace.attempts[0].reason == reason
        assert trace.resolved_by_tier == "fallback"
    assert verify_artifact(artifact).valid


def test_concordant_decisions_report_every_equivalent_code(cascade: BioSemAlign) -> None:
    artifact = cascade.align_artifact(cascade.make_request(CONCORDANT))
    decision = artifact.decision
    assert decision.concordance_state is ConcordanceState.MULTI_SOURCE_CONCORDANT
    assert [(m.vocabulary, m.concept_code) for m in decision.matched_codes] == [
        ("MeSH", "D-ALPHA"),
        ("RxNorm", "1001"),
        ("UMLS", "C-ALPHA"),
        ("UNII", "U-ALPHA"),
    ]
    group = artifact.knowledge_package.concordance.groups[0]
    # UMLS's atom came from MeSH and is counted once, under MeSH.
    assert group.sources == ["MeSH", "RxNorm", "UNII"]
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert artifact.knowledge_package.candidates[0].candidate_id == "1001"

    from biosemalign.storage.parquet import decision_to_row

    row = decision_to_row(decision)
    assert row["matched_codes"] == "MeSH:D-ALPHA|RxNorm:1001|UMLS:C-ALPHA|UNII:U-ALPHA"
    assert row["concordance_state"] == "MULTI_SOURCE_CONCORDANT"
    assert row["resolved_by_tier"] == "concordant"
    assert row["vocabulary_release"] == "synthetic-rxnorm-v1"

    forged = artifact.model_dump(mode="json")
    forged["decision"]["matched_codes"] = forged["decision"]["matched_codes"][:1]
    assert any("matched codes" in e for e in verify_artifact(_resign(forged)).errors)
    forged = artifact.model_dump(mode="json")
    forged["decision"]["concordance_state"] = "SINGLE_SOURCE_EXACT"
    assert any("concordance state" in e for e in verify_artifact(_resign(forged)).errors)


def test_redistributed_names_count_once(cascade: BioSemAlign) -> None:
    sau = cascade.build_sau(cascade.make_request(REDISTRIBUTED))[0]
    package = cascade.retrieve_knowledge(sau)
    [group] = package.concordance.groups
    assert [(m.vocabulary, m.concept_code) for m in group.members] == [
        ("RxNorm", "1002"),
        ("UMLS", "C-BETA"),
    ]
    assert group.sources == ["RxNorm"]


def test_authority_sources_accept_single_source_names(
    built: tuple[Path, Path, LexiconManifest], tmp_path: Path
) -> None:
    _, bundle, manifest = built
    profile = concordance_profile(authority_sources=["RxNorm", "MeSH"])
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=profile) as configured:
        trusted = configured.align_artifact(configured.make_request(REDISTRIBUTED))
        mesh_only = configured.align_artifact(configured.make_request(MESH_ONLY))
    assert trusted.decision.provenance.resolution.resolved_by_tier == "concordant"
    assert trusted.decision.selected_concepts[0].concept_id == "1002"
    assert mesh_only.decision.provenance.resolution.attempts[0].reason == "no_target_concept"


def test_salt_base_conflict_is_never_auto_accepted(cascade: BioSemAlign) -> None:
    decision = cascade.align_value(CONFLICT)
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert any("exact name of 2 RxNorm concepts: 1001, 1101" in r for r in decision.routing_reasons)


def test_equivalence_keys_bound_what_counts_as_the_same_entity(cascade: BioSemAlign) -> None:
    sau = cascade.build_sau(cascade.make_request(CONCORDANT))[0]
    retriever = cascade.workflow.tiers[0].retrievers[0]
    candidates = retriever.retrieve(sau)
    narrow = ConcordanceFuseOp(cascade.knowledge, cascade.profile, equivalence_keys=["RxNorm"])
    package = narrow.fuse(sau, [candidates])
    # MeSH:D-ALPHA links to the others only through a MeSH key, now excluded.
    assert package.concordance.state is ConcordanceState.CROSS_VOCAB_COLLISION
    assert narrow.options == {"equivalence_keys": ["RxNorm"]}


def _candidate(vocabulary: str, code: str, *, exact: bool = True) -> CandidateConcept:
    channel = RetrievalChannel.NORMALIZED_LOOKUP if exact else RetrievalChannel.LEXICAL
    return CandidateConcept(
        candidate_id=code,
        preferred_label=code,
        vocabulary=vocabulary,
        retrieval_evidence=[
            RetrievalEvidence(
                channel=channel, query="q", provider="rxnav", source_vocabulary="RXNORM"
            )
        ],
    )


def test_concordance_without_cross_references() -> None:
    aliases = {"RXNORM": "RxNorm"}
    single = assess_concordance(
        [_candidate("RxNorm", "1")], equivalence_keys=["RxNorm"], source_aliases=aliases
    )
    assert single.state is ConcordanceState.SINGLE_SOURCE_EXACT
    assert single.groups[0].sources == ["RxNorm"]
    homonym = assess_concordance(
        [_candidate("RxNorm", "1"), _candidate("RxNorm", "2")], equivalence_keys=["RxNorm"]
    )
    assert homonym.state is ConcordanceState.CROSS_VOCAB_COLLISION
    fuzzy = assess_concordance([_candidate("RxNorm", "1", exact=False)], equivalence_keys=[])
    assert fuzzy.state is ConcordanceState.NO_EXACT_MATCH and fuzzy.groups == []


def test_concordant_proposer_passes_without_an_assessment(pipeline: BioSemAlign) -> None:
    from biosemalign.adjudication.deterministic import DeterministicAdjudicator
    from biosemalign.ops import ConcordantExactProposeOp

    proposer = ConcordantExactProposeOp(
        DeterministicAdjudicator(pipeline.profile), pipeline.profile
    )
    sau = pipeline.build_sau(pipeline.make_request(ALPHA))[0]
    package = pipeline.retrieve_knowledge(sau)
    result = proposer.propose(sau, package)
    assert getattr(result, "reason", None) == "no_concordance_assessment"
    assert proposer.options["minimum_mention_length"] == 5


# -- OMOP import --------------------------------------------------------------


def _write_athena(directory: Path, tables: dict[str, list[list[str]]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (directory / f"{name}.csv").write_text(
            "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
        )
    return directory


ATHENA: dict[str, list[list[str]]] = {
    "VOCABULARY": [
        [
            "vocabulary_id",
            "vocabulary_name",
            "vocabulary_reference",
            "vocabulary_version",
            "vocabulary_concept_id",
        ],
        ["RxNorm", "Invented RxNorm", "x", "synthetic-2026", "0"],
        ["NDC", "Invented NDC", "x", "synthetic-ndc", "0"],
        ["SNOMED", "Invented SNOMED", "x", "synthetic-snomed", "0"],
    ],
    "CONCEPT": [
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ],
        [
            "1",
            "ExampleAlpha",
            "Drug",
            "RxNorm",
            "Ingredient",
            "S",
            "1001",
            "19700101",
            "20991231",
            "",
        ],
        [
            "2",
            "ExampleAlpha 10 MG Oral Tablet",
            "Drug",
            "RxNorm",
            "Clinical Drug",
            "S",
            "2001",
            "19700101",
            "20991231",
            "",
        ],
        ["3", "Brandix", "Drug", "RxNorm", "Brand Name", "", "3001", "19700101", "20991231", ""],
        [
            "4",
            "Brandix 10 MG Oral Tablet",
            "Drug",
            "RxNorm",
            "Branded Drug",
            "S",
            "2002",
            "19700101",
            "20991231",
            "",
        ],
        [
            "5",
            'Old "quoted" product',
            "Drug",
            "RxNorm",
            "Clinical Drug",
            "",
            "9001",
            "19700101",
            "20100101",
            "U",
        ],
        [
            "6",
            "Withdrawn product",
            "Drug",
            "RxNorm",
            "Clinical Drug",
            "",
            "9002",
            "19700101",
            "20100101",
            "D",
        ],
        ["7", "Oral Tablet", "Drug", "RxNorm", "Dose Form", "", "4001", "19700101", "20991231", ""],
        [
            "8",
            "00000000001",
            "Drug",
            "NDC",
            "11-digit NDC",
            "",
            "00000000001",
            "20200101",
            "20250101",
            "",
        ],
        [
            "9",
            "Invented finding",
            "Condition",
            "SNOMED",
            "Clinical Finding",
            "S",
            "999",
            "19700101",
            "20991231",
            "",
        ],
    ],
    "CONCEPT_SYNONYM": [
        ["concept_id", "concept_synonym_name", "language_concept_id"],
        ["1", "Example alpha synonym", "4180186"],
        ["9", "snomed synonym", "4180186"],
    ],
    "CONCEPT_RELATIONSHIP": [
        [
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ],
        ["4", "3", "Has tradename", "19700101", "20991231", ""],
        ["2", "7", "RxNorm has dose form", "19700101", "20991231", ""],
        ["5", "2", "Concept replaced by", "19700101", "20991231", ""],
        ["8", "2", "Maps to", "19700101", "20991231", ""],
        ["2", "4", "Invented unmapped", "19700101", "20991231", ""],
        ["4", "7", "RxNorm has dose form", "19700101", "20100101", "D"],
        ["1", "9", "Has finding", "19700101", "20991231", ""],
    ],
    "CONCEPT_ANCESTOR": [
        [
            "ancestor_concept_id",
            "descendant_concept_id",
            "min_levels_of_separation",
            "max_levels_of_separation",
        ],
        ["1", "2", "1", "1"],
        ["1", "4", "2", "2"],
        ["2", "4", "1", "1"],
        ["1", "1", "0", "0"],
    ],
}


def test_omop_import_produces_buildable_contract_tables(tmp_path: Path) -> None:
    athena = _write_athena(tmp_path / "athena", ATHENA)
    report = import_omop(
        athena, tmp_path / "tables", vocabularies=["RxNorm", "NDC"], license="public"
    )
    assert report.vocabularies == ["NDC", "RxNorm"]
    assert report.row_counts["concepts"] == 8
    assert report.unmapped_relationships == {
        "Concept replaced by": 1,
        "Invented unmapped": 1,
        "Maps to": 1,
    }

    from biosemalign.knowledge.lexicon.build import load_declaration

    declaration = load_declaration(tmp_path / "tables" / "vocabularies.yaml")
    assert {(v.vocabulary_id, v.release, v.license) for v in declaration.vocabularies} == {
        ("NDC", "synthetic-ndc", "public"),
        ("RxNorm", "synthetic-2026", "public"),
    }
    manifest = build_lexicon(tmp_path / "tables", tmp_path / "bundle", declaration=declaration)
    store = LexiconStore(tmp_path / "bundle", expected_sha256=manifest.bundle_sha256)
    rxnorm = LexiconProvider(store, "RxNorm", mode=manifest_mode())

    classes = {
        code: rxnorm.lookup(code).concept_class for code in ("1001", "2001", "2002", "3001", "4001")
    }
    assert classes == {"1001": "IN", "2001": "SCD", "2002": "SBD", "3001": "BN", "4001": "DF"}
    upgraded = rxnorm.lookup("9001")
    assert upgraded.status is ConceptStatus.REMAPPED and upgraded.remapped_to == ["2001"]
    assert upgraded.preferred_label == 'Old "quoted" product'
    assert rxnorm.lookup("9002").status is ConceptStatus.OBSOLETE
    assert "Example alpha synonym" in rxnorm.lookup("1001").synonyms

    edges = {
        (s.subject_id, s.predicate, s.object_id)
        for code in ("2001", "2002")
        for s in rxnorm.get_relationships(
            code, predicates=["has_ingredient", "has_tradename", "has_dose_form"]
        )
    }
    assert edges == {
        ("2001", "has_ingredient", "1001"),
        ("2001", "has_dose_form", "4001"),
        ("2002", "has_ingredient", "1001"),
        ("2002", "has_tradename", "3001"),
    }
    ndc = LexiconProvider(store, "NDC", mode=manifest_mode())
    assert ndc.lookup("00000000001").extensions["valid_end_date"] == "2025-01-01"
    keys = store.keys([("NDC", "00000000001"), ("RxNorm", "2001")])
    assert ("OMOP_STANDARD", "2") in keys[("NDC", "00000000001")]
    assert ("OMOP_STANDARD", "2") in keys[("RxNorm", "2001")]
    store.close()


def test_omop_import_reports_missing_inputs(tmp_path: Path) -> None:
    athena = _write_athena(tmp_path / "athena", ATHENA)
    with pytest.raises(LexiconError, match=r"not in VOCABULARY\.csv"):
        import_omop(athena, tmp_path / "out", vocabularies=["ATC"])
    with pytest.raises(LexiconError, match="at least one vocabulary"):
        import_omop(athena, tmp_path / "out", vocabularies=[])
    minimal = _write_athena(
        tmp_path / "minimal", {"VOCABULARY": ATHENA["VOCABULARY"], "CONCEPT": ATHENA["CONCEPT"]}
    )
    report = import_omop(
        minimal, tmp_path / "out2", vocabularies=["RxNorm"], derive_ingredients=False
    )
    assert report.row_counts["synonyms"] == 0 and report.row_counts["relationships"] == 0
    (minimal / "CONCEPT.csv").unlink()
    with pytest.raises(LexiconError, match=r"CONCEPT\.csv is missing"):
        import_omop(minimal, tmp_path / "out3", vocabularies=["RxNorm"])
    broken = _write_athena(
        tmp_path / "broken",
        {"VOCABULARY": ATHENA["VOCABULARY"], "CONCEPT": [["concept_id"], ["1"]]},
    )
    with pytest.raises(LexiconError, match="missing columns"):
        import_omop(broken, tmp_path / "out4", vocabularies=["RxNorm"])


# -- command line ------------------------------------------------------------


def test_lexicon_cli_builds_inspects_and_imports(tmp_path: Path) -> None:
    runner = CliRunner()
    source = write_sources(tmp_path / "source")
    declaration = tmp_path / "vocabularies.json"
    declaration.write_text(DECLARATION.model_dump_json())

    built = runner.invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(source),
            "--declaration",
            str(declaration),
            "--output",
            str(tmp_path / "bundle"),
        ],
    )
    assert built.exit_code == 0, built.output
    assert "BIOSEMALIGN_LEXICON_SHA256=" in built.output
    digest = built.output.split("bundle sha256: ")[1].split()[0]

    info = runner.invoke(app, ["lexicon", "info", str(tmp_path / "bundle"), "--sha256", digest])
    assert info.exit_code == 0, info.output
    assert (
        "integrity verified" in info.output and "UMLS synthetic-umls-v1 [restricted]" in info.output
    )
    wrong = runner.invoke(app, ["lexicon", "info", str(tmp_path / "bundle"), "--sha256", "0" * 64])
    assert wrong.exit_code == 2

    again = runner.invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(source),
            "--declaration",
            str(declaration),
            "--output",
            str(tmp_path / "bundle"),
        ],
    )
    assert again.exit_code == 2 and "already holds" in again.output

    athena = _write_athena(tmp_path / "athena", ATHENA)
    mapping = tmp_path / "map.yaml"
    mapping.write_text("Has tradename: has_tradename\n")
    imported = runner.invoke(
        app,
        [
            "lexicon",
            "import-omop",
            "--athena",
            str(athena),
            "--output",
            str(tmp_path / "tables"),
            "--vocabulary",
            "RxNorm",
            "--relationship-map",
            str(mapping),
            "--no-ancestors",
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert "RxNorm has dose form: 1" in imported.output
    assert "next: biosemalign lexicon build" in imported.output
    failed = runner.invoke(
        app,
        [
            "lexicon",
            "import-omop",
            "--athena",
            str(athena),
            "--output",
            str(tmp_path / "t2"),
            "--vocabulary",
            "ATC",
        ],
    )
    assert failed.exit_code == 2


def test_names_are_indexed_across_every_streaming_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A regression guard: inserting on the connection that owned the streaming
    # read once ended the stream after its first batch, silently dropping names.
    import biosemalign.knowledge.lexicon.build as builder

    monkeypatch.setattr(builder, "_BATCH_ROWS", 2)
    bundle, manifest = build_bundle(tmp_path)
    reference = build_lexicon(tmp_path / "source", tmp_path / "reference", declaration=DECLARATION)
    assert manifest.row_counts == reference.row_counts
    store = LexiconStore(bundle)
    assert [hit.concept_code for hit in store.exact("Meshonlyname", ["MeSH"])] == ["D-ONLY"]
    store.close()


def test_ranked_retrieval_spends_a_bounded_postings_budget(
    store: LexiconStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biosemalign.knowledge.providers.lexicon as provider_module

    monkeypatch.setattr(provider_module, "_POSTINGS_BUDGET", 1)
    # Only the rarest gram is read: the query is answered, if less precisely.
    bounded = store.bm25(MISSPELLING, ["RxNorm"], 5, kind="trigram")
    assert bounded and len(bounded) <= 5
    assert store.bm25("zzzz qqqq", ["RxNorm"], 5) == []
    assert store.bm25(ALPHA, ["UNII"], 5)[0].concept_code == "U-ALPHA"


def test_restricted_enclaves_may_use_the_lexicon_but_not_rxnav(
    built: tuple[Path, Path, LexiconManifest], tmp_path: Path
) -> None:
    from tests.support import seed_synthetic_cache, synthetic_settings

    _, bundle, manifest = built
    with lexicon_pipeline(
        tmp_path / "enclave", bundle, manifest, deployment_environment="restricted_enclave"
    ) as enclave:
        assert enclave.align_value(ALPHA).selected_concepts[0].concept_id == "1001"

    cache_dir = tmp_path / "remote"
    with pytest.raises(ResourceNotPermitted, match="restricted_enclave"):
        BioSemAlign(
            load_profile("faers_drug_normalization"),
            settings=synthetic_settings(cache_dir, deployment_environment="restricted_enclave"),
            cache=seed_synthetic_cache(cache_dir),
        )


def test_a_caller_supplied_store_outlives_the_router(store: LexiconStore, tmp_path: Path) -> None:
    from tests.support import synthetic_settings

    from biosemalign.knowledge.router import KnowledgeRouter
    from biosemalign.storage.cache import ResponseCache

    settings = synthetic_settings(tmp_path)
    router = KnowledgeRouter(
        lexicon_profile(),
        settings=settings,
        cache=ResponseCache(tmp_path, mode=settings.mode),
        lexicon=store,
    )
    provider = router.primary_provider()
    assert isinstance(provider, LexiconProvider)
    assert router.provider_for("RxNorm") is provider
    assert router.resource_provenance() == []
    router.close()
    assert store.exact(ALPHA, ["RxNorm"])


def test_unreadable_declarations_are_reported_cleanly(tmp_path: Path) -> None:
    from biosemalign.knowledge.lexicon.build import load_declaration

    with pytest.raises(LexiconError, match="cannot read lexicon declaration"):
        load_declaration(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("vocabularies: []\n")
    with pytest.raises(LexiconError, match="cannot read lexicon declaration"):
        load_declaration(bad)
    result = CliRunner().invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(tmp_path),
            "--declaration",
            str(tmp_path / "missing.yaml"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2 and "cannot read lexicon declaration" in result.output
