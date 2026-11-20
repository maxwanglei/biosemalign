"""Link derivation, family edges, Step 1 resolution to the target, and bulk coding."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest
from pydantic import ValidationError
from tests.lexicon_support import (
    COLLISION,
    CONCORDANT,
    MESH_ONLY,
    TYPO,
    build_bundle,
    concordance_profile,
    lexicon_pipeline,
)
from tests.links_support import (
    write_athena,
    write_csv,
    write_dictionary,
    write_mesh,
    write_unii_zip,
)
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.coding import code_terms, term_key
from biosemalign.enums import (
    ConcordanceState,
    DecisionType,
    RelationshipType,
    RetrievalChannel,
    SemanticLossKind,
)
from biosemalign.exceptions import LexiconError
from biosemalign.knowledge.concordance import assess_concordance
from biosemalign.knowledge.lexicon.audit import audit_bundle
from biosemalign.knowledge.lexicon.build import build_lexicon, load_declaration
from biosemalign.knowledge.lexicon.contract import LexiconDeclaration, LexiconManifest
from biosemalign.knowledge.lexicon.links import LINK_OVERLAYS, LinkReport, derive_links
from biosemalign.knowledge.providers.lexicon import LexiconStore
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept, RetrievalEvidence
from biosemalign.storage.parquet import decision_to_row

# -- fixtures -------------------------------------------------------------------


@pytest.fixture(scope="module")
def linked(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The synthetic dictionary, every link source, the links, and both bundles."""
    root = tmp_path_factory.mktemp("links")
    base = write_dictionary(root / "dictionary")
    report = derive_links(
        base,
        root / "links",
        athena=write_athena(root / "athena"),
        unii=write_unii_zip(root / "gsrs" / "UNIIs.zip"),
        mesh=write_mesh(root / "mesh"),
        product_evidence=base / "product_evidence.csv",
    )
    declaration = load_declaration(root / "links" / "vocabularies.yaml")
    plain = build_lexicon(
        base, root / "plain", declaration=load_declaration(base / "vocabularies.yaml")
    )
    bundle = build_lexicon(
        [base, *(root / "links" / name for name in LINK_OVERLAYS)],
        root / "bundle",
        declaration=declaration,
        trigram_index=False,
    )
    return {
        "root": root,
        "base": base,
        "report": report,
        "bundle": (root / "bundle", bundle),
        "plain": (root / "plain", plain),
    }


def _rule(report: LinkReport, rule: str) -> Any:
    return next(result for result in report.results if result.rule == rule)


def _emitted(links: Path, overlay: str) -> list[tuple[Any, ...]]:
    table = "relationships" if overlay == "internal" else "xrefs"
    return duckdb.sql(
        f"SELECT * FROM '{links / overlay / (table + '.parquet')}' ORDER BY ALL"
    ).fetchall()


STEP1_VOCABULARIES = [
    "RxNorm",
    "RxNorm Extension",
    "UNII",
    "MeSH",
    "NCIt",
    "SNOMED",
    "ATC",
    "UMLS",
    "Drugs@FDA",
]


def step1_profile(
    *, conflict_vocabularies: list[str] | None = None, source_terminologies: list[str] = ()
) -> TaskProfile:
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["task"]["source_terminologies"] = list(source_terminologies)
    payload["knowledge"]["secondary_resources"] = [v for v in STEP1_VOCABULARIES if v != "RxNorm"]
    fuser: dict[str, Any] = {"op": "cross_vocab_concordance"}
    if conflict_vocabularies is not None:
        fuser["options"] = {"conflict_vocabularies": conflict_vocabularies}
    payload["workflow"] = {
        "tiers": [
            {
                "name": "step1",
                "retrievers": [
                    {
                        "op": "lexicon",
                        "options": {"vocabularies": STEP1_VOCABULARIES, "methods": ["exact"]},
                    }
                ],
                "fuser": fuser,
                "proposer": {
                    "op": "concordant_exact",
                    "options": {"authority_sources": ["RxNorm", "RxNorm Extension"]},
                },
            },
            {
                "name": "fallback",
                "retrievers": [
                    {
                        "op": "lexicon",
                        "options": {"vocabularies": ["RxNorm"], "methods": ["exact", "bm25"]},
                    }
                ],
            },
        ]
    }
    return TaskProfile.model_validate(payload)


@pytest.fixture
def step1(linked: dict[str, Any], tmp_path: Path) -> Iterator[BioSemAlign]:
    bundle, manifest = linked["bundle"]
    profile = step1_profile(
        conflict_vocabularies=["RxNorm", "RxNorm Extension"],
        source_terminologies=["RxNorm Extension"],
    )
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=profile) as configured:
        yield configured


# -- link rules -------------------------------------------------------------------


def test_internal_rules_link_salts_combinations_and_brands(linked: dict[str, Any]) -> None:
    report: LinkReport = linked["report"]
    r1, r2, r3 = (_rule(report, r) for r in ("R1", "R2", "R3"))
    assert (r1.considered, r1.linked) == (3, 3)
    assert (r2.linked, r2.dropped) == (1, {"part_unresolved": 1})
    assert r3.linked == 4
    assert r3.dropped == {
        "products_disagree": 1,
        "tradename_disagrees": 1,
        "no_products_with_ingredients": 1,
    }
    assert any("Mixbrand" in example for example in r3.examples["products_disagree"])

    edges = {
        (row[0], row[1], row[3], row[4]): row[5]
        for row in _emitted(linked["root"] / "links", "internal")
    }
    assert edges[("RxNorm", "20", "RxNorm", "10")] == "OMOP Maps to"
    assert edges[("RxNorm", "21", "RxNorm", "11")] == "OMOP Form of"
    assert edges[("RxNorm", "22", "RxNorm", "12")] == "OMOP Form of"
    assert ("RxNorm", "30", "RxNorm", "10") in edges and ("RxNorm", "30", "RxNorm", "11") in edges
    assert edges[("RxNorm", "40", "RxNorm", "10")].startswith("OMOP brand products (2 agree)")
    # A precise ingredient inside the brand's set corroborates it.
    assert {("RxNorm", "42", "RxNorm", "10"), ("RxNorm", "42", "RxNorm", "12")} <= set(edges)
    # An RxNorm Extension brand reaches an RxNorm ingredient across the family.
    assert ("RxNorm Extension", "OMOP900", "RxNorm", "10") in edges
    assert not any(key[1] in {"41", "43", "44"} for key in edges)


def test_external_rules_link_by_asserted_codes(linked: dict[str, Any]) -> None:
    report: LinkReport = linked["report"]
    x1, x2, x3, x4 = (_rule(report, r) for r in ("X1", "X2", "X3", "X4"))
    assert (x1.linked, x1.dropped) == (
        5,
        {"unii_not_in_dictionary": 1, "code_retired_without_replacement": 1},
    )
    assert x1.emitted == 10  # both directions
    assert x2.linked == 2 and x2.dropped == {"not_a_unii (CAS or EC number)": 1}
    assert x3.linked == 2
    assert (x4.linked, x4.dropped) == (2, {"code_not_in_dictionary": 1})

    keys = {row[:4]: row[4] for row in _emitted(linked["root"] / "links", "external_id")}
    assert keys[("UNII", "UNIIALPHA1", "RxNorm", "10")].startswith("FDA UNII Records")
    assert ("RxNorm", "10", "UNII", "UNIIALPHA1") in keys
    assert ("NCIt", "C1", "UNII", "UNIIALPHA1") in keys
    # A retired RXCUI is followed to the concept that replaced it.
    assert ("UNII", "UNIIRETIRE", "RxNorm", "10") in keys
    assert not any(k[:2] == ("UNII", "UNIIOBSOL1") for k in keys)
    # The preferred concept's registry number only; a CAS number is not a UNII.
    assert keys[("MeSH", "D000001", "UNII", "UNIIALPHA1")] == "MeSH registry number"
    assert ("MeSH", "C000005", "UNII", "UNIIBETAL1") in keys
    assert not any(k[:2] == ("MeSH", "D000002") for k in keys)
    assert keys[("SNOMED", "900001", "RxNorm", "10")] == "OMOP Maps to"
    assert keys[("ATC", "Z99ZZ01", "RxNorm", "10")] == "OMOP ATC - RxNorm"
    assert keys[("MeSH", "D000001", "UMLS", "CUI0001")].startswith("UMLS synthetic-umls hub")
    # Rows the dictionary already had are not repeated.
    assert ("UMLS", "CUI0001", "RxNorm", "10") not in keys


def test_product_listings_link_only_when_name_and_composition_agree(
    linked: dict[str, Any],
) -> None:
    x5 = _rule(linked["report"], "X5")
    assert x5.linked == 4
    assert x5.dropped == {"no_matching_brand": 1, "ingredients_unresolved": 1}
    keys = {row[:4]: row[4] for row in _emitted(linked["root"] / "links", "external_derived")}
    assert set(keys) == {
        ("Drugs@FDA", "000001-001", "RxNorm", "40"),
        # Its ingredient is named only through a UNII whose record has an RXCUI.
        ("Drugs@FDA", "000005-001", "RxNorm", "40"),
        # A generic product named for its precise ingredient.
        ("Drugs@FDA", "000002-001", "RxNorm", "20"),
        # The RxNorm ingredient wins over the UNII substance of the same name.
        ("Drugs@FDA", "000006-001", "RxNorm", "10"),
    }
    assert keys[("Drugs@FDA", "000002-001", "RxNorm", "20")].endswith("(generic)")


def test_link_declaration_adds_the_target_family(linked: dict[str, Any]) -> None:
    declaration = load_declaration(linked["root"] / "links" / "vocabularies.yaml")
    assert declaration.relationship_families == [["RxNorm", "RxNorm Extension"]]
    _, manifest = linked["bundle"]
    assert manifest.relationship_families == [["RxNorm", "RxNorm Extension"]]
    assert manifest.family("RxNorm Extension") == ("RxNorm Extension", "RxNorm")
    _, plain = linked["plain"]
    assert plain.relationship_families == []
    report = (linked["root"] / "links" / "link_report.md").read_text()
    assert "| R3 |" in report and "Athena releases match" in report


def test_rules_without_their_sources_are_reported_as_skipped(tmp_path: Path) -> None:
    base = write_dictionary(tmp_path / "dictionary")
    report = derive_links(base, tmp_path / "links")
    skipped = {r.rule: r.skipped for r in report.results if r.skipped}
    assert set(skipped) == {"R3", "X1", "X2", "X3", "X5"}
    # The dictionary's own OMOP Maps-to keys still link one precise ingredient.
    r1 = _rule(report, "R1")
    assert r1.linked == 1 and r1.dropped == {"no_ingredient_found": 2}
    with pytest.raises(LexiconError, match="already holds links"):
        derive_links(base, tmp_path / "links")


def test_athena_from_another_release_is_refused_unless_allowed(tmp_path: Path) -> None:
    base = write_dictionary(tmp_path / "dictionary")
    athena = write_athena(tmp_path / "athena", rxnorm_release="RxNorm other-release")
    with pytest.raises(LexiconError, match="Athena release differs"):
        derive_links(base, tmp_path / "links", athena=athena)
    report = derive_links(base, tmp_path / "links", athena=athena, allow_version_mismatch=True)
    assert any(note.startswith("WARNING: Athena release differs") for note in report.notes)


def test_malformed_unii_records_are_refused(tmp_path: Path) -> None:
    base = write_dictionary(tmp_path / "dictionary")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("UNII_Records_x.txt", "UNII\tDisplay Name\nU1\tX\n")
    with pytest.raises(LexiconError, match=r"lacks columns \['RXCUI', 'NCIT'\]"):
        derive_links(base, tmp_path / "links", unii=archive)
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as handle:
        handle.writestr("readme.txt", "nothing")
    with pytest.raises(LexiconError, match="one UNII_Records"):
        derive_links(base, tmp_path / "links2", unii=empty)


def test_product_evidence_needs_its_columns(tmp_path: Path) -> None:
    base = write_dictionary(tmp_path / "dictionary")
    evidence = tmp_path / "evidence.csv"
    write_csv(evidence, [{"vocabulary_id": "Drugs@FDA", "concept_code": "x"}])
    with pytest.raises(LexiconError, match="lacks columns"):
        derive_links(base, tmp_path / "links", product_evidence=evidence)


# -- contract and build ----------------------------------------------------------


def test_relationship_families_must_name_declared_vocabularies() -> None:
    vocabularies = [
        {"vocabulary_id": "RxNorm", "release": "1"},
        {"vocabulary_id": "RxNorm Extension", "release": "1"},
        {"vocabulary_id": "UNII", "release": "1"},
    ]
    for families, message in [
        ([["RxNorm"]], "at least two"),
        ([["RxNorm", "SNOMED"]], "undeclared"),
        ([["RxNorm", "RxNorm Extension"], ["RxNorm", "UNII"]], "more than one"),
    ]:
        with pytest.raises(ValidationError, match=message):
            LexiconDeclaration.model_validate(
                {"vocabularies": vocabularies, "relationship_families": families}
            )


def test_overlays_are_validated_against_the_combined_dictionary(tmp_path: Path) -> None:
    base = write_dictionary(tmp_path / "dictionary")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    write_csv(
        overlay / "relationships.csv",
        [
            {
                "vocabulary_id_1": "RxNorm",
                "concept_code_1": "40",
                "predicate": "has_ingredient",
                "vocabulary_id_2": "RxNorm",
                "concept_code_2": "MISSING",
                "assertion_source": "test",
            }
        ],
    )
    declaration = load_declaration(base / "vocabularies.yaml")
    with pytest.raises(LexiconError, match="reference concepts absent"):
        build_lexicon([base, overlay], tmp_path / "bundle", declaration=declaration)
    with pytest.raises(LexiconError, match="does not exist"):
        build_lexicon([base, tmp_path / "nowhere"], tmp_path / "bundle", declaration=declaration)
    with pytest.raises(LexiconError, match="at least one"):
        build_lexicon([], tmp_path / "bundle", declaration=declaration)


def test_combined_bundles_record_every_input_and_edge_source(linked: dict[str, Any]) -> None:
    path, manifest = linked["bundle"]
    assert "dictionary/concepts.csv" in manifest.inputs_sha256
    assert "internal/relationships.parquet" in manifest.inputs_sha256
    store = LexiconStore(path)
    try:
        edges = store.relationships("RxNorm Extension", ["OMOP900"], ["has_ingredient"])
        assert [(e.object_vocabulary, e.object_code) for e in edges] == [("RxNorm", "10")]
        assert edges[0].assertion_source.startswith("OMOP brand products")
        # Without the family, an RxNorm Extension edge into RxNorm is not served.
        assert "has_ingredient" in store.predicates("RxNorm Extension")
    finally:
        store.close()
    plain_path, _ = linked["plain"]
    plain = LexiconStore(plain_path)
    try:
        assert plain.relationships("RxNorm Extension", ["OMOP901"], ["has_ingredient"]) == []
    finally:
        plain.close()


# -- concordance and Step 1 ------------------------------------------------------


def _candidate(vocabulary: str, code: str) -> CandidateConcept:
    return CandidateConcept(
        candidate_id=code,
        preferred_label=code,
        vocabulary=vocabulary,
        retrieval_evidence=[
            RetrievalEvidence(
                channel=RetrievalChannel.NORMALIZED_LOOKUP,
                query="x",
                provider="lexicon",
                source_vocabulary=vocabulary,
            )
        ],
    )


def test_conflicts_can_be_limited_to_the_target_family() -> None:
    candidates = [
        _candidate("RxNorm", "1"),
        _candidate("SNOMED", "S-SUBSTANCE"),
        _candidate("SNOMED", "S-PRODUCT"),
    ]
    keys = {
        ("RxNorm", "1"): {("RxNorm", "1")},
        ("SNOMED", "S-SUBSTANCE"): {("RxNorm", "1")},
        ("SNOMED", "S-PRODUCT"): {("RxNorm", "1")},
    }
    strict = assess_concordance(
        candidates, equivalence_keys=["RxNorm"], key_lookup=lambda members: keys
    )
    assert strict.state is ConcordanceState.EQUIVALENCE_CONFLICT
    narrowed = assess_concordance(
        candidates,
        equivalence_keys=["RxNorm"],
        key_lookup=lambda members: keys,
        conflict_vocabularies=["RxNorm", "RxNorm Extension"],
    )
    assert narrowed.state is ConcordanceState.MULTI_SOURCE_CONCORDANT
    assert not narrowed.groups[0].conflict
    # RxNorm and RxNorm Extension are one pool: a code from each conflicts.
    pooled = assess_concordance(
        [_candidate("RxNorm", "1"), _candidate("RxNorm Extension", "E1")],
        equivalence_keys=["RxNorm"],
        key_lookup=lambda members: {m: {("RxNorm", "1")} for m in members},
        conflict_vocabularies=["RxNorm", "RxNorm Extension"],
    )
    assert pooled.state is ConcordanceState.EQUIVALENCE_CONFLICT


def _attempt(artifact: Any) -> Any:
    return artifact.decision.provenance.resolution.attempts[0]


def test_links_turn_collisions_into_resolved_ingredients(
    linked: dict[str, Any], step1: BioSemAlign, tmp_path: Path
) -> None:
    artifact = step1.align_artifact(step1.make_request("Alphamab"))
    decision = artifact.decision
    assert decision.provenance.resolution.resolved_by_tier == "step1"
    assert decision.relationship_to_source is RelationshipType.EXACT
    assert {(m.vocabulary, m.concept_code) for m in decision.matched_codes} == {
        ("ATC", "Z99ZZ01"),
        ("Drugs@FDA", "000006-001"),
        ("MeSH", "D000001"),
        ("NCIt", "C1"),
        ("RxNorm", "10"),
        ("SNOMED", "900001"),
        ("UMLS", "CUI0001"),
        ("UNII", "UNIIALPHA1"),
    }
    assert verify_artifact(artifact).valid

    # Without the links, the same name is a collision.
    path, manifest = linked["plain"]
    with lexicon_pipeline(tmp_path / "plain", path, manifest, profile=step1_profile()) as plain:
        unlinked = plain.align_artifact(plain.make_request("Alphamab"))
    assert _attempt(unlinked).reason == "lexical_collision"


def test_brands_salts_and_combinations_reach_their_ingredients(step1: BioSemAlign) -> None:
    brand = step1.align_artifact(step1.make_request("Alphabrand"))
    assert brand.decision.provenance.resolution.resolved_by_tier == "step1"
    assert brand.decision.source_level_concept.concept_id == "40"
    assert [c.concept_id for c in brand.decision.selected_concepts] == ["10"]
    assert brand.decision.relationship_to_source is RelationshipType.TARGET_BROADER_THAN_SOURCE
    assert SemanticLossKind.BRAND_IDENTITY in {loss.kind for loss in brand.decision.semantic_loss}
    # Two FDA listings of one brand are recorded, not a conflict.
    assert ("Drugs@FDA", "000005-001") in {
        (m.vocabulary, m.concept_code) for m in brand.decision.matched_codes
    }
    statements = brand.knowledge_package.candidate_by_qualified_id("RxNorm", "40").relationships
    assert statements[0].extensions["assertion_source"].startswith("OMOP brand products")

    salt = step1.align_artifact(step1.make_request("alphamab sodium"))
    assert [c.concept_id for c in salt.decision.selected_concepts] == ["10"]
    assert SemanticLossKind.SALT_FORM in {loss.kind for loss in salt.decision.semantic_loss}

    combination = step1.align_artifact(step1.make_request("alphamab / betaprofen"))
    assert combination.decision.decision is DecisionType.MULTI_CONCEPT_MAPPED
    assert sorted(c.concept_id for c in combination.decision.selected_concepts) == ["10", "11"]
    for artifact in (brand, salt, combination):
        assert verify_artifact(artifact).valid


def test_the_strict_conflict_rule_still_holds_by_default(
    linked: dict[str, Any], tmp_path: Path
) -> None:
    path, manifest = linked["bundle"]
    with lexicon_pipeline(tmp_path, path, manifest, profile=step1_profile()) as strict:
        artifact = strict.align_artifact(strict.make_request("Alphabrand"))
    assert _attempt(artifact).reason == "equivalence_conflict"


def test_an_identity_without_an_ingredient_link_passes_on(step1: BioSemAlign) -> None:
    artifact = step1.align_artifact(step1.make_request("Lonebrand"))
    attempt = _attempt(artifact)
    assert attempt.reason == "no_ingredient_link"
    assert "RxNorm:44 (BN, 'Lonebrand')" in attempt.detail
    assert artifact.decision.provenance.resolution.resolved_by_tier == "fallback"
    assert verify_artifact(artifact).valid


def test_rxnorm_extension_identities_must_reach_rxnorm_ingredients(
    linked: dict[str, Any], step1: BioSemAlign, tmp_path: Path
) -> None:
    artifact = step1.align_artifact(step1.make_request("Extbrand"))
    decision = artifact.decision
    assert decision.provenance.resolution.resolved_by_tier == "step1"
    assert decision.source_level_concept.vocabulary == "RxNorm Extension"
    assert [(c.vocabulary, c.concept_id) for c in decision.selected_concepts] == [("RxNorm", "10")]
    assert [(m.vocabulary, m.concept_code) for m in decision.matched_codes] == [
        ("RxNorm Extension", "OMOP900")
    ]
    assert verify_artifact(artifact).valid
    row = decision_to_row(decision)
    assert row["source_level_id"] == "OMOP900" and row["selected_ids"] == "10"

    foreign = step1.align_artifact(step1.make_request("Deltabrand"))
    assert _attempt(foreign).reason == "non_target_ingredient"
    assert "RxNorm Extension:OMOP902" in _attempt(foreign).detail

    # Unless the profile permits RxNorm Extension as a source, it is not an identity.
    path, manifest = linked["bundle"]
    profile = step1_profile(conflict_vocabularies=["RxNorm", "RxNorm Extension"])
    with lexicon_pipeline(tmp_path, path, manifest, profile=profile) as narrow:
        refused = narrow.align_artifact(narrow.make_request("Extbrand"))
    assert _attempt(refused).reason == "no_target_concept"


def test_a_source_concept_outside_the_permitted_vocabularies_is_rejected(
    step1: BioSemAlign,
) -> None:
    artifact = step1.align_artifact(step1.make_request("Extbrand"))
    forged = artifact.model_dump(mode="json")
    forged["sau"]["task"]["source_terminologies"] = []
    from tests.unit.test_lexicon import _resign

    assert not verify_artifact(_resign(forged)).valid


def test_every_tier_keeps_the_codes_it_matched(step1: BioSemAlign) -> None:
    artifact = step1.align_artifact(step1.make_request("Mixbrand"))
    attempt = _attempt(artifact)
    assert attempt.reason == "lexical_collision"
    assert attempt.concordance_state is ConcordanceState.CROSS_VOCAB_COLLISION
    assert [[(m.vocabulary, m.concept_code) for m in g.members] for g in attempt.exact_matches] == [
        [("Drugs@FDA", "000003-001")],
        [("RxNorm", "41")],
    ]
    # The fallback tier's package replaced Step 1's, but the codes survive.
    assert artifact.decision.provenance.resolution.resolved_by_tier == "fallback"
    row = decision_to_row(artifact.decision)
    assert row["exact_match_tier"] == "step1"
    assert row["exact_match_state"] == "CROSS_VOCAB_COLLISION"
    assert row["exact_matches"] == "Drugs@FDA:000003-001;RxNorm:41"
    assert verify_artifact(artifact).valid


def test_codes_of_names_with_no_target_concept_are_kept(
    built_lexicon: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = built_lexicon
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=concordance_profile()) as cascade:
        artifact = cascade.align_artifact(cascade.make_request(MESH_ONLY))
    attempt = _attempt(artifact)
    assert attempt.reason == "single_source"
    assert [(m.vocabulary, m.concept_code) for m in attempt.exact_matches[0].members] == [
        ("MeSH", "D-ONLY")
    ]
    assert decision_to_row(artifact.decision)["exact_matches"] == "MeSH:D-ONLY"


def test_recorded_coding_must_match_the_artifacts_package(
    built_lexicon: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = built_lexicon
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=concordance_profile()) as cascade:
        artifact = cascade.align_artifact(cascade.make_request(CONCORDANT))
    from tests.unit.test_lexicon import _resign

    forged = artifact.model_dump(mode="json")
    forged["decision"]["provenance"]["resolution"]["attempts"][-1]["exact_matches"] = []
    assert any("exact matches" in e for e in verify_artifact(_resign(forged)).errors)


@pytest.fixture(scope="module")
def built_lexicon(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    return build_bundle(tmp_path_factory.mktemp("lexicon"))


# -- coding ------------------------------------------------------------------------


def test_coding_keeps_every_code_of_every_term(linked: dict[str, Any], tmp_path: Path) -> None:
    path, manifest = linked["bundle"]
    terms = tmp_path / "terms.csv"
    write_csv(
        terms,
        [
            {"report": "r1", "drugname": "Alphabrand"},
            {"report": "r2", "drugname": "  ALPHABRAND "},
            {"report": "r3", "drugname": "Mixbrand"},
            {"report": "r4", "drugname": "nothing like a drug"},
            {"report": "r5", "drugname": ""},
            {"report": "r6", "drugname": "Alphamab"},
        ],
    )
    out = tmp_path / "coded"
    report = code_terms(path, terms, out, text_column="drugname", id_columns=["report"])
    assert report.bundle_sha256 == manifest.bundle_sha256
    assert (report.rows, report.blank_rows, report.terms) == (6, 1, 4)
    assert (report.coded_terms, report.coded_rows) == (3, 4)
    assert report.terms_reaching_target == 2

    def read(name: str) -> list[dict[str, Any]]:
        relation = duckdb.sql(f"SELECT * FROM '{out / name}'")
        return [dict(zip(relation.columns, row, strict=True)) for row in relation.fetchall()]

    occurrences = read("occurrences.parquet")
    assert [o["report"] for o in occurrences] == ["r1", "r2", "r3", "r4", "r5", "r6"]
    assert occurrences[0]["term_key"] == occurrences[1]["term_key"] == term_key("alphabrand")
    assert occurrences[4]["term_key"] is None

    terms_out = {t["normalized"]: t for t in read("terms.parquet")}
    assert terms_out["alphabrand"]["occurrences"] == 2
    assert terms_out["alphabrand"]["target_ingredients"] == ["RxNorm:10"]
    assert terms_out["alphabrand"]["ingredient_sets"] == 1
    assert terms_out["mixbrand"]["target_ingredients"] == []
    assert terms_out["mixbrand"]["vocabularies"] == ["Drugs@FDA", "RxNorm"]

    codes = {
        (c["normalized"], c["vocabulary_id"], c["concept_code"]): c for c in read("codes.parquet")
    }
    fda = codes[("alphabrand", "Drugs@FDA", "000001-001")]
    assert fda["target_link"] == "key" and fda["linked_concepts"] == ["RxNorm:40"]
    assert fda["target_ingredients"] == ["RxNorm:10"]
    assert codes[("alphabrand", "RxNorm", "40")]["target_link"] == "self"
    assert codes[("alphabrand", "RxNorm", "40")]["vocabulary_release"] == "RxNorm synthetic-1"
    # A code with no link is kept, annotated as reaching nothing.
    assert codes[("mixbrand", "Drugs@FDA", "000003-001")]["target_link"] == "none"
    assert codes[("alphamab", "MeSH", "D000001")]["target_ingredients"] == ["RxNorm:10"]
    assert [u["normalized"] for u in read("uncoded.parquet")] == ["nothing like a drug"]

    manifest_json = json.loads((out / "coding_manifest.json").read_text())
    assert manifest_json["counts"]["coded_terms"] == 3
    assert "| Coded (≥1 exact code) | 3 (75.0%)" in (out / "coding_report.md").read_text()


def test_coding_matches_the_step1_tier_exactly(
    step1: BioSemAlign, linked: dict[str, Any], tmp_path: Path
) -> None:
    path, _ = linked["bundle"]
    terms = tmp_path / "terms.csv"
    names = ["Alphamab", "Alphabrand", "Mixbrand", "Extbrand", "alphamab sodium"]
    write_csv(terms, [{"drugname": name} for name in names])
    out = tmp_path / "coded"
    code_terms(path, terms, out, text_column="drugname", vocabularies=STEP1_VOCABULARIES)
    coded: dict[str, set[tuple[str, str]]] = {}
    for normalized, vocabulary, code in duckdb.sql(
        f"SELECT normalized, vocabulary_id, concept_code FROM '{out / 'codes.parquet'}'"
    ).fetchall():
        coded.setdefault(normalized, set()).add((vocabulary, code))
    for name in names:
        sau = step1.build_sau(step1.make_request(name))[0]
        package = step1.retrieve_knowledge(sau)
        matched = {
            (m.vocabulary, m.concept_code)
            for group in package.concordance.groups
            for m in group.members
        }
        assert matched == coded[sau.mention.normalized_text], name


def test_coding_refuses_bad_requests(linked: dict[str, Any], tmp_path: Path) -> None:
    path, manifest = linked["bundle"]
    terms = tmp_path / "terms.csv"
    write_csv(terms, [{"drugname": "Alphabrand"}])
    with pytest.raises(LexiconError, match="lacks columns"):
        code_terms(path, terms, tmp_path / "a", text_column="name")
    with pytest.raises(LexiconError, match="does not serve 'Nope'"):
        code_terms(path, terms, tmp_path / "b", text_column="drugname", vocabularies=["Nope"])
    with pytest.raises(LexiconError, match="does not exist"):
        code_terms(path, tmp_path / "missing.csv", tmp_path / "c", text_column="drugname")
    with pytest.raises(LexiconError, match="must be"):
        code_terms(path, path / "manifest.json", tmp_path / "d", text_column="drugname")
    with pytest.raises(LexiconError, match="does not match the pinned"):
        code_terms(path, terms, tmp_path / "e", text_column="drugname", expected_sha256="0" * 64)
    code_terms(path, terms, tmp_path / "f", text_column="drugname")
    with pytest.raises(LexiconError, match="already holds coding output"):
        code_terms(path, terms, tmp_path / "f", text_column="drugname")
    parquet = tmp_path / "terms.parquet"
    duckdb.sql(f"COPY (SELECT 'Alphabrand' AS drugname) TO '{parquet}' (FORMAT PARQUET)")
    report = code_terms(path, parquet, tmp_path / "f", text_column="drugname", overwrite=True)
    assert report.coded_terms == 1 and report.bundle_sha256 == manifest.bundle_sha256


def test_audit_measures_link_completeness(linked: dict[str, Any], tmp_path: Path) -> None:
    path, _ = linked["bundle"]
    report = audit_bundle(path, tmp_path / "audit", write_crosswalk=True)
    assert report.family == ["RxNorm", "RxNorm Extension"]
    assert report.family_classes["RxNorm|BN"] == {"active": 5, "reach": 2, "target": 2}
    assert report.family_classes["RxNorm|PIN"] == {"active": 3, "reach": 3, "target": 3}
    assert report.family_classes["RxNorm Extension|BN"] == {"active": 2, "reach": 2, "target": 1}
    assert report.vocabularies["UNII"]["linked"] == 4
    assert report.vocabularies["MeSH"]["reach"] == 2
    assert (tmp_path / "audit" / "code_to_ingredient.parquet").is_file()
    assert "| RxNorm | BN | 5 | 2 (40.0%)" in (tmp_path / "audit" / "audit.md").read_text()

    plain_path, _ = linked["plain"]
    before = audit_bundle(plain_path, tmp_path / "before")
    assert before.family == ["RxNorm"]
    assert before.family_classes["RxNorm|BN"]["reach"] == 0
    with pytest.raises(LexiconError, match="does not serve the target"):
        audit_bundle(plain_path, tmp_path / "x", target="Nope")


# -- CLI -------------------------------------------------------------------------


def test_cli_links_builds_codes_and_audits(linked: dict[str, Any], tmp_path: Path) -> None:
    runner = CliRunner()
    base = write_dictionary(tmp_path / "dictionary")
    links = tmp_path / "links"
    result = runner.invoke(
        app,
        [
            "lexicon",
            "link",
            "--base",
            str(base),
            "--output",
            str(links),
            "--athena",
            str(write_athena(tmp_path / "athena")),
            "--unii",
            str(write_unii_zip(tmp_path / "UNIIs.zip")),
            *[arg for path in write_mesh(tmp_path / "mesh") for arg in ("--mesh", str(path))],
            "--product-evidence",
            str(base / "product_evidence.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "R3 Brand name (BN)" in result.output and "next: biosemalign lexicon build" in result.output
    )
    again = runner.invoke(app, ["lexicon", "link", "--base", str(base), "--output", str(links)])
    assert again.exit_code == 2 and "already holds links" in again.output

    bundle = tmp_path / "bundle"
    sources = [arg for name in LINK_OVERLAYS for arg in ("--source", str(links / name))]
    result = runner.invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(base),
            *sources,
            "--declaration",
            str(links / "vocabularies.yaml"),
            "--output",
            str(bundle),
            "--no-trigram",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((bundle / "manifest.json").read_text())["indexes"] == ["exact", "bm25"]

    terms = tmp_path / "terms.tsv"
    terms.write_text("id\tdrugname\n1\tAlphabrand\n2\tunknown thing\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "code",
            "--bundle",
            str(bundle),
            "--input",
            str(terms),
            "--text-column",
            "drugname",
            "--id-column",
            "id",
            "--output",
            str(tmp_path / "coded"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "coded 1/2 distinct terms" in result.output
    failed = runner.invoke(
        app,
        [
            "code",
            "--bundle",
            str(bundle),
            "--input",
            str(terms),
            "--text-column",
            "x",
            "--output",
            str(tmp_path / "x"),
        ],
    )
    assert failed.exit_code == 2

    result = runner.invoke(app, ["lexicon", "audit", str(bundle), "--output", str(tmp_path / "a")])
    assert result.exit_code == 0, result.output
    assert "RxNorm: " in result.output
    failed = runner.invoke(
        app, ["lexicon", "audit", str(bundle), "--output", str(tmp_path / "b"), "--target", "Nope"]
    )
    assert failed.exit_code == 2


def test_typo_and_collision_names_keep_their_codes_in_the_lexicon_cascade(
    built_lexicon: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = built_lexicon
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=concordance_profile()) as cascade:
        typo = cascade.align_artifact(cascade.make_request(TYPO))
        collision = cascade.align_artifact(cascade.make_request(COLLISION))
        concordant = cascade.align_artifact(cascade.make_request(CONCORDANT))
    assert _attempt(typo).concordance_state is ConcordanceState.NO_EXACT_MATCH
    assert _attempt(typo).exact_matches == ()
    assert len(_attempt(collision).exact_matches) == 2
    # A resolved tier records its coding too, matching the artifact's package.
    assert (
        _attempt(concordant).exact_matches[0] == concordant.knowledge_package.concordance.groups[0]
    )


def test_omop_import_declares_the_rxnorm_family(tmp_path: Path) -> None:
    from biosemalign.knowledge.lexicon.omop import import_omop

    athena = write_athena(tmp_path / "athena")
    import_omop(athena, tmp_path / "both", vocabularies=["RxNorm", "RxNorm Extension"])
    both = load_declaration(tmp_path / "both" / "vocabularies.yaml")
    assert both.relationship_families == [["RxNorm", "RxNorm Extension"]]
    import_omop(athena, tmp_path / "one", vocabularies=["RxNorm"])
    assert load_declaration(tmp_path / "one" / "vocabularies.yaml").relationship_families == []
