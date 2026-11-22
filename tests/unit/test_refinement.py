"""WP4 unit tests: always-derived LinkSets, incoming edges, refinement, and G2 after expand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from tests.scenarios_support import (
    appendix_a_drug_profile,
    appendix_a_pipeline,
    build_appendix_a_bundle,
)
from tests.support import synthetic_settings
from typer.testing import CliRunner

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.cli import app
from biosemalign.enums import DecisionType, RoutingStatus
from biosemalign.kernel import Cascade, Tier
from biosemalign.knowledge.providers.lexicon import LexiconStore
from biosemalign.modules import OpContext, available, build
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet, ExpansionStep
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.storage.cache import ResponseCache


def test_expand_registry_lists_refine_by_slots() -> None:
    assert "refine_by_slots" in available("expand")


def test_lexicon_store_incoming_relationships_and_family_caching(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "concepts.csv").write_text(
        "\n".join(
            [
                "vocabulary_id,concept_code,concept_name,concept_class,status",
                "RxNorm,B1,BrandOne,BN,active",
                "RxNorm,P1,BrandOne 50 MG Oral Tablet,SBD,active",
                "RxNorm Extension,P2,BrandOne 50 MG Capsule,SBD,active",
                "OtherVocab,X1,ForeignThing,SBD,active",
            ]
        ),
        encoding="utf-8",
    )
    (src / "synonyms.csv").write_text(
        "vocabulary_id,concept_code,synonym\nRxNorm,B1,BrandOne\n", encoding="utf-8"
    )
    (src / "relationships.csv").write_text(
        "\n".join(
            [
                "vocabulary_id_1,concept_code_1,predicate,vocabulary_id_2,concept_code_2",
                "RxNorm,P1,has_tradename,RxNorm,B1",
                "RxNorm Extension,P2,has_tradename,RxNorm,B1",
                "OtherVocab,X1,has_tradename,RxNorm,B1",
            ]
        ),
        encoding="utf-8",
    )
    decl = tmp_path / "vocab.yaml"
    decl.write_text(
        yaml.safe_dump(
            {
                "vocabularies": [
                    {"vocabulary_id": "RxNorm", "release": "r1", "license": "synthetic"},
                    {
                        "vocabulary_id": "RxNorm Extension",
                        "release": "r1",
                        "license": "synthetic",
                    },
                    {"vocabulary_id": "OtherVocab", "release": "r1", "license": "synthetic"},
                ],
                "relationship_families": [["RxNorm", "RxNorm Extension"]],
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    res = CliRunner().invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(src),
            "--declaration",
            str(decl),
            "--output",
            str(bundle),
        ],
    )
    assert res.exit_code == 0, res.output

    store = LexiconStore(bundle)
    try:
        edges = store.incoming_relationships("RxNorm", ["B1"], ["has_tradename"])
        assert {(e.subject_vocabulary, e.subject, e.predicate) for e in edges} == {
            ("RxNorm", "P1", "has_tradename"),
            ("RxNorm Extension", "P2", "has_tradename"),
        }
        # Memoized on second call.
        edges2 = store.incoming_relationships("RxNorm", ["B1"], ["has_tradename"])
        assert edges2 == edges
    finally:
        store.close()


def test_refinement_multiple_keeps_coarse_anchor_and_records_candidates(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "concepts.csv").write_text(
        "\n".join(
            [
                "vocabulary_id,concept_code,concept_name,concept_class,status",
                "RxNorm,I1,mock ingredient,IN,active",
                "RxNorm,B1,MockBrand,BN,active",
                "RxNorm,P1,MockBrand 50 MG Oral Tablet,SBD,active",
                "RxNorm,P2,MockBrand 50 MG Chewable Tablet,SBD,active",
            ]
        ),
        encoding="utf-8",
    )
    (src / "synonyms.csv").write_text(
        "vocabulary_id,concept_code,synonym\nRxNorm,B1,MOCKBRAND\n", encoding="utf-8"
    )
    (src / "relationships.csv").write_text(
        "\n".join(
            [
                "vocabulary_id_1,concept_code_1,predicate,vocabulary_id_2,concept_code_2",
                "RxNorm,B1,has_ingredient,RxNorm,I1",
                "RxNorm,P1,has_tradename,RxNorm,B1",
                "RxNorm,P2,has_tradename,RxNorm,B1",
                "RxNorm,P1,has_ingredient,RxNorm,I1",
                "RxNorm,P2,has_ingredient,RxNorm,I1",
            ]
        ),
        encoding="utf-8",
    )
    decl = tmp_path / "vocab.yaml"
    decl.write_text(
        yaml.safe_dump(
            {"vocabularies": [{"vocabulary_id": "RxNorm", "release": "r1", "license": "synthetic"}]}
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    res = CliRunner().invoke(
        app,
        [
            "lexicon",
            "build",
            "--source",
            str(src),
            "--declaration",
            str(decl),
            "--output",
            str(bundle),
        ],
    )
    assert res.exit_code == 0, res.output
    from biosemalign.knowledge.lexicon.contract import LexiconManifest

    manifest = LexiconManifest.model_validate_json((bundle / "manifest.json").read_text())
    profile = appendix_a_drug_profile()
    with appendix_a_pipeline(tmp_path / "pipe", bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("MOCKBRAND 50 MG"))

    decision = artifact.decision
    assert decision.decision is DecisionType.MAPPED
    assert decision.source_level_concept is not None
    assert decision.source_level_concept.concept_id == "B1"
    assert [c.concept_id for c in decision.selected_concepts] == ["I1"]
    assert decision.refinement is not None
    assert decision.refinement.status == "multiple"
    assert decision.refinement.selected is None
    assert {c.concept_code for c in decision.refinement.candidates} == {"P1", "P2"}
    assert decision.rationale is not None
    assert "2 specific concepts" in decision.rationale
    assert decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert verify_artifact(artifact, profile=profile).valid

    # Tampering with decision.refinement must fail consumer verification.
    payload = artifact.model_dump(mode="json")
    payload["decision"]["refinement"]["status"] = "none"
    assert not verify_artifact(payload).valid

    # Schema 0.4.0 artifacts without package.links are rejected.
    no_links = artifact.model_dump(mode="json")
    no_links["knowledge_package"]["links"] = None
    with pytest.raises(ValidationError, match=r"require package\.links"):
        verify_artifact.__globals__["AlignmentArtifact"].model_validate(no_links)


def test_g2_strips_prohibited_expansion_paths(tmp_path: Path) -> None:
    bundle, manifest = build_appendix_a_bundle(tmp_path / "app_a")
    profile = appendix_a_drug_profile()

    class _LeakyExpander:
        op_id = "leaky"
        version = "0.1.0"
        uses_model = False

        def __init__(self) -> None:
            self.options: dict[str, Any] = {}

        def expand(
            self, sau: SemanticAlignmentUnit, anchors: list[CandidateConcept]
        ) -> ExpansionSet:
            del sau, anchors
            forbidden_candidate = CandidateConcept(
                candidate_id="T999",
                preferred_label="Forbidden Target",
                vocabulary="RxNorm",
            )
            anchor = QualifiedCode(vocabulary="RxNorm", concept_code="T100", label="toyprolol")
            to_code = QualifiedCode(
                vocabulary="RxNorm", concept_code="T999", label="Forbidden Target"
            )
            return ExpansionSet(
                expander="leaky",
                anchors=[anchor],
                members=[
                    ExpansionMember(
                        concept=forbidden_candidate,
                        anchor=anchor,
                        hops=1,
                        path=[
                            ExpansionStep(from_code=anchor, predicate="may_treat", to_code=to_code)
                        ],
                        via_operator="leaky",
                    )
                ],
            )

    def workflow(ctx: OpContext) -> Cascade:
        cfg = ctx.profile.tiers[0]
        return Cascade(
            [
                Tier(
                    name="checked",
                    match=build("match", cfg.match[0].op, ctx, cfg.match[0].options),
                    expander=_LeakyExpander(),  # type: ignore[arg-type]
                    linker=build("link", cfg.link.op, ctx, cfg.link.options),
                    resolver=build("resolve", "rules", ctx, {}),
                )
            ]
        )

    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=settings.mode),
        workflow=workflow,
    ) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("toyprolol"))

    assert all(c.candidate_id != "T999" for c in artifact.knowledge_package.candidates)
    assert [c.concept_id for c in artifact.decision.selected_concepts] == ["T100"]
