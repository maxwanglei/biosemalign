"""The coding → linking → resolution modules (v0.3 P2)."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from tests.lexicon_support import (
    CONCORDANCE_WORKFLOW,
    CONCORDANT,
    build_bundle,
    concordance_profile,
    lexicon_pipeline,
)
from tests.links_support import write_athena, write_dictionary, write_mesh, write_unii_zip
from tests.unit.test_lexicon import _resign

from biosemalign import BioSemAlign, verify_artifact
from biosemalign.coding import Coder, CodeSet
from biosemalign.enums import (
    Authority,
    ConceptStatus,
    ConcordanceState,
    LinkKind,
    RetrievalChannel,
)
from biosemalign.exceptions import ConfigurationError
from biosemalign.knowledge.lexicon.build import build_lexicon, load_declaration
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.knowledge.lexicon.links import LINK_OVERLAYS, derive_links
from biosemalign.linking import Linker, derive_link_set
from biosemalign.ops import OpContext
from biosemalign.ops.base import PassToNext
from biosemalign.ops.registry import available_ops, build_op, canonical_kind, register_op
from biosemalign.profiles.schema import TaskProfile, TierConfig
from biosemalign.resolution import Resolver
from biosemalign.schemas.codes import EquivalenceGroup, QualifiedCode
from biosemalign.schemas.knowledge import (
    CandidateConcept,
    ConcordanceAssessment,
    KnowledgePackage,
    KnowledgeStatement,
)
from biosemalign.targets import builtin_specs

RXNORM = builtin_specs()["rxnorm"]


@pytest.fixture(scope="module")
def lexicon(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    return build_bundle(tmp_path_factory.mktemp("lexicon"))


@pytest.fixture(scope="module")
def linked(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    root = tmp_path_factory.mktemp("linked")
    base = write_dictionary(root / "dictionary")
    derive_links(
        base,
        root / "links",
        athena=write_athena(root / "athena"),
        unii=write_unii_zip(root / "UNIIs.zip"),
        mesh=write_mesh(root / "mesh"),
        product_evidence=base / "product_evidence.csv",
    )
    bundle = root / "bundle"
    manifest = build_lexicon(
        [base, *(root / "links" / name for name in LINK_OVERLAYS)],
        bundle,
        declaration=load_declaration(root / "links" / "vocabularies.yaml"),
        trigram_index=False,
    )
    return bundle, manifest


VOCABULARIES = [
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


def module_profile(tier: Mapping[str, Any], *, spec: str | None = None) -> TaskProfile:
    """A two-tier profile written with the module keys."""
    payload: dict[str, Any] = {
        "profile_name": "modules",
        "profile_version": "1",
        "target": {
            "spec": spec or "rxnorm",
            "terminology": "RxNorm",
            "granularity": "ingredient",
            "source_terminologies": ["RxNorm Extension"],
        },
        "task": {"operation": "drug_normalization", "mapping_intent": "ingredient_level_analysis"},
        "knowledge": {
            "primary_resources": ["RxNorm"],
            "secondary_resources": [v for v in VOCABULARIES if v != "RxNorm"],
            "allowed_relationships": [
                "has_ingredient",
                "has_precise_ingredient",
                "has_tradename",
                "has_dose_form",
                "consists_of",
            ],
        },
        "workflow": {
            "tiers": [
                {"name": "step1", **tier},
                {
                    "name": "fallback",
                    "coding": {
                        "op": "lexicon",
                        "options": {"vocabularies": ["RxNorm"], "methods": ["exact", "bm25"]},
                    },
                },
            ]
        },
    }
    return TaskProfile.model_validate(payload)


STEP1 = {
    "coding": [{"op": "exact", "options": {"vocabularies": VOCABULARIES}}],
    "linking": {
        "op": "concordance",
        "options": {"conflict_vocabularies": ["RxNorm", "RxNorm Extension"]},
    },
    "resolution": {
        "op": "concordant",
        "options": {"authority_sources": ["RxNorm", "RxNorm Extension"]},
    },
}


@pytest.fixture
def step1(linked: tuple[Path, LexiconManifest], tmp_path: Path) -> Iterator[BioSemAlign]:
    bundle, manifest = linked
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=module_profile(STEP1)) as pipeline:
        yield pipeline


# -- registry ----------------------------------------------------------------------


def test_module_kinds_name_the_stages() -> None:
    assert canonical_kind("code") == "retrieve"
    assert canonical_kind("link") == "fuse"
    assert canonical_kind("resolve") == "propose"
    assert canonical_kind("validate") == "validate"
    with pytest.raises(ValueError, match="unknown operator kind"):
        canonical_kind("decide")
    assert {"exact", "lexicon", "dense", "rxnav"} <= set(available_ops("code"))
    assert available_ops("code") == available_ops("retrieve")
    assert {"concordance", "cross_vocab_concordance", "knowledge_package"} <= set(
        available_ops("link")
    )
    assert {"concordant", "concordant_exact", "adjudicator", "llm"} <= set(available_ops("resolve"))


def test_built_in_operators_implement_the_module_protocols(
    step1: BioSemAlign,
) -> None:
    tier = step1.kernel.workflow.tiers[0]
    assert isinstance(tier.retrievers[0], Coder)
    assert isinstance(tier.fuser, Linker)
    assert isinstance(tier.proposer, Resolver)
    sau = step1.build_sau(step1.make_request("Alphabrand"))[0]
    codes = tier.retrievers[0].code(sau)
    assert isinstance(codes, CodeSet) and codes.candidates
    package = tier.fuser.link(sau, [codes])
    assert package.concordance is not None
    assert isinstance(tier.proposer.resolve(sau, package), PassToNext | object)


# -- third-party module-spelled operators ----------------------------------------------


class _UpperCoder:
    """A coder spelled only with ``code``: exact lexicon matches of the upper-cased text."""

    op_id = "upper_coder"
    version = "1"
    requires_network = False

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    @property
    def options(self) -> dict[str, Any]:
        return {"wraps": self.inner.op_id}

    def code(self, sau: Any) -> CodeSet:
        return self.inner.retrieve(sau)


class _PassingLinker:
    op_id = "passing_linker"
    version = "1"

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def link(self, sau: Any, code_sets: Sequence[CodeSet]) -> KnowledgePackage:
        return self.inner.fuse(sau, code_sets)


class _AlwaysPass:
    op_id = "always_pass"
    version = "1"
    requires_network = False

    def resolve(self, sau: Any, package: KnowledgePackage) -> PassToNext:
        return PassToNext("declined_by_policy", f"{len(package.candidates)} candidates")

    def provenance(self) -> Any:
        raise AssertionError("a passing resolver is never asked for provenance")


def test_operators_spelled_with_module_names_are_adapted(
    linked: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    def coder(context: OpContext, options: Mapping[str, Any]) -> _UpperCoder:
        return _UpperCoder(build_op("code", "exact", context, {"vocabularies": VOCABULARIES}))

    def linker(context: OpContext, options: Mapping[str, Any]) -> _PassingLinker:
        return _PassingLinker(build_op("link", "concordance", context, {}))

    register_op("code", "upper_coder", coder)
    register_op("link", "passing_linker", linker)
    register_op("resolve", "always_pass", lambda context, options: _AlwaysPass())
    register_op("resolve", "not_an_operator", lambda context, options: object())
    bundle, manifest = linked
    tier = {"coding": "upper_coder", "linking": "passing_linker", "resolution": "always_pass"}
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=module_profile(tier)) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("Alphabrand"))
        built = pipeline.kernel.workflow.tiers[0]
        assert built.retrievers[0].requires_network is False
        assert built.retrievers[0].options == {"wraps": "lexicon"}
        assert built.proposer.requires_network is False
    attempt = artifact.decision.provenance.resolution.attempts[0]
    assert (attempt.tier, attempt.reason) == ("step1", "declined_by_policy")
    operators = artifact.decision.provenance.workflow.tiers[0].operators
    assert [op.op_id for op in operators] == ["upper_coder", "passing_linker", "always_pass"]
    assert verify_artifact(artifact).valid

    context = OpContext(profile=module_profile(STEP1))
    with pytest.raises(ConfigurationError, match="does not implement the ProposeOp"):
        build_op("resolve", "not_an_operator", context)


# -- profiles: two spellings, one profile -----------------------------------------------


def test_module_keys_and_stage_keys_are_the_same_profile(
    lexicon: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    old = concordance_profile()
    payload = old.model_dump(mode="json")
    new_tiers = []
    for tier in CONCORDANCE_WORKFLOW["tiers"]:
        renamed = {"name": tier["name"], "coding": tier["retrievers"]}
        if "fuser" in tier:
            renamed["linking"] = tier["fuser"]
        if "proposer" in tier:
            renamed["resolution"] = tier["proposer"]
        new_tiers.append(renamed)
    payload["workflow"] = {"tiers": new_tiers}
    new = TaskProfile.model_validate(payload)
    assert new.model_dump() == old.model_dump()

    bundle, manifest = lexicon
    artifacts = []
    for name, profile in (("old", old), ("new", new)):
        with lexicon_pipeline(tmp_path / name, bundle, manifest, profile=profile) as pipeline:
            artifacts.append(pipeline.align_artifact(pipeline.make_request(CONCORDANT)))
    first, second = (a.decision for a in artifacts)
    assert first.provenance.workflow == second.provenance.workflow
    assert first.selected_concepts == second.selected_concepts
    assert first.matched_codes == second.matched_codes


def test_tier_module_keys_are_checked() -> None:
    with pytest.raises(ValidationError, match="both 'coding' and 'retrievers'"):
        TierConfig.model_validate({"name": "t", "coding": "exact", "retrievers": ["lexicon"]})
    tier = TierConfig.model_validate({"name": "t", "coding": "exact", "resolution": "concordant"})
    assert [op.op for op in tier.retrievers] == ["exact"] and tier.proposer.op == "concordant"


def test_the_exact_coder_is_the_lexicon_coder_on_exact_names(
    lexicon: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = lexicon
    with lexicon_pipeline(tmp_path, bundle, manifest, profile=concordance_profile()) as pipeline:
        context = OpContext(profile=pipeline.profile, knowledge=pipeline.knowledge)
        exact = build_op("code", "exact", context, {"vocabularies": ["RxNorm"]})
        assert (exact.op_id, exact.options["methods"]) == ("lexicon", ["exact"])
        with pytest.raises(ConfigurationError, match="exact names only"):
            build_op("code", "exact", context, {"methods": ["bm25"]})


# -- links --------------------------------------------------------------------------------


def _links(package: KnowledgePackage) -> set[tuple[str, str, str, str]]:
    assert package.links is not None
    return {
        (
            f"{link.source.vocabulary}:{link.source.concept_code}",
            f"{link.via.vocabulary}:{link.via.concept_code}",
            f"{link.target.vocabulary}:{link.target.concept_code}",
            link.kind.value,
        )
        for link in package.links.links
    }


def test_the_kernel_links_every_matched_code_into_the_target(step1: BioSemAlign) -> None:
    artifact = step1.align_artifact(step1.make_request("Alphabrand"))
    package = artifact.knowledge_package
    assert package.links is not None
    assert (package.links.target_spec, package.links.granularity) == ("rxnorm@1.0", "ingredient")
    assert _links(package) == {
        ("RxNorm:40", "RxNorm:40", "RxNorm:10", "PROJECTION"),
        ("Drugs@FDA:000001-001", "RxNorm:40", "RxNorm:10", "PROJECTION"),
        ("Drugs@FDA:000005-001", "RxNorm:40", "RxNorm:10", "PROJECTION"),
    }
    projection = package.links.links[0]
    assert projection.predicates == ["has_ingredient"]
    assert projection.assertion_sources[0].startswith("OMOP brand products")
    assert package.links.targets_of("RxNorm", "40")[0].concept_code == "10"
    assert artifact.decision.provenance.workflow.target_spec == "rxnorm@1.0"

    exact = step1.align_artifact(step1.make_request("Alphamab")).knowledge_package
    assert ("UNII:UNIIALPHA1", "RxNorm:10", "RxNorm:10", "EQUIVALENCE") in _links(exact)


def test_forged_links_fail_verification(step1: BioSemAlign) -> None:
    artifact = step1.align_artifact(step1.make_request("Alphabrand"))
    assert verify_artifact(artifact).valid
    forged = artifact.model_dump(mode="json")
    forged["knowledge_package"]["links"]["links"][0]["target"]["concept_code"] = "11"
    assert any("links its evidence supports" in e for e in verify_artifact(_resign(forged)).errors)
    forged = artifact.model_dump(mode="json")
    forged["knowledge_package"]["links"]["granularity"] = "precise_ingredient"
    assert any("different granularity" in e for e in verify_artifact(_resign(forged)).errors)


def test_links_under_a_custom_spec_are_checked_against_the_evidence(
    linked: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    payload = RXNORM.model_dump(mode="json")
    payload.update(spec_id="custom", version="1")
    spec_path = tmp_path / "custom.yaml"
    spec_path.write_text(yaml.safe_dump(payload))
    bundle, manifest = linked
    profile = module_profile(STEP1, spec=str(spec_path))
    with lexicon_pipeline(tmp_path / "run", bundle, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("Alphabrand"))
    assert artifact.knowledge_package.links.target_spec == "custom@1"
    # A different spec is a different workflow.
    with lexicon_pipeline(
        tmp_path / "builtin", bundle, manifest, profile=module_profile(STEP1)
    ) as p:
        builtin = p.align_artifact(p.make_request("Alphabrand"))
    assert (
        artifact.decision.provenance.workflow.workflow_id
        != builtin.decision.provenance.workflow.workflow_id
    )
    assert verify_artifact(artifact).valid

    def forge(path: str, value: Any) -> list[str]:
        forged = artifact.model_dump(mode="json")
        node: Any = forged["knowledge_package"]["links"]["links"][0]
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
        return verify_artifact(_resign(forged)).errors

    assert any("edge absent" in e for e in forge("target.concept_code", "11"))
    assert any("no equivalence group" in e for e in forge("source.concept_code", "41"))
    # A route in the group but missing from the package's candidates.
    forged = artifact.model_dump(mode="json")
    forged["knowledge_package"]["candidates"] = [
        c for c in forged["knowledge_package"]["candidates"] if c["candidate_id"] != "000001-001"
    ]
    forged["knowledge_package"]["links"]["links"][0]["via"] = {
        "vocabulary": "Drugs@FDA",
        "concept_code": "000001-001",
        "label": None,
    }
    assert any("absent from the package" in e for e in verify_artifact(_resign(forged)).errors)
    assert any("lead to its route" in e for e in forge("kind", "EQUIVALENCE"))


def _member(vocabulary: str, code: str) -> QualifiedCode:
    return QualifiedCode(vocabulary=vocabulary, concept_code=code)


def _family(vocabulary: str, code: str, klass: str, *ingredients: str) -> CandidateConcept:
    return CandidateConcept(
        candidate_id=code,
        preferred_label=code,
        vocabulary=vocabulary,
        concept_class=klass,
        status=ConceptStatus.ACTIVE,
        relationships=[
            KnowledgeStatement(
                subject_id=code,
                predicate="has_ingredient",
                object_id=ingredient,
                object_vocabulary="RxNorm",
                source_resource=vocabulary,
                authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
            )
            for ingredient in ingredients
        ],
    )


def test_codes_leading_to_different_targets_are_reported_ambiguous() -> None:
    package = KnowledgePackage(
        sau_id="SAU-1",
        candidates=[
            _family("RxNorm", "B1", "BN", "10"),
            _family("RxNorm Extension", "E1", "BN", "10", "11"),
        ],
        concordance=ConcordanceAssessment(
            state=ConcordanceState.MULTI_SOURCE_CONCORDANT,
            groups=[
                EquivalenceGroup(
                    members=[
                        _member("RxNorm", "B1"),
                        _member("RxNorm Extension", "E1"),
                        _member("UNII", "U1"),
                        _member("RxNorm", "MISSING"),
                    ]
                )
            ],
        ),
    )
    links = derive_link_set(package, RXNORM, "ingredient")
    assert links is not None
    assert {(c.vocabulary, c.concept_code) for c in links.ambiguous} == {
        ("RxNorm", "B1"),
        ("RxNorm Extension", "E1"),
        ("UNII", "U1"),
        ("RxNorm", "MISSING"),
    }
    assert {t.concept_code for t in links.targets_of("RxNorm Extension", "E1")} == {"10", "11"}
    assert derive_link_set(package.model_copy(update={"concordance": None}), RXNORM, "x") is None
    assert derive_link_set(package, RXNORM, "nonsense").links == []


# -- resolution -------------------------------------------------------------------------


def test_the_concordant_resolver_uses_target_neutral_pass_reasons(step1: BioSemAlign) -> None:
    lone = step1.align_artifact(step1.make_request("Lonebrand"))
    assert lone.decision.provenance.resolution.attempts[0].reason == "no_target_link"
    foreign = step1.align_artifact(step1.make_request("Deltabrand"))
    assert foreign.decision.provenance.resolution.attempts[0].reason == "non_target_code"
    brand = step1.align_artifact(step1.make_request("Alphabrand"))
    assert brand.decision.provenance.resolution.resolved_by_tier == "step1"
    assert brand.decision.provenance.workflow.tiers[0].operators[-1].op_id == "concordant"


def test_a_resolver_without_kernel_links_applies_the_same_rule(step1: BioSemAlign) -> None:
    sau = step1.build_sau(step1.make_request("Alphabrand"))[0]
    package = step1.retrieve_knowledge(sau)
    resolver = step1.kernel.workflow.tiers[0].proposer
    with_links = resolver.resolve(sau, package)
    without = resolver.resolve(sau, package.model_copy(update={"links": None}))
    assert with_links.selected_concepts == without.selected_concepts
    lone = step1.build_sau(step1.make_request("Lonebrand"))[0]
    lone_package = step1.retrieve_knowledge(lone).model_copy(update={"links": None})
    assert resolver.resolve(lone, lone_package).reason == "no_target_link"


def test_module_packages_expose_their_protocols() -> None:
    import biosemalign.coding as coding
    import biosemalign.linking as linking
    import biosemalign.resolution as resolution

    for module in (coding, linking, resolution):
        with pytest.raises(AttributeError):
            module.nothing_here  # noqa: B018
    assert resolution.PassToNext is PassToNext
    assert copy.copy(LinkKind.PROJECTION) is LinkKind.PROJECTION
    assert RetrievalChannel.EXACT_LOOKUP in {RetrievalChannel.EXACT_LOOKUP}


def test_bulk_tools_refuse_a_granularity_the_spec_lacks(
    linked: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    from biosemalign.coding import code_terms
    from biosemalign.exceptions import LexiconError
    from biosemalign.knowledge.lexicon.audit import audit_bundle

    bundle, _ = linked
    terms = tmp_path / "terms.csv"
    terms.write_text("drugname\nAlphabrand\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="has no granularity 'clinical_drug'"):
        code_terms(
            bundle, terms, tmp_path / "out", text_column="drugname", granularity="clinical_drug"
        )
    with pytest.raises(LexiconError, match="has no granularity 'clinical_drug'"):
        audit_bundle(bundle, tmp_path / "audit", granularity="clinical_drug")
    report = code_terms(
        bundle, terms, tmp_path / "pin", text_column="drugname", granularity="precise_ingredient"
    )
    assert report.coded_terms == 1
