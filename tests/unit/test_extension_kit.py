"""Tests for the published ``biosemalign.testing`` contract kit and plugin kit (WP7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from tests.scenarios_support import (
    appendix_a_drug_profile,
    appendix_a_phenotyping_profile,
    appendix_a_toy_disease_hierarchy_profile,
    build_appendix_a_bundle,
)
from tests.support import synthetic_settings

from biosemalign import BioSemAlign
from biosemalign.enums import DecisionType, RelationshipType, Severity
from biosemalign.exceptions import WorkflowViolation
from biosemalign.knowledge.lexicon.contract import LexiconManifest
from biosemalign.modules import CandidateSet, OpContext, PassToNext, available, build
from biosemalign.modules.expand.hierarchy import HierarchyExpander
from biosemalign.modules.expand.lexical import LexicalNeighborsExpander
from biosemalign.modules.expand.refine import RefineBySlotsExpander
from biosemalign.modules.link.standard import StandardLinker
from biosemalign.modules.resolve.concordant import ConcordantResolver
from biosemalign.modules.resolve.deterministic import DeterministicResolver
from biosemalign.modules.select.hierarchy import HierarchySelector
from biosemalign.schemas.codes import QualifiedCode
from biosemalign.schemas.decision import AlignmentProposal, SelectedConcept, ValidationFinding
from biosemalign.schemas.expansion import ExpansionMember, ExpansionSet
from biosemalign.schemas.knowledge import KnowledgePackage, KnowledgeStatement
from biosemalign.schemas.phenotyping import ConceptSetMember, ConceptSetProposal
from biosemalign.schemas.provenance import ModelProvenance
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.storage.cache import ResponseCache
from biosemalign.targets import builtin_specs
from biosemalign.testing import (
    check_check_contract,
    check_cleaner_contract,
    check_domain_pack_contract,
    check_expander_contract,
    check_linker_contract,
    check_matcher_contract,
    check_resolver_contract,
    check_selector_contract,
    check_target_spec_contract,
    make_contract_candidate,
    make_contract_package,
    make_contract_sau,
)


@pytest.fixture(scope="module")
def kit_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, LexiconManifest]:
    return build_appendix_a_bundle(tmp_path_factory.mktemp("ext_kit"))


def test_no_public_route_outside_kernel_and_plugin_autoregistry() -> None:
    # Stage-by-stage methods return proposals & validation results only; no bare route().
    assert not hasattr(BioSemAlign, "route")
    assert not hasattr(BioSemAlign, "_provenance")
    # Calling available() / build() triggers built-in and plugin registration cleanly.
    assert "unicode_normalize" in available("clean")
    assert "drug.extract_strength" in available("clean")
    assert "hierarchy" in available("expand")
    assert "hierarchy" in available("select")
    check_domain_pack_contract("drug")
    check_domain_pack_contract("hierarchy")
    specs = builtin_specs()
    check_target_spec_contract(specs["rxnorm"])
    check_target_spec_contract(specs["toydisease"])


def test_contract_kit_validates_all_builtin_operators(
    kit_bundle: tuple[Path, LexiconManifest], tmp_path: Path
) -> None:
    bundle, manifest = kit_bundle
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir, lexicon_path=bundle, lexicon_sha256=manifest.bundle_sha256
    )

    drug_profile = appendix_a_drug_profile()
    with BioSemAlign(
        drug_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pipe:
        ctx = OpContext(profile=drug_profile, settings=settings, knowledge=pipe.knowledge)

        # Cleaners
        sau_strength = make_contract_sau("TOYBRAND XL 50MG")
        for clean_name in ("unicode_normalize", "drug.extract_strength"):
            cleaner = build("clean", clean_name, ctx, {})
            check_cleaner_contract(cleaner, sau_strength)

        # Matchers
        sau_brand = make_contract_sau("TOYBRAND XL")
        exact = build("match", "exact", ctx, {"vocabularies": ["RxNorm"]})
        exact_cs = check_matcher_contract(exact, sau_brand)
        assert len(exact_cs.candidates) == 1
        bm25 = build("match", "lexical", ctx, {"vocabularies": ["RxNorm"], "method": "bm25"})
        check_matcher_contract(bm25, sau_brand)

        # Linker + Expand + Resolve + Check
        linker = build("link", "standard", ctx, {"concordance": True})
        assert isinstance(linker, StandardLinker)
        pkg = check_linker_contract(linker, sau_brand, [exact_cs], profile=drug_profile)

        refine = build("expand", "refine_by_slots", ctx, {})
        assert isinstance(refine, RefineBySlotsExpander)
        check_expander_contract(refine, sau_strength, exact_cs.candidates, profile=drug_profile)

        for res_name, opts in (
            ("rules", {}),
            ("concordant", {"authority_sources": ["RxNorm"]}),
        ):
            resolver = build("resolve", res_name, ctx, opts)
            assert isinstance(resolver, DeterministicResolver | ConcordantResolver)
            out = check_resolver_contract(resolver, sau_brand, pkg, profile=drug_profile)
            if isinstance(out, AlignmentProposal):
                for chk_name in ("drug.semantics",):
                    chk = build("check", chk_name, ctx, {})
                    check_check_contract(chk, out, pkg, sau_brand)

    # Phenotyping expanders, selector, and hierarchy check
    pheno_profile = appendix_a_phenotyping_profile()
    with BioSemAlign(
        pheno_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as pheno_pipe:
        pctx = OpContext(profile=pheno_profile, settings=settings, knowledge=pheno_pipe.knowledge)
        td_sau = make_contract_sau(
            "toy fever", target_terminology="ToyDisease", desired_granularity="Disease"
        )
        d1 = pheno_pipe.knowledge.provider_for("ToyDisease").lookup("D1")
        assert d1 is not None

        hier_exp = build("expand", "hierarchy", pctx, {})
        assert isinstance(hier_exp, HierarchyExpander)
        exp_set = check_expander_contract(hier_exp, td_sau, [d1], profile=pheno_profile)

        lex_exp = build("expand", "lexical_neighbors", pctx, {"method": "trigram"})
        assert isinstance(lex_exp, LexicalNeighborsExpander)
        check_expander_contract(lex_exp, td_sau, [d1], profile=pheno_profile)

        hier_sel = build("select", "hierarchy", pctx, {})
        assert isinstance(hier_sel, HierarchySelector)
        check_selector_contract(hier_sel, td_sau, exp_set, profile=pheno_profile)

    hier_map_profile = appendix_a_toy_disease_hierarchy_profile()
    with BioSemAlign(
        hier_map_profile, settings=settings, cache=ResponseCache(cache_dir, mode=settings.mode)
    ) as hpipe:
        hctx = OpContext(profile=hier_map_profile, settings=settings, knowledge=hpipe.knowledge)
        chk = build("check", "hierarchy.semantics", hctx, {})
        assert chk.op_id == "hierarchy.semantics"
        sau_d = make_contract_sau(
            "severe toy fever", target_terminology="ToyDisease", desired_granularity="Group"
        )
        cand = make_contract_candidate(
            code="D2", label="severe toy fever", vocabulary="ToyDisease", concept_class="Disease"
        )
        pkg_d = make_contract_package(sau_d, [cand])
        prop_d = AlignmentProposal(
            sau_id=sau_d.sau_id,
            decision=DecisionType.NO_VALID_MAPPING,
            relationship_to_source=RelationshipType.UNSUPPORTED_MAPPING,
            rationale="test",
        )
        check_check_contract(chk, prop_d, pkg_d, sau_d)


def test_contract_kit_catches_broken_operators() -> None:
    profile = appendix_a_drug_profile()
    sau = make_contract_sau()
    cand = make_contract_candidate()
    pkg = make_contract_package(sau, [cand])

    # 1. Broken cleaner overwriting source mention fails G1.
    class _MutatingCleaner:
        op_id = "bad.cleaner"
        version = "0.1.0"
        uses_model = False

        def clean(self, sau: SemanticAlignmentUnit) -> SemanticAlignmentUnit:
            return sau.model_copy(
                update={"mention": sau.mention.model_copy(update={"text": "tampered"})}
            )

    with pytest.raises(WorkflowViolation):
        check_cleaner_contract(_MutatingCleaner(), sau)

    # 2. Non-deterministic non-model matcher fails reproducibility check.
    class _RandomMatcher:
        op_id = "bad.matcher"
        version = "0.1.0"
        uses_model = False
        requires_network = False

        def __init__(self) -> None:
            self.n = 0

        def match(self, sau: SemanticAlignmentUnit) -> CandidateSet:
            del sau
            self.n += 1
            return CandidateSet(
                matcher=self.op_id,
                candidates=[make_contract_candidate(code=f"T{self.n}")],
            )

    with pytest.raises(AssertionError, match="reproducible"):
        check_matcher_contract(_RandomMatcher(), sau)

    # 3. Expander emitting a member with an empty path fails path requirement.
    class _PathlessExpander:
        op_id = "bad.expander"
        version = "0.1.0"
        uses_model = False

        def expand(self, sau: SemanticAlignmentUnit, anchors: Any) -> ExpansionSet:
            del sau, anchors
            return ExpansionSet(
                expander=self.op_id,
                anchors=[],
                members=[
                    ExpansionMember(
                        concept=make_contract_candidate(code="T999"),
                        anchor=QualifiedCode(vocabulary="RxNorm", concept_code="T100"),
                        hops=0,
                        path=[],
                        via_operator=self.op_id,
                    )
                ],
            )

    with pytest.raises(AssertionError, match="non-empty path"):
        check_expander_contract(_PathlessExpander(), sau, [cand], profile=profile)

    # 4. Resolver proposing code outside package fails G4.
    class _InventingResolver:
        op_id = "bad.resolver"
        version = "0.1.0"
        uses_model = False
        requires_network = False

        def provenance(self) -> ModelProvenance:
            return ModelProvenance(backend="bad", model_name="bad@0.1.0")

        def resolve(
            self, sau: SemanticAlignmentUnit, package: KnowledgePackage
        ) -> AlignmentProposal:
            del package
            return AlignmentProposal(
                sau_id=sau.sau_id,
                decision=DecisionType.MAPPED,
                selected_concepts=[
                    SelectedConcept(
                        concept_id="INVENTED",
                        label="invented",
                        vocabulary="RxNorm",
                        relationship_to_source=RelationshipType.EXACT,
                    )
                ],
                relationship_to_source=RelationshipType.EXACT,
                rationale="invented code",
            )

    with pytest.raises(AssertionError, match="outside package"):
        check_resolver_contract(_InventingResolver(), sau, pkg, profile=profile)

    # 5. Selector proposing code outside expansion fails G4.
    class _InventingSelector:
        op_id = "bad.selector"
        version = "0.1.0"
        uses_model = False
        requires_network = False

        def provenance(self) -> ModelProvenance:
            return ModelProvenance(backend="bad", model_name="bad@0.1.0")

        def select(self, sau: SemanticAlignmentUnit, expansion: ExpansionSet) -> ConceptSetProposal:
            del expansion
            return ConceptSetProposal(
                sau_id=sau.sau_id,
                idea=sau.mention.text,
                target_terminology=sau.task.target_terminology,
                members=[
                    ConceptSetMember(
                        vocabulary="RxNorm",
                        concept_code="T999",
                        label="invented",
                        relation=RelationshipType.EXACT,
                        scope="narrow",
                        reason="outside expansion",
                    )
                ],
            )

    exp = ExpansionSet(expander="none", anchors=[], members=[])
    with pytest.raises(AssertionError, match="outside the expansion"):
        check_selector_contract(_InventingSelector(), sau, exp, profile=profile)

    # 6. Linker leaking prohibited relationship fails G2, Check returning wrong type or mutating fails.
    class _LeakingLinker:
        op_id = "bad.linker"
        version = "0.1.0"
        uses_model = False

        def link(self, sau: SemanticAlignmentUnit, candidate_sets: Any) -> KnowledgePackage:
            del candidate_sets
            from biosemalign.enums import Authority

            bad_cand = make_contract_candidate(
                relationships=[
                    KnowledgeStatement(
                        subject_id="T100",
                        predicate="has_adverse_effect",
                        object_id="OUTCOME",
                        source_resource="RxNorm",
                        authority=Authority.TARGET_TERMINOLOGY,
                    )
                ]
            )
            return KnowledgePackage(sau_id=sau.sau_id, candidates=[bad_cand])

    with pytest.raises(AssertionError, match="prohibited predicates"):
        check_linker_contract(_LeakingLinker(), sau, [], profile=profile)

    class _GoodCheck:
        op_id = "good.check"
        version = "0.1.0"
        uses_model = False
        options: ClassVar[dict[str, Any]] = {}

        def check(
            self,
            proposal: AlignmentProposal,
            package: KnowledgePackage,
            sau: SemanticAlignmentUnit | None,
        ) -> list[ValidationFinding]:
            del proposal, package, sau
            return [ValidationFinding(code="OK", severity=Severity.INFO, message="ok")]

    prop_ok = AlignmentProposal(
        sau_id=sau.sau_id,
        decision=DecisionType.NO_VALID_MAPPING,
        relationship_to_source=RelationshipType.UNSUPPORTED_MAPPING,
        rationale="none",
    )
    assert len(check_check_contract(_GoodCheck(), prop_ok, pkg, sau)) == 1
    assert isinstance(PassToNext(reason="skip"), PassToNext)
