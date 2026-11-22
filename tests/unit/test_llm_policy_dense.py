"""M3: the leashed LLM adjudicator, declarative policy packs, and dense retrieval."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel
from tests.lexicon_support import build_bundle, lexicon_pipeline, lexicon_profile
from tests.support import (
    ALPHA,
    NO_MATCH,
    PRODUCT,
    UNSUPPORTED,
    make_pipeline,
    synthetic_settings,
)
from typer.testing import CliRunner

from biosemalign import BioSemAlign
from biosemalign.cli import app
from biosemalign.enums import DecisionType, EgressDataClass, ResourceMode, RoutingStatus
from biosemalign.exceptions import (
    AdjudicationFailure,
    CacheMiss,
    ConfigurationError,
    LexiconError,
    ResourceNotPermitted,
)
from biosemalign.knowledge.lexicon.embeddings import (
    EmbeddingIndex,
    HashingEncoder,
    build_embedding_index,
)
from biosemalign.knowledge.providers.lexicon import LexiconStore
from biosemalign.modules.link.fusion import fuse_ranked
from biosemalign.modules.resolve.backend import GenerationConfig
from biosemalign.modules.resolve.leashed import LeashedLLMResolver, _choice_model
from biosemalign.modules.resolve.llm_backend import OpenAICompatibleBackend
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.storage.cache import ResponseCache
from biosemalign.validation.policy_dsl import load_policy_pack

LOCAL = "http://localhost:8000/v1"
COMPLETIONS = f"{LOCAL}/chat/completions"

AGREEING_PRODUCT = {
    "candidate": 1,
    "abstain_reason": "none",
    "relationship": "TARGET_BROADER_THAN_SOURCE",
    "semantic_loss": ["brand_identity", "dose_form", "salt_form", "strength"],
    "needs_review": False,
    "rationale": "the branded tablet",
}


def _mock_rxnav_release() -> None:
    """Live modes ask RxNav for its release before serving cached responses."""
    from tests.support import ENDPOINT, SYNTHETIC_RELEASE

    respx.get(f"{ENDPOINT}/version.json").mock(
        return_value=httpx.Response(200, json={"version": SYNTHETIC_RELEASE})
    )


def _reply(content: dict[str, Any] | str) -> httpx.Response:
    body = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(
        200, json={"choices": [{"message": {"content": body}, "finish_reason": "stop"}]}
    )


class _Echo(BaseModel):
    answer: int


class FakeBackend:
    """Replies with a fixed choice and remembers the prompt."""

    backend_name = "fake"

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []

    def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        generation_config: GenerationConfig,
    ) -> BaseModel:
        self.messages = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return output_schema.model_validate(self.reply)


def _leashed(reply: dict[str, Any] | Exception, profile: TaskProfile | None = None) -> Any:
    backend = FakeBackend(reply)
    resolver = LeashedLLMResolver(
        profile or load_profile("faers_drug_normalization"), backend, model_name="fake-model"
    )
    return resolver, backend


# -- backend -----------------------------------------------------------------


def _backend(tmp_path: Path, mode: ResourceMode = ResourceMode.CACHE_PREFER, **kwargs: Any) -> Any:
    options: dict[str, Any] = {
        "base_url": LOCAL,
        "model": "local-model",
        "model_revision": "rev-1",
        "cache": ResponseCache(tmp_path / "cache", mode=mode),
    }
    options.update(kwargs)
    return OpenAICompatibleBackend(**options)


@respx.mock
def test_backend_requests_schema_constrained_decoding_and_caches(tmp_path: Path) -> None:
    route = respx.post(COMPLETIONS).mock(return_value=_reply({"answer": 3}))
    backend = _backend(tmp_path)
    config = GenerationConfig(seed=7)
    messages = [{"role": "user", "content": "q"}]
    first = backend.generate_structured(
        messages=messages, output_schema=_Echo, generation_config=config
    )
    second = backend.generate_structured(
        messages=messages, output_schema=_Echo, generation_config=config
    )
    assert first == second == _Echo(answer=3)
    assert route.call_count == 1 and backend.cache_hits == 1 and backend.network_calls == 1
    sent = json.loads(route.calls[0].request.content)
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"]["required"] == ["answer"]
    assert sent["seed"] == 7 and sent["temperature"] == 0.0
    assert len(backend.request_digest(messages)) == 16


@respx.mock
def test_frozen_mode_replays_and_fails_closed(tmp_path: Path) -> None:
    respx.post(COMPLETIONS).mock(return_value=_reply({"answer": 1}))
    messages = [{"role": "user", "content": "q"}]
    _backend(tmp_path).generate_structured(
        messages=messages, output_schema=_Echo, generation_config=GenerationConfig()
    )
    replay = _backend(tmp_path, mode=ResourceMode.FROZEN)
    respx.post(COMPLETIONS).mock(side_effect=AssertionError("network used in FROZEN mode"))
    assert replay.generate_structured(
        messages=messages, output_schema=_Echo, generation_config=GenerationConfig()
    ) == _Echo(answer=1)
    with pytest.raises(CacheMiss):
        replay.generate_structured(
            messages=[{"role": "user", "content": "unseen"}],
            output_schema=_Echo,
            generation_config=GenerationConfig(),
        )
    with pytest.raises(ConfigurationError, match="LLM_MODEL_REVISION"):
        _backend(tmp_path, mode=ResourceMode.FROZEN, model_revision=None)


@respx.mock
def test_remote_endpoints_need_sensitive_text_egress_and_https(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "patient text"}]
    generate = {
        "messages": messages,
        "output_schema": _Echo,
        "generation_config": GenerationConfig(),
    }
    refused = _backend(tmp_path, base_url="https://llm.example.org/v1")
    with pytest.raises(ResourceNotPermitted, match="outside the deployment boundary"):
        refused.generate_structured(**generate)
    plain = _backend(
        tmp_path,
        base_url="http://llm.example.org/v1",
        permitted_egress_classes=frozenset({EgressDataClass.SENSITIVE_TEXT}),
    )
    with pytest.raises(ResourceNotPermitted, match="HTTPS"):
        plain.generate_structured(**generate)
    respx.post("https://llm.example.org/v1/chat/completions").mock(
        return_value=_reply({"answer": 2})
    )
    allowed = _backend(
        tmp_path,
        base_url="https://llm.example.org/v1",
        permitted_egress_classes=frozenset({EgressDataClass.SENSITIVE_TEXT}),
        api_key="secret",
    )
    assert allowed.generate_structured(**generate) == _Echo(answer=2)
    with pytest.raises(ConfigurationError, match="absolute"):
        _backend(tmp_path, base_url="not a url")
    with pytest.raises(ConfigurationError, match="credentials"):
        _backend(tmp_path, base_url="https://user:pw@llm.example.org/v1")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(302, headers={"location": "https://elsewhere"}), "redirect"),
        (httpx.Response(200, text="not json"), "unreadable"),
        (_reply('{"answer": "three"}'), "does not conform"),
        (httpx.Response(200, json={"choices": [{"message": {"content": None}}]}), "no message"),
    ],
)
@respx.mock
def test_unusable_replies_are_adjudication_failures(
    tmp_path: Path, response: httpx.Response, message: str
) -> None:
    respx.post(COMPLETIONS).mock(return_value=response)
    with pytest.raises(AdjudicationFailure, match=message):
        _backend(tmp_path).generate_structured(
            messages=[{"role": "user", "content": message}],
            output_schema=_Echo,
            generation_config=GenerationConfig(),
        )


@respx.mock
def test_transport_failures_are_adjudication_failures(tmp_path: Path) -> None:
    respx.post(COMPLETIONS).mock(side_effect=httpx.ConnectError("refused"))
    backend = _backend(tmp_path)
    with pytest.raises(AdjudicationFailure, match="transport"):
        backend.generate_structured(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_Echo,
            generation_config=GenerationConfig(),
        )
    backend.close()


# -- leashed adjudication ----------------------------------------------------


def test_reply_schema_admits_only_listed_candidates() -> None:
    model = _choice_model(3, ["EXACT"])
    schema = model.model_json_schema()
    assert schema["properties"]["candidate"]["enum"] == [0, 1, 2, 3]
    assert schema["additionalProperties"] is False
    with pytest.raises(ValueError):
        model.model_validate({**AGREEING_PRODUCT, "candidate": 4, "relationship": "EXACT"})


def test_agreeing_model_reproduces_the_rule_projection(tmp_path: Path) -> None:
    resolver, backend = _leashed(AGREEING_PRODUCT)
    with make_pipeline(tmp_path, resolver=resolver) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(PRODUCT))
    assert artifact.decision.selected_concepts[0].concept_id == "1001"
    assert artifact.decision.routing_status is RoutingStatus.AUTO_ACCEPT
    assert "llm" not in artifact.proposal.extensions
    assert len(artifact.decision.provenance.model_calls) == 1
    record = artifact.decision.provenance.model_calls[0]
    assert record.agrees_with_terminology is True and record.choice == 1
    assert artifact.proposal.rationale.startswith("model selected candidate 1 (2001)")
    assert "1. ExampleAlpha sodium 10 MG Oral Tablet" in backend.messages[1]["content"]
    provenance = artifact.decision.provenance.model
    assert (provenance.backend, provenance.model_name) == ("fake", "fake-model")
    assert provenance.prompt_version.startswith("leashed-1#")


def test_disagreement_with_terminology_goes_to_review(tmp_path: Path) -> None:
    resolver, _ = _leashed({**AGREEING_PRODUCT, "semantic_loss": ["strength"]})
    with make_pipeline(tmp_path, resolver=resolver) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(PRODUCT))
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert artifact.decision.provenance.model_calls[0].agrees_with_terminology is False
    assert any("model declared" in item for item in artifact.decision.unsupported_inferences)


@pytest.mark.parametrize(
    ("reason", "allow_auto_abstain", "decision", "routing"),
    [
        (
            "no_valid_candidate",
            False,
            DecisionType.NO_VALID_MAPPING,
            RoutingStatus.HUMAN_REVIEW,
        ),
        (
            "no_valid_candidate",
            True,
            DecisionType.NO_VALID_MAPPING,
            RoutingStatus.ABSTAIN,
        ),
        (
            "insufficient_context",
            False,
            DecisionType.INSUFFICIENT_CONTEXT,
            RoutingStatus.HUMAN_REVIEW,
        ),
        (
            "insufficient_context",
            True,
            DecisionType.INSUFFICIENT_CONTEXT,
            RoutingStatus.HUMAN_REVIEW,
        ),
    ],
)
def test_declining_every_candidate_abstains(
    tmp_path: Path,
    reason: str,
    allow_auto_abstain: bool,
    decision: DecisionType,
    routing: RoutingStatus,
) -> None:
    profile = load_profile("faers_drug_normalization")
    payload = profile.model_dump(mode="json")
    payload["routing"]["allow_model_auto_abstain"] = allow_auto_abstain
    configured = TaskProfile.model_validate(payload)
    resolver, _ = _leashed(
        {
            **AGREEING_PRODUCT,
            "candidate": 0,
            "abstain_reason": reason,
            "relationship": "UNSUPPORTED_MAPPING",
            "semantic_loss": [],
        },
        configured,
    )
    with make_pipeline(tmp_path, profile=configured, resolver=resolver) as pipeline:
        result = pipeline.align_artifact(pipeline.make_request(PRODUCT, route="oral"))
    assert result.decision.decision is decision
    assert result.decision.routing_status is routing
    assert result.decision.preserved_attributes == {"route": "oral"}


def test_unusable_model_reply_is_an_abstention_for_review(tmp_path: Path) -> None:
    resolver, _ = _leashed(AdjudicationFailure("model reply does not conform"))
    with make_pipeline(tmp_path, resolver=resolver) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request(PRODUCT))
    assert artifact.decision.decision is DecisionType.ABSTAINED
    assert artifact.decision.routing_status is RoutingStatus.HUMAN_REVIEW
    assert "does not conform" in (artifact.decision.provenance.model_calls[0].error or "")


def test_no_candidates_never_reach_the_model(tmp_path: Path) -> None:
    resolver, backend = _leashed(AssertionError("model consulted"))  # type: ignore[arg-type]
    with make_pipeline(tmp_path, resolver=resolver) as pipeline:
        decision = pipeline.align_value(NO_MATCH)
    assert decision.decision is DecisionType.NO_VALID_MAPPING
    assert backend.messages == []


def test_candidate_cap_and_blinding_reach_the_prompt(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    payload = profile.model_dump(mode="json")
    for field in payload["context"]["fields"]:
        if field["name"] == "indication":
            field["blinded"] = True
    profile = TaskProfile.model_validate(payload)
    resolver, backend = _leashed(AGREEING_PRODUCT, profile)
    with make_pipeline(tmp_path, profile=profile, resolver=resolver) as pipeline:
        # Startup verification keeps limits.candidates within the cap; the resolver
        # still truncates on its own, as a second line.
        resolver.max_candidates = 1
        pipeline.align_value(PRODUCT, indication="studied outcome", route="oral")
    prompt = backend.messages[1]["content"]
    assert "studied outcome" not in prompt and "route: oral" in prompt
    assert "\n2. " not in prompt

    payload["limits"]["candidates"] = 20
    with pytest.raises(ConfigurationError, match="accepts at most 15 candidates"):
        make_pipeline(
            tmp_path / "wide",
            profile=TaskProfile.model_validate(payload),
            resolver=_leashed(AGREEING_PRODUCT)[0],
        )


@respx.mock
def test_vllm_backend_runs_through_the_pipeline(tmp_path: Path) -> None:
    from tests.support import seed_synthetic_cache

    from biosemalign.modules import OpContext, build

    respx.post(COMPLETIONS).mock(return_value=_reply(AGREEING_PRODUCT))
    _mock_rxnav_release()
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["tiers"][0]["resolve"] = {"op": "llm"}
    cache_dir = tmp_path / "cache"
    settings = synthetic_settings(
        cache_dir,
        mode=ResourceMode.CACHE_PREFER,
        deployment_environment="local",
        llm_base_url=LOCAL,
        llm_model="local-model",
    )
    seed_synthetic_cache(cache_dir)
    profile = TaskProfile.model_validate(payload)
    with BioSemAlign(
        profile,
        settings=settings,
        cache=ResponseCache(cache_dir, mode=ResourceMode.CACHE_PREFER),
    ) as pipeline:
        decision = pipeline.align_value(PRODUCT)
    assert decision.selected_concepts[0].concept_id == "1001"
    assert decision.provenance.model.backend == "openai_compatible"

    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        build(
            "resolve",
            "llm",
            OpContext(
                profile=profile,
                settings=synthetic_settings(cache_dir),
                knowledge=pipeline.knowledge,
            ),
        )
    with pytest.raises(ConfigurationError, match="settings"):
        build("resolve", "llm", OpContext(profile=profile, knowledge=pipeline.knowledge))


@respx.mock
def test_llm_resolver_operator_in_a_cascade(tmp_path: Path) -> None:
    from tests.support import seed_synthetic_cache

    respx.post(COMPLETIONS).mock(return_value=_reply(AGREEING_PRODUCT))
    _mock_rxnav_release()
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["tiers"] = [
        {
            "name": "rules",
            "match": ["drug.rxnav"],
            "resolve": "rules",
            "pass_on_review_request": True,
        },
        {
            "name": "model",
            "match": ["drug.rxnav"],
            "resolve": {"op": "llm", "options": {"max_candidates": 10}},
        },
    ]
    cache_dir = tmp_path / "cache"
    seed_synthetic_cache(cache_dir)
    settings = synthetic_settings(
        cache_dir,
        mode=ResourceMode.CACHE_PREFER,
        deployment_environment="local",
        llm_base_url=LOCAL,
        llm_model="local-model",
    )
    with BioSemAlign(
        TaskProfile.model_validate(payload),
        settings=settings,
        cache=ResponseCache(cache_dir, mode=ResourceMode.CACHE_PREFER),
    ) as pipeline:
        settled = pipeline.align_value(PRODUCT)
        escalated = pipeline.align_value(UNSUPPORTED)
        operators = pipeline.kernel.workflow_provenance.tiers[1].operators
    assert settled.provenance.resolution.resolved_by_tier == "rules"
    trace = escalated.provenance.resolution
    assert [a.reason for a in trace.attempts] == ["review_requested", None]
    assert trace.resolved_by_tier == "model"
    assert escalated.provenance.model.backend == "openai_compatible"
    assert operators[-1].op_id == "llm"


# -- policy packs ------------------------------------------------------------


def _with_policy(pack: dict[str, Any]) -> TaskProfile:
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    payload["checks"] = ["drug.semantics", {"op": "policy", "options": {"pack": pack}}]
    return TaskProfile.model_validate(payload)


PACK: dict[str, Any] = {
    "policy_id": "perinatal",
    "policy_version": "1.0",
    "preserve_distinction_pairs": [
        {
            "rule_id": "salt",
            "source": {"label_pattern": "sodium"},
            "message": "the sodium salt is not interchangeable with the base",
        }
    ],
    "context_compatibility_rules": [
        {
            "rule_id": "route",
            "context_field": "route",
            "context_pattern": "intramuscular",
            "incompatible_pattern": "oral",
            "message": "an intramuscular route contradicts an oral product",
        }
    ],
    "non_target_category_tags": [
        {
            "rule_id": "control",
            "tag": "CONTROL_VEHICLE",
            "mention_pattern": "^ExampleAlpha$",
            "message": "named a control",
            "severity": "info",
        }
    ],
}


def test_policy_pack_findings_route_to_review(tmp_path: Path) -> None:
    with make_pipeline(tmp_path, profile=_with_policy(PACK)) as pipeline:
        product = pipeline.align_value(PRODUCT, route="intramuscular")
        exact = pipeline.align_value(ALPHA)
        version = pipeline.validator.version
    codes = {finding.code for finding in product.validation.findings}
    assert {"CRITICAL_SEMANTIC_LOSS", "CONTEXT_INCOMPATIBLE"} <= codes
    assert product.routing_status is RoutingStatus.HUMAN_REVIEW
    assert [f.code for f in exact.validation.findings if f.code == "NON_TARGET_CATEGORY"]
    assert exact.routing_status is RoutingStatus.AUTO_ACCEPT
    assert "+policy@1:perinatal@1.0#" in version


def test_policy_errors_reject(tmp_path: Path) -> None:
    pack = {
        **PACK,
        "preserve_distinction_pairs": [
            {**PACK["preserve_distinction_pairs"][0], "severity": "error"}
        ],
    }
    with make_pipeline(tmp_path, profile=_with_policy(pack)) as pipeline:
        decision = pipeline.align_value(PRODUCT)
    assert decision.decision is DecisionType.ABSTAINED
    assert decision.routing_status is RoutingStatus.HUMAN_REVIEW


def test_policy_packs_validate_on_load(tmp_path: Path) -> None:
    path = tmp_path / "pack.yaml"
    path.write_text(json.dumps(PACK))
    assert load_policy_pack(path).policy_id == "perinatal"
    for broken, message in [
        (
            {
                **PACK,
                "non_target_category_tags": [
                    {**PACK["non_target_category_tags"][0], "mention_pattern": "("}
                ],
            },
            "invalid regular expression",
        ),
        (
            {
                **PACK,
                "preserve_distinction_pairs": [
                    {**PACK["preserve_distinction_pairs"][0], "source": {}}
                ],
            },
            "needs codes",
        ),
        (
            {
                **PACK,
                "context_compatibility_rules": [
                    {**PACK["context_compatibility_rules"][0], "rule_id": "salt"}
                ],
            },
            "unique",
        ),
    ]:
        with pytest.raises(ConfigurationError, match=message):
            load_policy_pack(broken)
    with pytest.raises(ConfigurationError, match="cannot read policy pack"):
        load_policy_pack(tmp_path / "missing.yaml")
    with pytest.raises(ConfigurationError, match="exactly one"):
        make_pipeline(
            tmp_path / "both",
            profile=_with_policy(PACK).model_copy(
                update={
                    "checks": [
                        _with_policy(PACK)
                        .checks[1]
                        .model_copy(update={"options": {"pack": PACK, "path": str(path)}})
                    ]
                }
            ),
        )


def test_selectors_match_codes_vocabulary_and_labels() -> None:
    from biosemalign.schemas.decision import SelectedConcept
    from biosemalign.validation.policy_dsl import ConceptSelector

    concept = SelectedConcept(concept_id="1101", label="ExampleAlpha sodium", vocabulary="RxNorm")
    assert ConceptSelector(codes=["1101"]).matches(concept)
    assert not ConceptSelector(codes=["1101"], vocabulary="UNII").matches(concept)
    assert not ConceptSelector(codes=["9"]).matches(concept)
    assert ConceptSelector(codes=["1101"], label_pattern="SODIUM").matches(concept)


# -- dense retrieval and rank fusion ------------------------------------------


@pytest.fixture
def bundle(tmp_path: Path) -> Iterator[tuple[Path, Any]]:
    yield build_bundle(tmp_path)


def test_embedding_index_is_bound_to_bundle_and_encoder(
    bundle: tuple[Path, Any], tmp_path: Path
) -> None:
    path, manifest = bundle
    built = build_embedding_index(path, HashingEncoder(64), name="hash64")
    assert built.vectors == manifest.row_counts["synonyms"]
    store = LexiconStore(path)
    index = EmbeddingIndex(store, "hash64", HashingEncoder(64))
    assert index.search("ExampelAlpha", ["RxNorm"], 3)[0].concept_code == "1001"
    assert index.search("!!", ["RxNorm"], 3) == []
    assert index.search(ALPHA, ["SNOMED"], 3) == []
    index.close()
    with pytest.raises(LexiconError, match="not comparable"):
        EmbeddingIndex(store, "hash64", HashingEncoder(32))
    with pytest.raises(LexiconError, match="no embedding index"):
        EmbeddingIndex(store, "other", HashingEncoder(64))
    store.close()
    with pytest.raises(LexiconError, match="already exists"):
        build_embedding_index(path, HashingEncoder(64), name="hash64")
    with pytest.raises(LexiconError, match="must match"):
        build_embedding_index(path, HashingEncoder(64), name="bad name")

    vectors = path / "embeddings-hash64.duckdb"
    vectors.write_bytes(vectors.read_bytes() + b"\0")
    store = LexiconStore(path)
    with pytest.raises(LexiconError, match="modified, or corrupted"):
        EmbeddingIndex(store, "hash64", HashingEncoder(64))
    store.close()


def test_rebuilt_bundles_invalidate_their_embedding_indexes(tmp_path: Path) -> None:
    from tests.lexicon_support import DECLARATION

    from biosemalign.knowledge.lexicon.build import build_lexicon

    path, _ = build_bundle(tmp_path)
    build_embedding_index(path, HashingEncoder(64), name="hash64")
    declaration = DECLARATION.model_copy(
        update={
            "vocabularies": [
                v.model_copy(update={"release": "next"}) for v in DECLARATION.vocabularies
            ]
        }
    )
    build_lexicon(tmp_path / "source", path, declaration=declaration, overwrite=True)
    store = LexiconStore(path)
    with pytest.raises(LexiconError, match="rebuild it"):
        EmbeddingIndex(store, "hash64", HashingEncoder(64))
    store.close()


def test_hashing_encoder_is_deterministic_and_normalized() -> None:
    encoder = HashingEncoder(32)
    first, second = encoder.encode(["ExampleAlpha", "ExampleAlpha"])
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
    assert encoder.encode([""])[0] == [0.0] * 32
    with pytest.raises(ConfigurationError, match="at least 8"):
        HashingEncoder(4)


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    def candidate(code: str, vocabulary: str = "RxNorm") -> CandidateConcept:
        return CandidateConcept(candidate_id=code, preferred_label=code, vocabulary=vocabulary)

    lexical = [candidate("a"), candidate("b"), candidate("c")]
    dense = [candidate("b"), candidate("d"), candidate("x", "UNII"), candidate("b")]
    fused = fuse_ranked([lexical, dense], maximum=5, target_vocabulary="RxNorm")
    assert [c.candidate_id for c in fused] == ["b", "a", "d", "c", "x"]
    assert [c.candidate_id for c in fuse_ranked([lexical, dense], maximum=2)] == ["b", "a"]


def test_hybrid_retrieval_through_the_pipeline(bundle: tuple[Path, Any], tmp_path: Path) -> None:
    path, manifest = bundle
    build_embedding_index(path, HashingEncoder(64), name="hash64")
    profile = lexicon_profile(
        tiers=[
            {
                "name": "hybrid",
                "match": [
                    {
                        "op": "lexical",
                        "options": {"vocabularies": ["RxNorm"], "method": "trigram"},
                    },
                    {"op": "dense", "options": {"index": "hash64", "dimension": 64}},
                ],
                "link": {"op": "standard", "options": {"merge": "rrf", "k": 30}},
            }
        ]
    )
    with lexicon_pipeline(tmp_path, path, manifest, profile=profile) as pipeline:
        artifact = pipeline.align_artifact(pipeline.make_request("ExampelAlpha"))
        operators = pipeline.kernel.workflow_provenance.tiers[0].operators
    assert artifact.decision.selected_concepts[0].concept_id == "1001"
    channels = {c.value for c in artifact.knowledge_package.candidates[0].channels}
    assert channels == {"embedding", "lexical"}
    dense = next(op for op in operators if op.op_id == "dense")
    assert dense.options["model_id"] == "hashing-trigram-64"
    assert next(op for op in operators if op.kind == "link").options == {
        "merge": "rrf",
        "concordance": False,
        "k": 30,
    }

    model_profile = lexicon_profile(
        tiers=[
            {
                "name": "dense",
                "match": [
                    {
                        "op": "dense",
                        "options": {"index": "hash64", "encoder": "sentence_transformers"},
                    }
                ],
            }
        ]
    )
    with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL_PATH"):
        lexicon_pipeline(tmp_path / "model", path, manifest, profile=model_profile)
    with pytest.raises(ConfigurationError, match="local directory"):
        lexicon_pipeline(
            tmp_path / "model2",
            path,
            manifest,
            profile=model_profile,
            embedding_model_path=tmp_path / "nowhere",
        )


def test_embed_command(bundle: tuple[Path, Any]) -> None:
    path, _ = bundle
    runner = CliRunner()
    result = runner.invoke(app, ["lexicon", "embed", str(path), "--name", "h", "--dimension", "16"])
    assert result.exit_code == 0, result.output
    assert "hashing-trigram-16 (16 dimensions)" in result.output
    again = runner.invoke(app, ["lexicon", "embed", str(path), "--name", "h"])
    assert again.exit_code == 2 and "already exists" in again.output
    missing = runner.invoke(
        app, ["lexicon", "embed", str(path), "--name", "m", "--encoder", "sentence_transformers"]
    )
    assert missing.exit_code == 2


def test_leashed_prompt_and_policy_check_use_context_view(tmp_path: Path) -> None:
    profile = load_profile("faers_drug_normalization")
    payload = profile.model_dump(mode="json")
    payload["context"]["fields"] = [
        *payload["context"]["fields"],
        {"name": "matrix", "kind": "context", "use": ["resolve"]},
        {"name": "cohort_note", "kind": "context", "use": ["check"]},
        {"name": "blinded_endpoint", "kind": "context", "use": [], "blinded": True},
    ]
    extended = TaskProfile.model_validate(payload)
    resolver, backend = _leashed(AGREEING_PRODUCT, extended)
    with make_pipeline(tmp_path / "llm", profile=extended, resolver=resolver) as pipeline:
        pipeline.align_value(
            PRODUCT,
            route="oral",
            matrix="serum",
            cohort_note="intramuscular cohort",
            blinded_endpoint="secret",
            undeclared_col="noise",
        )
    prompt = backend.messages[1]["content"]
    assert "matrix: serum" in prompt
    assert "route: oral" in prompt
    assert "cohort_note" not in prompt
    assert "secret" not in prompt
    assert "noise" not in prompt

    pack = {
        "policy_id": "ctx_use",
        "policy_version": "1.0",
        "context_compatibility_rules": [
            {
                "rule_id": "cohort_check",
                "context_field": "cohort_note",
                "context_pattern": "intramuscular",
                "incompatible_pattern": "Oral",
                "message": "check-permitted extension context triggers rule",
            },
            {
                "rule_id": "matrix_blocked",
                "context_field": "matrix",
                "context_pattern": "serum",
                "incompatible_pattern": "Oral",
                "message": "resolve-only extension must not be seen by check",
            },
        ],
    }
    pol_payload = extended.model_dump(mode="json")
    pol_payload["checks"] = ["drug.semantics", {"op": "policy", "options": {"pack": pack}}]
    with make_pipeline(
        tmp_path / "pol", profile=TaskProfile.model_validate(pol_payload)
    ) as pipeline:
        decision = pipeline.align_value(PRODUCT, cohort_note="intramuscular cohort", matrix="serum")
    codes = {f.code: f.message for f in decision.validation.findings}
    assert "[cohort_check]" in codes.get("CONTEXT_INCOMPATIBLE", "")
    assert "[matrix_blocked]" not in codes.get("CONTEXT_INCOMPATIBLE", "")
