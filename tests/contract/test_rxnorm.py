"""Provider contracts against invented responses, never live RxNorm content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.support import ALPHA, ENDPOINT, HISTORICAL, SYNTHETIC_RELEASE

from biosemalign import BioSemAlign
from biosemalign.enums import ConceptStatus, EgressDataClass, ResourceMode, RetrievalChannel
from biosemalign.exceptions import (
    CacheMiss,
    ConfigurationError,
    ProviderError,
    ResourceNotPermitted,
)
from biosemalign.knowledge.providers.base import HttpJsonProvider, KnowledgeProvider, RateLimiter
from biosemalign.knowledge.providers.rxnorm import RxNavProvider, _parse_status
from biosemalign.storage.cache import ResponseCache


def test_provider_protocol_release_lookup_search_and_native_evidence(
    synthetic_pipeline: BioSemAlign,
) -> None:
    provider = synthetic_pipeline.knowledge.primary_provider()
    assert isinstance(provider, KnowledgeProvider)
    assert provider.terminology_release() == SYNTHETIC_RELEASE
    assert provider.known_terminology_release == SYNTHETIC_RELEASE
    ingredient = provider.lookup("1001", version=SYNTHETIC_RELEASE)
    assert ingredient is not None
    assert ingredient.candidate_id == "1001"
    assert ingredient.vocabulary_version == SYNTHETIC_RELEASE
    assert ingredient.status is ConceptStatus.ACTIVE
    results = provider.search(ALPHA, top_k=5)
    assert len(results) == 1
    assert results[0].channels == {RetrievalChannel.NORMALIZED_LOOKUP}
    assert provider.network_calls == 0
    assert provider.cache_hits > 0
    assert provider.mode is ResourceMode.FROZEN


def test_unknown_historical_and_remapped_concepts_are_distinct(
    synthetic_pipeline: BioSemAlign,
) -> None:
    provider = synthetic_pipeline.knowledge.primary_provider()
    assert provider.lookup("99999999") is None
    invalid = provider.validate_concept("99999999")
    assert invalid.exists is False
    historic = provider.lookup("6001")
    assert historic is not None
    assert historic.preferred_label == HISTORICAL
    assert historic.status is ConceptStatus.NOT_CURRENT
    remapped = provider.lookup("9001")
    assert remapped is not None
    assert remapped.remapped_to == ["2001"]
    validation = provider.validate_concept("9001")
    assert validation.exists
    assert validation.status is ConceptStatus.REMAPPED
    assert validation.remapped_to == ["2001"]
    assert provider.approximate(HISTORICAL, current_only=True) == []
    assert provider.approximate(HISTORICAL, current_only=False)[0]["rxcui"] == "6001"


def test_relationship_provenance_preserves_traversal_and_direct_edges(
    synthetic_pipeline: BioSemAlign,
) -> None:
    provider = synthetic_pipeline.knowledge.primary_provider()
    edges = provider.get_relationships("2001", predicates=["has_ingredient"])
    assert [(edge.subject_id, edge.predicate, edge.object_id) for edge in edges] == [
        ("2001", "has_ingredient", "1001")
    ]
    assert edges[0].direct is False
    assert edges[0].evidence_type == "rxnav_tty_traversal"
    boss = provider.get_relationships("2001", predicates=["has_boss"])
    assert boss[0].direct is True
    assert boss[0].extensions["target_tty"] == "IN"
    assert provider.get_relationships("2001", predicates=["unsupported_predicate"]) == []
    assert provider.get_relationships("2001", predicates=[]) == []


def test_frozen_provider_rejects_missing_pin_incompatible_release_and_missing_response(
    tmp_path: Path, synthetic_pipeline: BioSemAlign
) -> None:
    with pytest.raises(ConfigurationError, match="requires BIOSEMALIGN_RXNAV_TERMINOLOGY_RELEASE"):
        RxNavProvider(base_url=ENDPOINT, cache=ResponseCache(tmp_path, mode=ResourceMode.FROZEN))
    provider = synthetic_pipeline.knowledge.primary_provider()
    with pytest.raises(ProviderError, match="cannot serve requested release"):
        provider.lookup("1001", version="different-release")
    with pytest.raises(CacheMiss):
        provider.find_by_name("never seeded")
    assert provider.network_calls == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ConceptStatus.UNKNOWN),
        ("invented", ConceptStatus.UNKNOWN),
        ("Remapped", ConceptStatus.REMAPPED),
    ],
)
def test_lifecycle_status_parser(raw: str | None, expected: ConceptStatus) -> None:
    assert _parse_status(raw) is expected


def http_provider(
    tmp_path: Path,
    handler: Any,
    *,
    base_url: str = "https://synthetic.invalid",
    permitted: tuple[EgressDataClass, ...] = (EgressDataClass.PUBLIC_TERMINOLOGY,),
    attempts: int = 1,
) -> HttpJsonProvider:
    return HttpJsonProvider(
        base_url=base_url,
        cache=ResponseCache(tmp_path, mode=ResourceMode.ONLINE),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        requests_per_second=0,
        max_attempts=attempts,
        permitted_egress_classes=permitted,
    )


def test_http_transport_caching_accounting_and_lifecycle(tmp_path: Path) -> None:
    with http_provider(tmp_path, lambda _: httpx.Response(200, json={"answer": 42})) as provider:
        assert provider.get_json("example", {}, operation="lookup", query="opaque") == {
            "answer": 42
        }
        assert provider.network_calls == provider.network_operations == 1
        assert provider.last_access_metadata["cache_hit"] is False
        provider.cache.mode = ResourceMode.FROZEN
        # FROZEN requires a bound terminology key, so use CACHE_PREFER for this
        # terminology-agnostic HTTP base-class cache replay.
        provider.cache.mode = ResourceMode.CACHE_PREFER
        assert provider.get_json("example", {}, operation="lookup", query="opaque") == {
            "answer": 42
        }
        assert provider.cache_hits == 1
        provider.reset_counters()
        assert provider.network_calls == provider.network_operations == provider.cache_hits == 0
        assert provider.last_access_metadata == {}


@pytest.mark.parametrize("status", [400, 401, 404, 302])
def test_http_failure_does_not_become_no_match(tmp_path: Path, status: int) -> None:
    with http_provider(
        tmp_path, lambda _: httpx.Response(status, text="private source text")
    ) as provider:
        with pytest.raises(ProviderError) as error:
            provider.get_json("example", {}, operation="lookup", query="private source text")
        assert "private source text" not in str(error.value)
        assert provider.network_calls == 1
        assert list(provider.cache.entries()) == []


def test_http_malformed_json_is_technical_error(tmp_path: Path) -> None:
    with (
        http_provider(tmp_path, lambda _: httpx.Response(200, text="not JSON")) as provider,
        pytest.raises(ProviderError, match="non-JSON"),
    ):
        provider.get_json("example", {}, operation="lookup", query="opaque")


@pytest.mark.parametrize("status", [429, 503])
def test_transient_http_errors_are_retried_with_attempt_accounting(
    tmp_path: Path, status: int
) -> None:
    responses = iter([httpx.Response(status), httpx.Response(200, json={"ok": True})])
    with http_provider(tmp_path, lambda _: next(responses), attempts=2) as provider:
        assert provider.get_json("example", {}, operation="lookup", query="opaque") == {"ok": True}
        assert provider.network_calls == 2
        assert provider.network_operations == 1


def test_exhausted_transport_failure_remains_provider_error(tmp_path: Path) -> None:
    def failed(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic failure", request=request)

    with http_provider(tmp_path, failed) as provider:
        with pytest.raises(ProviderError, match="transport failed"):
            provider.get_json("example", {}, operation="lookup", query="opaque")
        assert provider.network_calls == 1


@pytest.mark.parametrize(
    ("base_url", "permitted"),
    [
        ("https://synthetic.invalid", (EgressDataClass.PUBLIC_TERMINOLOGY,)),
        ("http://synthetic.invalid", (EgressDataClass.SENSITIVE_TEXT,)),
    ],
)
def test_sensitive_text_requires_egress_authorization_and_encryption(
    tmp_path: Path, base_url: str, permitted: tuple[EgressDataClass, ...]
) -> None:
    with http_provider(
        tmp_path,
        lambda _: pytest.fail("transport must not run"),
        base_url=base_url,
        permitted=permitted,
    ) as provider:
        with pytest.raises(ResourceNotPermitted):
            provider.get_json(
                "example",
                {},
                operation="search",
                query="sensitive",
                egress_class=EgressDataClass.SENSITIVE_TEXT,
            )
        assert provider.network_calls == 0


@pytest.mark.parametrize(
    "base_url", ["relative", "ftp://synthetic.invalid", "https://secret:pass@synthetic.invalid"]
)
def test_invalid_http_endpoints_fail_before_client_creation(tmp_path: Path, base_url: str) -> None:
    with pytest.raises(ConfigurationError):
        HttpJsonProvider(base_url=base_url, cache=ResponseCache(tmp_path))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"max_attempts": 0},
        {"requests_per_second": -1},
        {"permitted_egress_classes": ["invalid"]},
    ],
)
def test_invalid_http_settings_fail_closed(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        HttpJsonProvider(base_url=ENDPOINT, cache=ResponseCache(tmp_path), **kwargs)


def test_rate_limiter_spaces_requests_without_wall_clock_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []
    readings = iter([0.01, 0.1])
    monkeypatch.setattr(
        "biosemalign.knowledge.providers.base.time.monotonic", lambda: next(readings)
    )
    monkeypatch.setattr("biosemalign.knowledge.providers.base.time.sleep", sleeps.append)
    RateLimiter(10).wait()
    assert sleeps == pytest.approx([0.09])
    RateLimiter(0).wait()
