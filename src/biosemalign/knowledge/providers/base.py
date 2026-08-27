"""Knowledge provider interface and shared HTTP machinery (§13.1).

Every provider implements the same protocol and is expected to pass the same
contract suite (§21.2), so that adding UMLS or OMOP later cannot quietly
introduce a differently-shaped candidate.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from biosemalign.enums import EgressDataClass, ResourceMode
from biosemalign.exceptions import ConfigurationError, ProviderError, ResourceNotPermitted
from biosemalign.logging import get_logger
from biosemalign.schemas.decision import ConceptValidation
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgeStatement
from biosemalign.storage.cache import CacheKey, ResponseCache
from biosemalign.util import stable_digest

__all__ = ["HttpJsonProvider", "KnowledgeProvider", "RateLimiter"]

log = get_logger(__name__)


@runtime_checkable
class KnowledgeProvider(Protocol):
    """The contract every terminology resource adapter must satisfy."""

    provider_name: str
    provider_version: str

    def terminology_release(self) -> str | None:
        """Release identifier of the vocabulary currently being served."""
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[CandidateConcept]:
        """Retrieve candidates for a query string."""
        ...

    def lookup(self, concept_id: str, *, version: str | None = None) -> CandidateConcept | None:
        """Fetch one concept by identifier, or ``None`` if it does not exist."""
        ...

    def get_relationships(
        self,
        concept_id: str,
        *,
        predicates: list[str],
        max_hops: int = 1,
    ) -> list[KnowledgeStatement]:
        """Retrieve typed relationships, restricted to the given predicates."""
        ...

    def validate_concept(self, concept_id: str, *, version: str | None = None) -> ConceptValidation:
        """Check that an identifier exists and report its lifecycle status."""
        ...


class RateLimiter:
    """Minimal request spacer.

    RxNav publishes a per-IP cap and a batch of a hundred thousand FAERS strings
    will hit it immediately. Spacing requests is cheaper and friendlier than
    discovering the cap through 429s.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second < 0:
            raise ConfigurationError("requests_per_second must not be negative")
        self.rate = requests_per_second
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


class HttpJsonProvider:
    """Shared base for REST-backed providers: cache, throttle, retry, decode."""

    provider_name: str = "http"
    provider_version: str = "0.1.0"
    api_version: str | None = None

    def __init__(
        self,
        *,
        base_url: str,
        cache: ResponseCache,
        timeout: float = 30.0,
        max_attempts: int = 4,
        requests_per_second: float = 15.0,
        client: httpx.Client | None = None,
        resource_name: str | None = None,
        permitted_egress_classes: Iterable[EgressDataClass | str] = (
            EgressDataClass.PUBLIC_TERMINOLOGY,
        ),
    ) -> None:
        endpoint = httpx.URL(base_url.rstrip("/"))
        if not endpoint.is_absolute_url or endpoint.scheme not in {"http", "https"}:
            raise ConfigurationError("provider base_url must be an absolute HTTP(S) endpoint")
        if endpoint.userinfo:
            raise ConfigurationError("provider base_url must not contain credentials")
        if timeout <= 0:
            raise ConfigurationError("provider timeout must be positive")
        if max_attempts < 1:
            raise ConfigurationError("provider max_attempts must be at least 1")
        self.base_url = str(endpoint)
        self._endpoint_scheme = endpoint.scheme
        self.resource_name = resource_name or getattr(self, "vocabulary", self.provider_name)
        self.cache = cache
        self.max_attempts = max_attempts
        try:
            self.permitted_egress_classes = frozenset(
                item
                if isinstance(item, EgressDataClass)
                else EgressDataClass(str(item).strip().lower())
                for item in permitted_egress_classes
            )
        except ValueError as exc:
            raise ConfigurationError(f"unknown provider egress data class: {exc}") from exc
        self._limiter = RateLimiter(requests_per_second)
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": "BioSemAlign/0.1 (research; +https://github.com/maxwanglei/biosemalign)"
            },
            follow_redirects=True,
        )
        self._owns_client = client is None
        self._counter_lock = threading.Lock()
        self.cache_hits = 0
        # ``network_calls`` means actual HTTP attempts, including retries.  The
        # old implementation incremented once around the entire retry loop and
        # therefore understated provider load and outage severity.
        self.network_calls = 0
        self.network_operations = 0
        self.last_access_metadata: dict[str, Any] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpJsonProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def reset_counters(self) -> None:
        """Start a fresh run-level provider accounting window."""
        with self._counter_lock:
            self.cache_hits = 0
            self.network_calls = 0
            self.network_operations = 0
            self.last_access_metadata = {}

    def _increment(self, field: str) -> None:
        with self._counter_lock:
            setattr(self, field, getattr(self, field) + 1)

    def _cache_terminology_release(self, operation: str) -> str | None:
        """Release identity for a cache key; concrete providers override this."""
        return None

    # -- request plumbing --------------------------------------------------

    def _get_json_uncached(self, path: str, params: dict[str, Any]) -> Any:
        """Perform one GET, retrying transient failures."""

        @retry(
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )
        def _do() -> Any:
            self._limiter.wait()
            self._increment("network_calls")
            response = self._client.get(
                f"{self.base_url}/{path.lstrip('/')}",
                params=params,
                # Provider requests can contain sensitive text in the query
                # string. Never let a service redirect it to a downgrade or a
                # different origin implicitly.
                follow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                raise ProviderError(
                    f"{self.provider_name} returned a redirect for {path}; automatic "
                    "provider redirects are refused to protect request egress"
                )
            # 4xx other than 429 are not worth retrying: the request is wrong,
            # not unlucky.
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            if response.status_code >= 400:
                raise ProviderError(
                    f"{self.provider_name} returned HTTP {response.status_code} for {path}; "
                    "request parameters and response body were redacted"
                )
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                # RxNav answers an unsupported parameter with a plain-text
                # error page rather than JSON, which is how the invalid
                # `has_active_moiety` predicate surfaces.
                raise ProviderError(
                    f"{self.provider_name} returned non-JSON for {path}; "
                    "request parameters and response body were redacted"
                ) from exc

        self._increment("network_operations")
        try:
            return _do()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.provider_name} returned transient HTTP "
                f"{exc.response.status_code} for {path} after {self.max_attempts} attempt(s); "
                "request parameters and response body were redacted"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"{self.provider_name} transport failed for {path} after "
                f"{self.max_attempts} attempt(s); request details were redacted"
            ) from exc

    def _get_json_authorized(
        self,
        path: str,
        params: dict[str, Any],
        *,
        egress_class: EgressDataClass,
        query: str,
    ) -> Any:
        """Enforce privacy policy at the point immediately before transport."""
        if egress_class not in self.permitted_egress_classes:
            raise ResourceNotPermitted(
                f"{self.provider_name}.{path} requires {egress_class.value!r} egress, which "
                "is not permitted; request refused before transport "
                f"[query_digest={stable_digest(query, length=12)}]"
            )
        if (
            egress_class in {EgressDataClass.SENSITIVE_TEXT, EgressDataClass.DIRECT_IDENTIFIER}
            and self._endpoint_scheme != "https"
        ):
            raise ResourceNotPermitted(
                f"{self.provider_name}.{path} requires encrypted HTTPS transport for "
                f"{egress_class.value!r} egress; request refused before transport "
                f"[query_digest={stable_digest(query, length=12)}]"
            )
        return self._get_json_uncached(path, params)

    def get_json(
        self,
        path: str,
        params: dict[str, Any],
        *,
        operation: str,
        query: str,
        filters: dict[str, Any] | None = None,
        allowed_predicates: list[str] | None = None,
        max_hops: int | None = None,
        metadata: dict[str, Any] | None = None,
        egress_class: EgressDataClass = EgressDataClass.PUBLIC_TERMINOLOGY,
    ) -> Any:
        """Perform a cached GET under the active resource mode."""
        release = self._cache_terminology_release(operation)
        key = CacheKey(
            provider=self.provider_name,
            provider_version=self.provider_version,
            resource_name=self.resource_name,
            terminology_release=release,
            endpoint=self.base_url,
            operation=operation,
            query=query,
            filters=filters,
            allowed_predicates=allowed_predicates,
            max_hops=max_hops,
        )
        entry, was_cached = self.cache.get_or_fetch_entry(
            key,
            lambda: self._get_json_authorized(
                path,
                params,
                egress_class=egress_class,
                query=query,
            ),
            metadata=metadata,
        )
        if was_cached:
            self._increment("cache_hits")
        self.last_access_metadata = {
            **entry.metadata,
            "cache_stored_at": entry.stored_at,
            "cache_checksum": entry.checksum,
            "cache_legacy": entry.legacy,
            "cache_hit": was_cached,
            "endpoint": entry.endpoint,
            "terminology_release": (
                entry.terminology_release or entry.metadata.get("terminology_release")
            ),
        }
        if self.api_version is not None:
            self.last_access_metadata.setdefault("api_version", self.api_version)
        log.debug(
            "provider.request",
            provider=self.provider_name,
            operation=operation,
            query_digest=stable_digest(query, length=12),
            cache_key=entry.key[:12],
            cache_hit=was_cached,
            mode=self.cache.mode.value,
            egress_class=egress_class.value,
        )
        return entry.payload

    @property
    def mode(self) -> ResourceMode:
        return self.cache.mode
