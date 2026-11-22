"""OpenAI-compatible structured-output backend (v0.2).

Speaks the chat-completions wire format that vLLM, other local inference
servers, and hosted APIs share, with JSON-schema constrained decoding. It is a
plain HTTP client rather than a vendor SDK, so it can enforce the same rules as
the terminology providers at the point of transport:

* A prompt carries source text. Sending it to a host outside the deployment
  needs the ``sensitive_text`` egress class and HTTPS; hosts listed in
  ``llm_local_hosts`` — an enclave's own model server — are not egress.
* Redirects are refused, so a prompt is never re-sent somewhere implicitly.
* Every response is stored in the content-addressed cache under the full
  request and the pinned model revision. FROZEN runs replay those responses
  exactly and fail closed on a miss: a model is never consulted silently.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from biosemalign.enums import EgressDataClass, ResourceMode
from biosemalign.exceptions import AdjudicationFailure, ConfigurationError, ResourceNotPermitted
from biosemalign.logging import get_logger
from biosemalign.modules.resolve.backend import GenerationConfig
from biosemalign.storage.cache import CacheKey, ResponseCache
from biosemalign.util import stable_digest

__all__ = ["LLM_BACKEND_VERSION", "OpenAICompatibleBackend"]

LLM_BACKEND_VERSION = "0.1.0"

log = get_logger(__name__)


class OpenAICompatibleBackend:
    """Structured generation against an OpenAI-compatible endpoint."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cache: ResponseCache,
        model_revision: str | None = None,
        api_key: str | None = None,
        permitted_egress_classes: frozenset[EgressDataClass] = frozenset(),
        local_hosts: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"}),
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        endpoint = httpx.URL(base_url.rstrip("/"))
        if not endpoint.is_absolute_url or endpoint.scheme not in {"http", "https"}:
            raise ConfigurationError("llm_base_url must be an absolute HTTP(S) endpoint")
        if endpoint.userinfo:
            raise ConfigurationError("llm_base_url must not contain credentials")
        if cache.mode is ResourceMode.FROZEN and not model_revision:
            raise ConfigurationError(
                "FROZEN mode replays model responses under a pinned identity; set "
                "BIOSEMALIGN_LLM_MODEL_REVISION"
            )
        self.base_url = str(endpoint)
        self.model = model
        self.model_revision = model_revision
        self.cache = cache
        self._local = endpoint.host in local_hosts
        self._https = endpoint.scheme == "https"
        self._permitted = permitted_egress_classes
        headers = {"User-Agent": "BioSemAlign/0.2 (research)"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=timeout, headers=headers)
        self._owns_client = client is None
        self.cache_hits = 0
        self.network_calls = 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
        generation_config: GenerationConfig,
    ) -> BaseModel:
        """Return the model's reply, decoded against ``output_schema``.

        A reply that does not conform raises :class:`AdjudicationFailure`; the
        caller decides what an unusable reply means for the decision.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        if generation_config.seed is not None:
            request["seed"] = generation_config.seed
        cache_payload: dict[str, Any] = dict(request)
        if generation_config.prompt_version is not None:
            cache_payload["prompt_version"] = generation_config.prompt_version
        key = CacheKey(
            provider="llm",
            provider_version=LLM_BACKEND_VERSION,
            resource_name=self.model,
            terminology_release=self.model_revision or self.model,
            endpoint=self.base_url,
            operation="chat.completions",
            query=stable_digest(cache_payload),
        )
        entry, cached = self.cache.get_or_fetch_entry(
            key,
            lambda: self._post(request),
            metadata={"model": self.model, "model_revision": self.model_revision},
        )
        if cached:
            self.cache_hits += 1
        content = entry.payload.get("content") if isinstance(entry.payload, dict) else None
        if not isinstance(content, str):
            raise AdjudicationFailure("model reply carried no message content")
        try:
            return output_schema.model_validate_json(content)
        except ValidationError as exc:
            raise AdjudicationFailure(
                f"model reply does not conform to {output_schema.__name__}: "
                f"{exc.error_count()} error(s)"
            ) from exc

    def request_digest(self, messages: list[dict[str, str]]) -> str:
        """A short, non-reversible identity for the prompt, for provenance."""
        return stable_digest(messages, length=16)

    def _post(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._local:
            if EgressDataClass.SENSITIVE_TEXT not in self._permitted:
                raise ResourceNotPermitted(
                    "the model endpoint is outside the deployment boundary and prompts carry "
                    "source text; permit the 'sensitive_text' egress class or list the host in "
                    "llm_local_hosts. Request refused before transport."
                )
            if not self._https:
                raise ResourceNotPermitted(
                    "prompts carry source text and require HTTPS outside the deployment "
                    "boundary; request refused before transport"
                )
        self.network_calls += 1
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions", json=request, follow_redirects=False
            )
        except httpx.TransportError as exc:
            raise AdjudicationFailure("model endpoint transport failed; prompt redacted") from exc
        if 300 <= response.status_code < 400:
            raise AdjudicationFailure("model endpoint returned a redirect, which is refused")
        if response.status_code >= 400:
            raise AdjudicationFailure(
                f"model endpoint returned HTTP {response.status_code}; prompt redacted"
            )
        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise AdjudicationFailure("model endpoint returned an unreadable reply") from exc
        log.debug("llm.completed", model=self.model, finish_reason=choice.get("finish_reason"))
        return {"content": content, "finish_reason": choice.get("finish_reason")}
