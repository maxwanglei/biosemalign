"""Single-doorway model service, prompt templates, and backend registry (§5.10, G9).

Every model-backed operator — in ``clean``, ``match``, ``expand``, ``link``,
``resolve``, ``select``, or ``check`` — calls language models exclusively through
:class:`ModelService`. Direct transport calls from operators are forbidden by
architecture tests.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from biosemalign.exceptions import AdjudicationFailure, ConfigurationError
from biosemalign.input.context import context_view
from biosemalign.modules.resolve.backend import GenerationConfig, LLMBackend
from biosemalign.modules.resolve.llm_backend import OpenAICompatibleBackend
from biosemalign.plugins import ensure_plugins_loaded, load_domain
from biosemalign.profiles.schema import ContextUse, TaskProfile
from biosemalign.schemas.knowledge import CandidateConcept
from biosemalign.schemas.provenance import ModelCallRecord
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.settings import Settings
from biosemalign.storage.cache import ResponseCache
from biosemalign.util import stable_digest

__all__ = [
    "BackendFactory",
    "ModelCallResult",
    "ModelService",
    "PromptTemplate",
    "available_llm_backends",
    "build_llm_backend",
    "capture_model_calls",
    "load_prompt",
    "register_llm_backend",
    "register_prompt",
    "render_declared_context",
    "render_numbered_candidates",
    "unregister_llm_backend",
    "unregister_prompt",
]

_ACTIVE_SINKS: contextvars.ContextVar[dict[int, list[ModelCallRecord]] | None] = (
    contextvars.ContextVar("biosemalign_model_call_sinks", default=None)
)


@contextmanager
def capture_model_calls(services: Sequence[ModelService]) -> Iterator[list[ModelCallRecord]]:
    """Collect :class:`ModelCallRecord`s emitted by ``services`` within this dynamic scope."""
    bucket: list[ModelCallRecord] = []
    previous = _ACTIVE_SINKS.get()
    updated = dict(previous) if previous is not None else {}
    for svc in services:
        updated[id(svc)] = bucket
    token = _ACTIVE_SINKS.set(updated)
    try:
        yield bucket
    finally:
        _ACTIVE_SINKS.reset(token)


TModel = TypeVar("TModel", bound=BaseModel)

BackendFactory = Callable[[Settings, ResponseCache, Mapping[str, Any]], LLMBackend]

_BACKEND_REGISTRY: dict[str, BackendFactory] = {}
_PROMPT_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {}


def register_llm_backend(name: str, factory: BackendFactory) -> None:
    """Register an LLM backend factory by name (default: ``openai_compatible``)."""
    _BACKEND_REGISTRY[name] = factory


def unregister_llm_backend(name: str) -> None:
    """Remove a registered LLM backend factory."""
    _BACKEND_REGISTRY.pop(name, None)


def available_llm_backends() -> list[str]:
    """Sorted names of registered LLM backends."""
    ensure_plugins_loaded()
    return sorted(_BACKEND_REGISTRY)


def build_llm_backend(
    name: str,
    *,
    settings: Settings,
    cache: ResponseCache,
    options: Mapping[str, Any] | None = None,
) -> LLMBackend:
    """Construct a registered LLM backend from settings and cache."""
    ensure_plugins_loaded()
    factory = _BACKEND_REGISTRY.get(name)
    if factory is None:
        raise ConfigurationError(
            f"no LLM backend named {name!r} is registered; available: {available_llm_backends()}"
        )
    return factory(settings, cache, dict(options or {}))


def _openai_compatible_factory(
    settings: Settings, cache: ResponseCache, options: Mapping[str, Any]
) -> OpenAICompatibleBackend:
    if not settings.llm_base_url or not settings.llm_model:
        raise ConfigurationError(
            "the 'llm' resolver needs BIOSEMALIGN_LLM_BASE_URL and BIOSEMALIGN_LLM_MODEL; "
            "use the 'rules' resolver to run the terminology-only pipeline"
        )
    if options:
        raise ConfigurationError(
            f"openai_compatible backend takes no factory options, got {sorted(options)}"
        )
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return OpenAICompatibleBackend(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        model_revision=settings.llm_model_revision,
        cache=cache,
        api_key=api_key,
        permitted_egress_classes=frozenset(settings.permitted_egress_classes),
        local_hosts=frozenset(settings.llm_local_hosts),
        timeout=settings.llm_timeout_seconds,
    )


register_llm_backend("openai_compatible", _openai_compatible_factory)


def register_prompt(task: str, domain: str, package: str, filename: str) -> None:
    """Register a domain-specific prompt template file for ``(task, domain)``."""
    _PROMPT_OVERRIDES[(task, domain.casefold())] = (package, filename)


def unregister_prompt(task: str, domain: str) -> None:
    """Remove a registered domain prompt template override."""
    _PROMPT_OVERRIDES.pop((task, domain.casefold()), None)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Prompt file plus optional worked examples loaded with a content digest."""

    name: str
    text: str
    domain: str | None = None
    examples_text: str | None = None

    @property
    def prompt_id(self) -> str:
        scope = self.domain or "core"
        return f"{self.name}:{scope}"

    @property
    def digest(self) -> str:
        return stable_digest(
            {"name": self.name, "text": self.text, "examples": self.examples_text},
            length=12,
        )


def load_prompt(
    task: str,
    *,
    domain: str | None = None,
    examples_path: str | Path | None = None,
) -> PromptTemplate:
    """Load the prompt template for ``task``, applying a domain override when registered."""
    override: tuple[str, str] | None = None
    resolved_domain: str | None = None
    if domain:
        key = (task, domain.casefold())
        override = _PROMPT_OVERRIDES.get(key)
        if override is None:
            ensure_plugins_loaded()
            load_domain(domain)
            override = _PROMPT_OVERRIDES.get(key)
        if override is not None:
            resolved_domain = domain.casefold()

    if override is not None:
        package, filename = override
        try:
            text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        except (OSError, FileNotFoundError, ModuleNotFoundError) as exc:
            raise ConfigurationError(
                f"cannot load prompt template {filename!r} from {package!r}: {exc}"
            ) from exc
    else:
        filename = f"{task}.txt"
        try:
            text = (
                resources.files("biosemalign.modules.prompts")
                .joinpath(filename)
                .read_text(encoding="utf-8")
            )
        except (OSError, FileNotFoundError, ModuleNotFoundError) as exc:
            raise ConfigurationError(
                f"no core prompt template for task {task!r} ({filename}): {exc}"
            ) from exc

    examples_text: str | None = None
    if examples_path is not None:
        path = Path(examples_path)
        try:
            examples_text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigurationError(f"cannot read worked examples file {path}: {exc}") from exc

    return PromptTemplate(
        name=task,
        text=text.rstrip("\n"),
        domain=resolved_domain,
        examples_text=examples_text,
    )


def render_declared_context(
    sau: SemanticAlignmentUnit,
    purpose: ContextUse,
    profile: TaskProfile,
) -> list[str]:
    """Render declared context lines for ``purpose`` from ``context_view`` only (M5)."""
    return [
        f"  {key}: {value}" for key, value in sorted(context_view(sau, purpose, profile).items())
    ]


def render_numbered_candidates(candidates: Sequence[CandidateConcept]) -> list[str]:
    """Render numbered candidates (or expansion members) using one shared format (§5.10)."""
    lines: list[str] = []
    for number, candidate in enumerate(candidates, start=1):
        details = [candidate.concept_class or "concept", candidate.status.value]
        entry = (
            f"{number}. {candidate.preferred_label} "
            f"[{candidate.vocabulary} {candidate.candidate_id}; {', '.join(details)}]"
        )
        if candidate.synonyms:
            entry += f"; also: {'; '.join(candidate.synonyms[:3])}"
        related = [
            f"{s.predicate}: {s.object_label or s.object_id}" for s in candidate.relationships[:5]
        ]
        if related:
            entry += f"; {'; '.join(related)}"
        lines.append(entry)
    return lines


@dataclass(frozen=True, slots=True)
class ModelCallResult[TModel: BaseModel]:
    """Result of one call through :class:`ModelService` plus its typed provenance record."""

    reply: TModel
    record: ModelCallRecord
    record_index: int


class ModelService:
    """The single doorway for every model call across all workflow modules (§5.10, G9)."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        model_name: str,
        model_revision: str | None = None,
        owns_backend: bool = False,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.model_revision = model_revision
        self._owns_backend = owns_backend
        self.calls = 0
        self.cache_hits = 0
        self._records: list[ModelCallRecord] = []
        self._last_record: ModelCallRecord | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        cache: ResponseCache,
        *,
        backend_name: str = "openai_compatible",
        options: Mapping[str, Any] | None = None,
    ) -> ModelService:
        """Build a service backed by ``settings`` and ``cache``."""
        backend = build_llm_backend(backend_name, settings=settings, cache=cache, options=options)
        assert settings.llm_model is not None
        return cls(
            backend=backend,
            model_name=settings.llm_model,
            model_revision=settings.llm_model_revision,
            owns_backend=True,
        )

    @property
    def backend_name(self) -> str:
        return self.backend.backend_name

    def _active_sink(self) -> list[ModelCallRecord] | None:
        sinks = _ACTIVE_SINKS.get()
        return sinks.get(id(self)) if sinks is not None else None

    @property
    def record_count(self) -> int:
        sink = self._active_sink()
        return len(sink) if sink is not None else 0

    def records_since(self, index: int) -> list[ModelCallRecord]:
        sink = self._active_sink()
        return list(sink[index:]) if sink is not None else []

    def update_last_record(self, **updates: Any) -> ModelCallRecord | None:
        """Update the most recent call record in the active sink (or standalone)."""
        sink = self._active_sink()
        if sink:
            updated = sink[-1].model_copy(update=updates)
            sink[-1] = updated
            self._last_record = updated
            return updated
        if self._last_record is not None:
            self._last_record = self._last_record.model_copy(update=updates)
            return self._last_record
        return None

    def update_record(self, index: int, **updates: Any) -> ModelCallRecord:
        sink = self._active_sink()
        if sink is not None and 0 <= index < len(sink):
            updated = sink[index].model_copy(update=updates)
            sink[index] = updated
            if index == len(sink) - 1:
                self._last_record = updated
            return updated
        if self._last_record is not None:
            updated = self._last_record.model_copy(update=updates)
            self._last_record = updated
            return updated
        raise IndexError("no active model call record to update")

    def reset_counters(self) -> None:
        self.calls = 0
        self.cache_hits = 0
        self._records.clear()
        self._last_record = None

    def close(self) -> None:
        if self._owns_backend:
            close_fn = getattr(self.backend, "close", None)
            if callable(close_fn):
                close_fn()

    def call(
        self,
        *,
        operator_id: str,
        operator_kind: str,
        task: str,
        messages: list[dict[str, str]],
        output_schema: type[TModel],
        generation_config: GenerationConfig | None = None,
        prompt_id: str = "",
        prompt_digest: str = "",
    ) -> ModelCallResult[TModel]:
        """Execute one schema-constrained model call with M7/M8/G9 guarantees."""
        config = generation_config or GenerationConfig(temperature=0.0, seed=0)
        resolved_prompt_id = prompt_id or task
        resolved_prompt_digest = prompt_digest or stable_digest(messages, length=12)
        request_digest = stable_digest(
            {
                "messages": messages,
                "schema": output_schema.model_json_schema(),
                "model": self.model_name,
                "model_revision": self.model_revision,
                "prompt_id": resolved_prompt_id,
                "prompt_digest": resolved_prompt_digest,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
            },
            length=16,
        )
        hits_before = int(getattr(self.backend, "cache_hits", 0))
        try:
            reply = self.backend.generate_structured(
                messages=messages,
                output_schema=output_schema,
                generation_config=config,
            )
        except AdjudicationFailure as exc:
            self.calls += 1
            failed = ModelCallRecord(
                operator_id=operator_id,
                operator_kind=operator_kind,
                task=task,
                model_name=self.model_name,
                model_revision=self.model_revision,
                prompt_id=resolved_prompt_id,
                prompt_digest=resolved_prompt_digest,
                request_digest=request_digest,
                from_cache=False,
                error=str(exc),
            )
            sink = self._active_sink()
            if sink is not None:
                sink.append(failed)
            self._last_record = failed
            raise

        hits_after = int(getattr(self.backend, "cache_hits", 0))
        from_cache = hits_after > hits_before
        self.calls += 1
        if from_cache:
            self.cache_hits += 1

        record = ModelCallRecord(
            operator_id=operator_id,
            operator_kind=operator_kind,
            task=task,
            model_name=self.model_name,
            model_revision=self.model_revision,
            prompt_id=resolved_prompt_id,
            prompt_digest=resolved_prompt_digest,
            request_digest=request_digest,
            from_cache=from_cache,
        )
        sink = self._active_sink()
        if sink is not None:
            idx = len(sink)
            sink.append(record)
        else:
            idx = -1
        self._last_record = record
        return ModelCallResult(reply=reply, record=record, record_index=idx)  # type: ignore[arg-type]
