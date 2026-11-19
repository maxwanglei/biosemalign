"""Adjudicator selection.

The profile names a backend; this resolves it. A backend that cannot be built
fails with an explanation rather than silently falling back to different
behaviour — a silent downgrade would corrupt an evaluation comparing the two.
"""

from __future__ import annotations

from biosemalign.adjudication.backend import Adjudicator
from biosemalign.adjudication.deterministic import DeterministicAdjudicator
from biosemalign.adjudication.mock_backend import MockAdjudicator
from biosemalign.exceptions import ConfigurationError
from biosemalign.profiles.schema import TaskProfile
from biosemalign.settings import Settings
from biosemalign.storage.cache import ResponseCache

__all__ = ["build_adjudicator", "build_llm_adjudicator"]


def build_adjudicator(
    profile: TaskProfile,
    *,
    settings: Settings | None = None,
    cache: ResponseCache | None = None,
) -> Adjudicator:
    """Construct the adjudicator the profile asks for."""
    backend = profile.adjudication.backend.strip().casefold()

    if backend == "deterministic":
        return DeterministicAdjudicator(profile)
    if backend == "mock":
        return MockAdjudicator()
    if backend == "vllm":
        if settings is None or cache is None:
            raise ConfigurationError(
                "the vllm adjudicator needs the pipeline's settings and response cache"
            )
        return build_llm_adjudicator(profile, settings=settings, cache=cache)
    raise ConfigurationError(
        f"unknown adjudication backend {profile.adjudication.backend!r}; "
        "expected 'deterministic', 'mock', or 'vllm'"
    )


def build_llm_adjudicator(
    profile: TaskProfile,
    *,
    settings: Settings,
    cache: ResponseCache,
    max_candidates: int | None = None,
) -> Adjudicator:
    """The leashed LLM adjudicator against the configured endpoint."""
    from biosemalign.adjudication.leashed import LeashedLLMAdjudicator
    from biosemalign.adjudication.llm_backend import OpenAICompatibleBackend

    if not settings.llm_base_url or not settings.llm_model:
        raise ConfigurationError(
            "the LLM adjudicator needs BIOSEMALIGN_LLM_BASE_URL and BIOSEMALIGN_LLM_MODEL; "
            "set adjudication.backend to 'deterministic' to run the terminology-only pipeline"
        )
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    backend = OpenAICompatibleBackend(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        model_revision=settings.llm_model_revision,
        cache=cache,
        api_key=api_key,
        permitted_egress_classes=frozenset(settings.permitted_egress_classes),
        local_hosts=frozenset(settings.llm_local_hosts),
        timeout=settings.llm_timeout_seconds,
    )
    return LeashedLLMAdjudicator(
        profile,
        backend,
        model_name=settings.llm_model,
        model_revision=settings.llm_model_revision,
        max_candidates=max_candidates or profile.adjudication.llm_max_candidates,
    )
