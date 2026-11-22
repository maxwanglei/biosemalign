"""Resolve module: deterministic rules, concordant, and candidate-constrained LLM resolvers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from biosemalign.exceptions import ConfigurationError
from biosemalign.modules.base import OpContext
from biosemalign.modules.model import ModelService
from biosemalign.modules.registry import register
from biosemalign.modules.resolve.backend import GenerationConfig, LLMBackend
from biosemalign.modules.resolve.concordant import ConcordantResolver
from biosemalign.modules.resolve.deterministic import RESOLVER_VERSION, DeterministicResolver
from biosemalign.modules.resolve.leashed import (
    DEFAULT_MAX_CANDIDATES,
    PROMPT_VERSION,
    LeashedLLMResolver,
)
from biosemalign.modules.resolve.llm_backend import OpenAICompatibleBackend
from biosemalign.modules.resolve.mock_backend import MockLLMBackend, MockResolver
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.settings import Settings
from biosemalign.storage.cache import ResponseCache

__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "PROMPT_VERSION",
    "RESOLVER_VERSION",
    "ConcordantResolver",
    "DeterministicResolver",
    "GenerationConfig",
    "LLMBackend",
    "LeashedLLMResolver",
    "MockLLMBackend",
    "MockResolver",
    "OpenAICompatibleBackend",
    "build_llm_resolver",
]


class _RulesOptions(BioSemAlignModel):
    pass


class _ConcordantOptions(BioSemAlignModel):
    authority_sources: list[str] = Field(
        default_factory=list,
        description="Sources whose single-source exact matches are trusted, e.g. ['RxNorm'].",
    )
    minimum_mention_length: int = Field(
        default=5,
        ge=1,
        description="Shorter mentions (acronyms such as 'MTX') always pass to the next tier.",
    )


class _LLMOptions(BioSemAlignModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_candidates: int = Field(default=DEFAULT_MAX_CANDIDATES, ge=1, le=50)
    prompt_version: str = PROMPT_VERSION
    seed: int | None = 0
    examples_path: str | None = None


def build_llm_resolver(
    profile: TaskProfile,
    *,
    settings: Settings,
    cache: ResponseCache,
    model_service: ModelService | None = None,
    temperature: float = 0.0,
    max_candidates: int | None = None,
    prompt_version: str | None = None,
    seed: int | None = 0,
    examples_path: str | None = None,
) -> LeashedLLMResolver:
    """Construct the leashed LLM resolver routed through :class:`ModelService`."""
    service = model_service or ModelService.from_settings(settings, cache)
    return LeashedLLMResolver(
        profile,
        service,
        model_name=service.model_name,
        model_revision=service.model_revision,
        max_candidates=max_candidates if max_candidates is not None else DEFAULT_MAX_CANDIDATES,
        temperature=temperature,
        prompt_version=prompt_version,
        seed=seed,
        examples_path=examples_path,
    )


def _rules_factory(context: OpContext, options: Mapping[str, Any]) -> DeterministicResolver:
    _RulesOptions.model_validate(options)
    if isinstance(context.resolver, DeterministicResolver):
        return context.resolver
    return DeterministicResolver(context.profile)


def _concordant_factory(context: OpContext, options: Mapping[str, Any]) -> ConcordantResolver:
    parsed = _ConcordantOptions.model_validate(options)
    rules = (
        context.resolver
        if isinstance(context.resolver, DeterministicResolver)
        else DeterministicResolver(context.profile)
    )
    return ConcordantResolver(
        rules,
        context.profile,
        authority_sources=parsed.authority_sources,
        minimum_mention_length=parsed.minimum_mention_length,
    )


def _llm_factory(context: OpContext, options: Mapping[str, Any]) -> LeashedLLMResolver:
    parsed = _LLMOptions.model_validate(options)
    if context.model_service is not None:
        return LeashedLLMResolver(
            context.profile,
            context.model_service,
            max_candidates=parsed.max_candidates,
            temperature=parsed.temperature,
            prompt_version=parsed.prompt_version,
            seed=parsed.seed,
            examples_path=parsed.examples_path,
        )
    if context.knowledge is None:
        raise ConfigurationError("operator 'llm' needs the pipeline's knowledge router")
    if context.settings is None:
        raise ConfigurationError("operator 'llm' needs the pipeline's settings")
    return build_llm_resolver(
        context.profile,
        settings=context.settings,
        cache=context.knowledge.cache,
        temperature=parsed.temperature,
        max_candidates=parsed.max_candidates,
        prompt_version=parsed.prompt_version,
        seed=parsed.seed,
        examples_path=parsed.examples_path,
    )


register("resolve", "rules", _rules_factory)
register("resolve", "concordant", _concordant_factory)
register("resolve", "llm", _llm_factory)
