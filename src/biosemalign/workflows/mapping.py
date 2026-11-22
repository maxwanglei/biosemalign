"""Mapping workflow assembly from Schema 2 task profiles (§5.4)."""

from __future__ import annotations

from biosemalign.enums import DecisionType
from biosemalign.exceptions import ConfigurationError
from biosemalign.kernel.tiers import Cascade, ParallelUnion, Tier
from biosemalign.modules import OpContext, build
from biosemalign.modules.base import Matcher, Resolver
from biosemalign.modules.registry import default_matcher_for_domain
from biosemalign.profiles.schema import OperatorConfig, TierConfig

__all__ = ["DEFAULT_TIER", "workflow_from_profile"]

#: Name of the single tier that reproduces the default mapping pipeline.
DEFAULT_TIER = "default"


def workflow_from_profile(context: OpContext) -> Cascade:
    """Assemble the mapping cascade a profile declares, or the default cascade."""
    tiers = context.profile.tiers
    if tiers is None:
        tiers = _default_tiers(context)
    return Cascade(_build_tier(config, context) for config in tiers)


def _default_tiers(context: OpContext) -> list[TierConfig]:
    local = context.knowledge is not None and context.knowledge.serves_locally(
        context.profile.target.terminology
    )
    if not local or context.knowledge is None or context.knowledge.lexicon is None:
        default_matcher = default_matcher_for_domain(context.profile.domain)
        return [TierConfig(name=DEFAULT_TIER, match=[OperatorConfig(op=default_matcher)])]

    lexicon = context.knowledge.lexicon
    exact_tier = TierConfig(
        name="exact",
        match=[OperatorConfig(op="exact")],
        pass_on=[DecisionType.NO_VALID_MAPPING],
    )
    if lexicon.supports("trigram"):
        return [
            exact_tier,
            TierConfig(
                name=DEFAULT_TIER,
                match=[OperatorConfig(op="lexical", options={"method": "trigram"})],
            ),
        ]
    if lexicon.supports("bm25"):
        return [
            exact_tier,
            TierConfig(
                name=DEFAULT_TIER,
                match=[OperatorConfig(op="lexical", options={"method": "bm25"})],
            ),
        ]
    return [TierConfig(name=DEFAULT_TIER, match=[OperatorConfig(op="exact")])]


def _build_tier(config: TierConfig, context: OpContext) -> Tier:
    matchers: list[Matcher] = [build("match", op.op, context, op.options) for op in config.match]
    if config.link is None:
        raise ConfigurationError(f"mapping tier {config.name!r} requires a linker")
    if config.resolve is None:
        raise ConfigurationError(f"mapping tier {config.name!r} requires a resolver")
    resolver: Resolver
    if context.resolver is not None and config.resolve.op == "rules" and not config.resolve.options:
        resolver = context.resolver
    else:
        resolver = build("resolve", config.resolve.op, context, config.resolve.options)
    expand_ops = config.expand_ops
    if len(expand_ops) > 1:
        raise ConfigurationError(
            f"mapping tier {config.name!r} supports at most one expand operator"
        )
    expander = (
        build("expand", expand_ops[0].op, context, expand_ops[0].options) if expand_ops else None
    )
    return Tier(
        name=config.name,
        cleaners=tuple(build("clean", op.op, context, op.options) for op in config.clean),
        match=matchers[0] if len(matchers) == 1 else ParallelUnion(matchers),
        expander=expander,
        linker=build("link", config.link.op, context, config.link.options),
        resolver=resolver,
        pass_on=config.pass_on,
        pass_on_review_request=config.pass_on_review_request,
    )
