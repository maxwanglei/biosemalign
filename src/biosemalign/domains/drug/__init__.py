"""Drug domain pack: RxNorm specs, profiles, cleaners, RxNav matcher, checks, routing, and tools."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources

import yaml

from biosemalign.domains.drug.checks import drug_coding_policies, drug_semantics_factory
from biosemalign.domains.drug.cleaners import extract_strength_factory
from biosemalign.domains.drug.providers.matcher import _rxnav_factory
from biosemalign.domains.drug.providers.rxnav import rxnav_provider_factory
from biosemalign.domains.drug.routing import drug_difficult_pattern, drug_routing_reasons
from biosemalign.domains.drug.tools.links import derive_links
from biosemalign.domains.drug.tools.omop import import_omop
from biosemalign.knowledge.lexicon import register_lexicon_tool, unregister_lexicon_tool
from biosemalign.knowledge.providers.registry import register_provider, unregister_provider
from biosemalign.modules.link.package_builder import (
    register_check_policies,
    unregister_check_policies,
)
from biosemalign.modules.model import register_prompt, unregister_prompt
from biosemalign.modules.registry import (
    register as register_operator,
)
from biosemalign.modules.registry import (
    register_domain_default_matcher,
    unregister,
    unregister_domain_default_matcher,
)
from biosemalign.profiles.loader import register_profile_package, unregister_profile_package
from biosemalign.profiles.schema import (
    ContextFieldConfig,
    register_domain_fields,
    unregister_domain_fields,
)
from biosemalign.routing.features import (
    register_difficult_pattern_hook,
    unregister_difficult_pattern_hook,
)
from biosemalign.routing.policy import register_routing_rule_hook, unregister_routing_rule_hook
from biosemalign.targets import register_target_spec, unregister_target_spec
from biosemalign.targets.spec import TargetSpec

__all__ = ["register"]


def _load_spec() -> TargetSpec:
    text = (
        resources.files("biosemalign.domains.drug.specs")
        .joinpath("rxnorm.yaml")
        .read_text(encoding="utf-8")
    )
    return TargetSpec.model_validate(yaml.safe_load(text))


def _load_fields() -> list[ContextFieldConfig]:
    text = (
        resources.files("biosemalign.domains.drug")
        .joinpath("fields.yaml")
        .read_text(encoding="utf-8")
    )
    raw = yaml.safe_load(text) or {}
    return [ContextFieldConfig.model_validate(entry) for entry in raw.get("fields", [])]


def register() -> list[Callable[[], None]]:
    """Register all drug-domain components and return cleanup callbacks."""
    spec = _load_spec()
    register_target_spec(spec)
    register_profile_package("drug", "biosemalign.domains.drug.profiles")
    register_domain_fields("drug", _load_fields())
    register_provider("rxnav", rxnav_provider_factory)
    register_operator("clean", "drug.extract_strength", extract_strength_factory)
    register_operator("match", "drug.rxnav", _rxnav_factory)
    register_operator("check", "drug.semantics", drug_semantics_factory)
    register_prompt(
        "choose_candidate", "drug", "biosemalign.domains.drug.prompts", "choose_candidate.txt"
    )
    register_check_policies("drug.semantics", drug_coding_policies)
    register_difficult_pattern_hook("drug.unresolved_salt", drug_difficult_pattern)
    register_routing_rule_hook("drug.unresolved_salt", drug_routing_reasons)
    register_domain_default_matcher("drug", "drug.rxnav")
    register_lexicon_tool("link", derive_links)
    register_lexicon_tool("import_omop", import_omop)

    return [
        lambda: unregister_lexicon_tool("import_omop"),
        lambda: unregister_lexicon_tool("link"),
        lambda: unregister_domain_default_matcher("drug"),
        lambda: unregister_routing_rule_hook("drug.unresolved_salt"),
        lambda: unregister_difficult_pattern_hook("drug.unresolved_salt"),
        lambda: unregister_check_policies("drug.semantics"),
        lambda: unregister_prompt("choose_candidate", "drug"),
        lambda: unregister("check", "drug.semantics"),
        lambda: unregister("match", "drug.rxnav"),
        lambda: unregister("clean", "drug.extract_strength"),
        lambda: unregister_provider("rxnav"),
        lambda: unregister_domain_fields("drug"),
        lambda: unregister_profile_package("drug"),
        lambda: unregister_target_spec(spec.spec_id),
    ]
