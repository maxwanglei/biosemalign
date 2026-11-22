"""Hierarchy domain pack (tested on Appendix A ``ToyDisease``)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources

import yaml

from biosemalign.domains.hierarchy.checks import hierarchy_semantics_factory
from biosemalign.modules.registry import register as register_operator
from biosemalign.modules.registry import unregister
from biosemalign.profiles.schema import (
    ContextFieldConfig,
    register_domain_fields,
    unregister_domain_fields,
)
from biosemalign.targets import register_target_spec, unregister_target_spec
from biosemalign.targets.spec import TargetSpec

__all__ = ["register"]


def _load_spec() -> TargetSpec:
    text = (
        resources.files("biosemalign.domains.hierarchy.specs")
        .joinpath("toydisease.yaml")
        .read_text(encoding="utf-8")
    )
    return TargetSpec.model_validate(yaml.safe_load(text))


def _load_fields() -> list[ContextFieldConfig]:
    text = (
        resources.files("biosemalign.domains.hierarchy")
        .joinpath("fields.yaml")
        .read_text(encoding="utf-8")
    )
    raw = yaml.safe_load(text) or {}
    return [ContextFieldConfig.model_validate(entry) for entry in raw.get("fields", [])]


def register() -> list[Callable[[], None]]:
    """Register hierarchy specs, context fields, and checks."""
    spec = _load_spec()
    register_target_spec(spec)
    register_domain_fields("hierarchy", _load_fields())
    register_operator("check", "hierarchy.semantics", hierarchy_semantics_factory)

    return [
        lambda: unregister("check", "hierarchy.semantics"),
        lambda: unregister_domain_fields("hierarchy"),
        lambda: unregister_target_spec(spec.spec_id),
    ]
