"""Example external BioSemAlign plugin registered under ``biosemalign.plugins``."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources

import yaml

from biosemalign.modules.registry import register as register_operator
from biosemalign.modules.registry import unregister
from biosemalign.profiles.loader import register_profile_package, unregister_profile_package
from biosemalign.profiles.schema import (
    ContextFieldConfig,
    register_domain_fields,
    unregister_domain_fields,
)
from biosemalign.targets import register_target_spec, unregister_target_spec
from biosemalign.targets.spec import TargetSpec
from biosemalign_pk_pack.checks import specimen_check_factory
from biosemalign_pk_pack.cleaners import normalize_pk_parameter_factory

__all__ = ["register"]


def _load_spec() -> TargetSpec:
    text = (
        resources.files("biosemalign_pk_pack.specs")
        .joinpath("toypk.yaml")
        .read_text(encoding="utf-8")
    )
    return TargetSpec.model_validate(yaml.safe_load(text))


def _load_fields() -> list[ContextFieldConfig]:
    text = (
        resources.files("biosemalign_pk_pack").joinpath("fields.yaml").read_text(encoding="utf-8")
    )
    raw = yaml.safe_load(text) or {}
    return [ContextFieldConfig.model_validate(entry) for entry in raw.get("fields", [])]


def register() -> list[Callable[[], None]]:
    """Register the PK target spec, context fields, profiles, cleaner, and check."""
    spec = _load_spec()
    register_target_spec(spec)
    register_profile_package("pk", "biosemalign_pk_pack.profiles")
    register_domain_fields("pk", _load_fields())
    register_operator("clean", "pk.normalize_parameter", normalize_pk_parameter_factory)
    register_operator("check", "pk.specimen_check", specimen_check_factory)

    return [
        lambda: unregister("check", "pk.specimen_check"),
        lambda: unregister("clean", "pk.normalize_parameter"),
        lambda: unregister_domain_fields("pk"),
        lambda: unregister_profile_package("pk"),
        lambda: unregister_target_spec(spec.spec_id),
    ]
