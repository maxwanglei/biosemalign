"""Target terminology specs: built-ins, loading, and lookup (v0.4).

A profile names its target spec — a built-in id such as ``rxnorm``, or a path
to a YAML file — or leaves it to be found by its target terminology. A
terminology with no spec gets an empty one: identity mapping works; named levels
require a spec. Built-in domain packs register their specs via
:func:`register_target_spec`.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

from biosemalign.exceptions import ProfileError
from biosemalign.plugins import ensure_plugins_loaded
from biosemalign.targets.spec import GranularitySpec, LossRule, SaltSpec, TargetSpec

__all__ = [
    "GranularitySpec",
    "LossRule",
    "SaltSpec",
    "TargetSpec",
    "builtin_specs",
    "load_target_spec",
    "register_target_spec",
    "spec_for_vocabulary",
    "unregister_target_spec",
]

_PACKAGE = "biosemalign.targets"
_REGISTERED_SPECS: dict[str, TargetSpec] = {}


def register_target_spec(spec: TargetSpec) -> None:
    """Register a target spec so profiles may reference it by ``spec_id``."""
    _REGISTERED_SPECS[spec.spec_id] = spec
    builtin_specs.cache_clear()


def unregister_target_spec(spec_id: str) -> None:
    """Remove a registered target spec, if present."""
    _REGISTERED_SPECS.pop(spec_id, None)
    builtin_specs.cache_clear()


@lru_cache(maxsize=1)
def builtin_specs() -> dict[str, TargetSpec]:
    """Every target spec registered by core or loaded plugins, by id."""
    ensure_plugins_loaded()
    specs = dict(_REGISTERED_SPECS)
    for entry in resources.files(_PACKAGE).iterdir():
        if entry.name.endswith(".yaml"):
            spec = _parse(entry.read_text(encoding="utf-8"), origin=entry.name)
            specs.setdefault(spec.spec_id, spec)
    return specs


def load_target_spec(reference: str | Path | None, *, terminology: str) -> TargetSpec:
    """The spec a profile names, or the registered one for its terminology.

    ``reference`` is a built-in/registered id or a path to a YAML file; None
    selects the registered spec whose target is ``terminology``.
    """
    if reference is None:
        return _for_terminology(terminology)
    builtin = builtin_specs().get(str(reference))
    if builtin is not None:
        return _checked(builtin, terminology)
    path = Path(reference)
    if not path.is_file():
        raise ProfileError(
            f"target spec {str(reference)!r} is neither a built-in "
            f"({sorted(builtin_specs())}) nor a file"
        )
    return _checked(_load_file(str(path.resolve())), terminology)


def spec_for_vocabulary(vocabulary: str | None) -> TargetSpec | None:
    """The registered spec whose family includes ``vocabulary``, if any."""
    if vocabulary is None:
        return None
    return next((s for s in builtin_specs().values() if s.in_family(vocabulary)), None)


@lru_cache(maxsize=32)
def _load_file(path: str) -> TargetSpec:
    return _parse(Path(path).read_text(encoding="utf-8"), origin=path)


def _for_terminology(terminology: str) -> TargetSpec:
    wanted = terminology.casefold()
    for spec in builtin_specs().values():
        if spec.target.casefold() == wanted:
            return spec
    return TargetSpec(spec_id=f"none:{terminology}", version="0", target=terminology)


def _checked(spec: TargetSpec, terminology: str) -> TargetSpec:
    if spec.target.casefold() != terminology.casefold():
        raise ProfileError(
            f"target spec {spec.label} describes {spec.target!r}, but the task targets "
            f"{terminology!r}"
        )
    return spec


def _parse(text: str, *, origin: str) -> TargetSpec:
    try:
        return TargetSpec.model_validate(yaml.safe_load(text))
    except (ValueError, yaml.YAMLError) as exc:
        raise ProfileError(f"invalid target spec {origin}: {exc}") from exc
