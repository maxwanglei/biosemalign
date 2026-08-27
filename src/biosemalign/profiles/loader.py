"""Task-profile discovery and loading.

Search order, first hit wins:

1. An explicit path passed by the caller.
2. ``BIOSEMALIGN_PROFILE_PATH``, if set.
3. ``./configs/profiles/`` relative to the working directory.
4. Profiles packaged inside the distribution.

Packaged defaults come last so a project can override one without forking, and
exist at all so ``BioSemAlign.from_profile("faers_drug_normalization")`` works
from a bare ``pip install`` with no repository checkout — which is the situation
inside the All of Us Workbench and on HPC login nodes.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from biosemalign.exceptions import ProfileError
from biosemalign.profiles.schema import TaskProfile
from biosemalign.settings import get_settings

__all__ = ["available_profiles", "load_profile", "load_profile_file"]

_PACKAGED = "biosemalign.profiles.defaults"


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    settings = get_settings()
    if settings.profile_path is not None:
        dirs.append(settings.profile_path)
    dirs.append(Path.cwd() / "configs" / "profiles")
    return dirs


def load_profile_file(path: str | Path) -> TaskProfile:
    """Load and validate a profile from an explicit path."""
    path = Path(path)
    if not path.is_file():
        raise ProfileError(f"profile file not found: {path}")
    return _parse(path.read_text(encoding="utf-8"), origin=str(path))


def load_profile(name_or_path: str | Path) -> TaskProfile:
    """Load a profile by name, or by path if the argument points at a file."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_file():
        return load_profile_file(candidate)

    name = str(name_or_path)
    for directory in _search_dirs():
        for suffix in (".yaml", ".yml"):
            path = directory / f"{name}{suffix}"
            if path.is_file():
                return load_profile_file(path)

    packaged = resources.files(_PACKAGED).joinpath(f"{name}.yaml")
    if packaged.is_file():
        return _parse(packaged.read_text(encoding="utf-8"), origin=f"{_PACKAGED}/{name}.yaml")

    raise ProfileError(
        f"unknown profile {name!r}. Available packaged profiles: "
        f"{', '.join(available_profiles()) or '(none)'}"
    )


def available_profiles() -> list[str]:
    """Names of every profile discoverable on the current search path."""
    names: set[str] = set()
    for directory in _search_dirs():
        if directory.is_dir():
            names.update(p.stem for p in directory.iterdir() if p.suffix in {".yaml", ".yml"})
    for entry in resources.files(_PACKAGED).iterdir():
        if entry.name.endswith(".yaml"):
            names.add(entry.name.removesuffix(".yaml"))
    return sorted(names)


def _parse(text: str, *, origin: str) -> TaskProfile:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ProfileError(f"{origin} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProfileError(f"{origin} must contain a YAML mapping at the top level")

    try:
        return TaskProfile.model_validate(raw)
    except ValidationError as exc:
        raise ProfileError(f"{origin} is not a valid task profile:\n{exc}") from exc
