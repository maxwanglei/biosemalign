#!/usr/bin/env python3
"""Reject incomplete distributions and accidental raw/restricted data leaks."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

WHEEL_REQUIRED = {
    "biosemalign/py.typed",
    "biosemalign/profiles/defaults/faers_drug_normalization.yaml",
    "biosemalign/resources/registry.yaml",
}
SDIST_REQUIRED = {
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "constraints.txt",
    "LICENSE",
    "NOTICE",
    "scripts/check_distribution.py",
    "scripts/rxnorm_release_canary.py",
    "src/biosemalign/py.typed",
    "src/biosemalign/profiles/defaults/faers_drug_normalization.yaml",
    "src/biosemalign/resources/registry.yaml",
}
FORBIDDEN_COMPONENTS = {
    ".biosemalign_cache",
    "__pycache__",
    "cassettes",
    "raw_terminology",
    "restricted",
}
FORBIDDEN_SUFFIXES = {".db", ".parquet", ".sqlite", ".sqlite3"}


def _archive_names(path: Path) -> set[str]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {name.rstrip("/") for name in archive.namelist() if name.rstrip("/")}
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            return {member.name.rstrip("/") for member in archive if member.name.rstrip("/")}
    raise ValueError(f"not a supported wheel or source archive: {path}")


def _archive_text(path: Path, member: str) -> str:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return archive.read(member).decode("utf-8")
    with tarfile.open(path, mode="r:*") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"archive member is not a file: {member}")
        return extracted.read().decode("utf-8")


def _safe_members(names: set[str]) -> list[str]:
    unsafe: list[str] = []
    for name in names:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            unsafe.append(name)
    return sorted(unsafe)


def _without_sdist_root(names: set[str]) -> set[str]:
    """Remove the conventional ``project-version/`` source-archive prefix."""
    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(roots) != 1:
        return names
    root = next(iter(roots))
    return {
        str(PurePosixPath(*PurePosixPath(name).parts[1:]))
        for name in names
        if len(PurePosixPath(name).parts) > 1
    } | ({""} if root in names else set())


def check_distribution(path: str | Path) -> list[str]:
    """Return human-readable policy violations for one built artifact."""
    artifact = Path(path)
    try:
        names = _archive_names(artifact)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return [str(exc)]

    errors = [f"unsafe archive member: {name}" for name in _safe_members(names)]
    is_wheel = artifact.suffix == ".whl"
    normalized = names if is_wheel else _without_sdist_root(names)
    required = WHEEL_REQUIRED if is_wheel else SDIST_REQUIRED
    for missing in sorted(required - normalized):
        errors.append(f"missing required file: {missing}")

    if is_wheel:
        entry_points = next(
            (name for name in normalized if name.endswith(".dist-info/entry_points.txt")),
            None,
        )
        if entry_points is None:
            errors.append("missing console-script metadata: *.dist-info/entry_points.txt")
        else:
            try:
                entry_point_text = _archive_text(artifact, entry_points)
            except (OSError, UnicodeError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
                errors.append(f"could not read console-script metadata: {exc}")
            else:
                expected_entry_point = "biosemalign = biosemalign.cli:main"
                if expected_entry_point not in entry_point_text.splitlines():
                    errors.append(f"unsafe console-script target; expected {expected_entry_point}")
        if any(name.startswith("tests/") for name in normalized):
            errors.append("wheel must not contain the test suite")

    for name in sorted(normalized):
        member = PurePosixPath(name)
        lowered_parts = {part.lower() for part in member.parts}
        if len(member.parts) >= 2 and tuple(part.lower() for part in member.parts[:2]) == (
            "tests",
            "fixtures",
        ):
            errors.append(f"forbidden test fixture path: {name}")
        if lowered_parts & FORBIDDEN_COMPONENTS:
            errors.append(f"forbidden raw/restricted path: {name}")
        if member.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden data artifact: {name}")
        if any(part.lower().startswith(".env") for part in member.parts):
            errors.append(f"forbidden environment file: {name}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = False
    for artifact in args.artifacts:
        errors = check_distribution(artifact)
        if errors:
            failed = True
            print(f"{artifact}:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"checked {artifact}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
