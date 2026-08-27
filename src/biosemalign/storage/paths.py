"""Symlink-safe filesystem path checks shared by private output stores."""

from __future__ import annotations

import os
from pathlib import Path

from biosemalign.exceptions import ConfigurationError

__all__ = ["ensure_directory", "reject_symlink_components"]


def reject_symlink_components(path: str | Path) -> None:
    """Reject an existing or dangling symlink anywhere in ``path``.

    This is a preflight guard against redirecting private cache/output writes
    through a pre-positioned link in a shared directory. Direct file opens use
    ``O_NOFOLLOW`` as a second guard where append semantics are required.
    """
    target = Path(path).absolute()
    for component in reversed((target, *target.parents)):
        if component.is_symlink():
            raise ConfigurationError(f"private storage path contains a symbolic link: {component}")


def ensure_directory(path: Path, mode: int) -> None:
    """Create a directory tree without following pre-existing symlinks."""
    reject_symlink_components(path)
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    reject_symlink_components(path)
    for created in reversed(missing):
        os.chmod(created, mode)
