"""Package version, in its own module.

Kept separate from ``biosemalign/__init__`` so that internal modules can stamp
provenance with the version without importing the partially-initialized
top-level package.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("biosemalign")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
