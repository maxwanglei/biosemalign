"""JSONL input and output (§17.1).

JSONL holds the complete nested record — every candidate, every validator
finding, the full provenance chain. Parquet flattens for analysis; this is the
format that loses nothing.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from biosemalign.storage.paths import ensure_directory

__all__ = ["atomic_text_writer", "read_jsonl", "read_models", "write_jsonl", "write_models"]

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_models(
    path: str | Path,
    models: Iterable[BaseModel],
    *,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
) -> int:
    """Write Pydantic models as JSONL. Returns the number of records written."""
    count = 0
    with atomic_text_writer(path, dir_mode=dir_mode, file_mode=file_mode) as handle:
        for model in models:
            handle.write(model.model_dump_json(exclude_none=False))
            handle.write("\n")
            count += 1
    return count


def read_models[ModelT: BaseModel](path: str | Path, model_type: type[ModelT]) -> Iterator[ModelT]:
    """Read JSONL back into validated models."""
    for record in read_jsonl(path):
        yield model_type.model_validate(record)


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
) -> int:
    """Write plain mappings as JSONL."""
    count = 0
    with atomic_text_writer(path, dir_mode=dir_mode, file_mode=file_mode) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")
            count += 1
    return count


@contextmanager
def atomic_text_writer(
    path: str | Path, *, dir_mode: int = 0o700, file_mode: int = 0o600
) -> Iterator[TextIOWrapper]:
    """Write a private text file and replace the target only after fsync.

    The temporary file lives beside the target, making ``os.replace`` atomic on
    the destination filesystem.  A failed producer leaves any prior target
    intact and removes only its own temporary file.
    """
    target = Path(path)
    ensure_directory(target.parent, dir_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    os.fchmod(descriptor, file_mode)
    handle = os.fdopen(descriptor, "w", encoding="utf-8")
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, target)
        os.chmod(target, file_mode)
        _fsync_directory(target.parent)
    except BaseException:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:  # pragma: no cover - unsupported by some non-POSIX filesystems
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read JSONL, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
