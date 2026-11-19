"""Persistent crosswalk memo: resolve each distinct input once (v0.2).

Reported drug strings repeat heavily — a few thousand distinct strings cover
most of a FAERS quarter — so a workflow's retrieval, fusion, and proposal for
one input can be reused for every later identical input, across runs.

Reuse is exact rather than approximate. The key covers everything a proposal
depends on: the SAU's mention, context, attributes, task, and readiness (minus
the source row and occurrence identifiers), the workflow digest, and the
pinned identity of every resource that answered. Only proposal and evidence
are reused; validation and routing re-run in the kernel on every hit, so a
changed calibration or validation rule applies to memoized decisions too, and
provenance marks each reused decision with the run that first produced it.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.schemas.decision import AlignmentProposal
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.provenance import ModelProvenance, ResolutionTrace
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.storage.paths import reject_symlink_components
from biosemalign.util import stable_digest

__all__ = ["MEMO_VERSION", "CrosswalkMemo", "MemoEntry", "memo_key"]

MEMO_VERSION = "1"

#: SAU fields that identify an occurrence rather than shape its resolution.
_OCCURRENCE: dict[str, Any] = {
    "sau_id": True,
    "source": True,
    "provenance": True,
    "context": {"record_id", "neighboring_fields"},
}


class MemoEntry(BioSemAlignModel):
    """What a workflow produced for one input, before validation and routing."""

    package: KnowledgePackage
    proposal: AlignmentProposal
    trace: ResolutionTrace
    model: ModelProvenance | None = None
    run_id: str
    stored_at: datetime


def memo_key(
    sau: SemanticAlignmentUnit, *, workflow_id: str, resources: dict[str, str | None]
) -> str:
    """The identity of an input under one workflow and resource configuration."""
    return stable_digest(
        {
            "memo_version": MEMO_VERSION,
            "workflow_id": workflow_id,
            "resources": dict(sorted(resources.items())),
            "sau": sau.model_dump(mode="json", exclude=_OCCURRENCE),
        }
    )


class CrosswalkMemo:
    """An SQLite-backed memo, safe to share between threads of one process."""

    def __init__(self, path: str | Path, *, dir_mode: int = 0o700, file_mode: int = 0o600) -> None:
        self.path = Path(path)
        reject_symlink_components(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=dir_mode)
        existed = self.path.exists()
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        if not existed:
            os.chmod(self.path, file_mode)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, memo_version TEXT NOT NULL, "
            "payload TEXT NOT NULL, stored_at TEXT NOT NULL)"
        )
        self._con.commit()

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __len__(self) -> int:
        with self._lock:
            row = self._con.execute("SELECT count(*) FROM memo").fetchone()
        return int(row[0]) if row else 0

    def get(self, key: str) -> MemoEntry | None:
        with self._lock:
            row = self._con.execute(
                "SELECT payload FROM memo WHERE key = ? AND memo_version = ?", (key, MEMO_VERSION)
            ).fetchone()
        return MemoEntry.model_validate_json(row[0]) if row else None

    def put(self, key: str, entry: MemoEntry) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO memo (key, memo_version, payload, stored_at) "
                "VALUES (?, ?, ?, ?)",
                (key, MEMO_VERSION, entry.model_dump_json(), entry.stored_at.isoformat()),
            )
            self._con.commit()

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)
