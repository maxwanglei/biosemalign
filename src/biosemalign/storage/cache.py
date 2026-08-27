"""Release-bound provider cache, offline fixture store, and audit index.

The cache is part of the scientific evidence chain, not merely a performance
optimization. New-format entries bind the provider call to its resource release
and endpoint, carry a checksum, and are committed atomically. ``FROZEN`` and
``OFFLINE`` are zero-network modes. ``ONLINE`` always refreshes, while
``CACHE_PREFER`` invokes ``fetch`` only when no valid entry exists.

Pre-0.2 cassettes can still be replayed in ``OFFLINE`` mode so the committed
test corpus remains usable. They are never accepted by ``FROZEN`` because the
old format did not bind an endpoint or release into the cache key.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from biosemalign.enums import ResourceMode
from biosemalign.exceptions import CacheCorruption, CacheMiss, ConfigurationError
from biosemalign.storage.paths import reject_symlink_components
from biosemalign.util import stable_digest, utc_now

try:  # pragma: no cover - fallback supports importing on non-POSIX systems
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = ["CacheEntry", "CacheKey", "ResponseCache"]

_FORMAT_VERSION = 2
_INDEX_FILENAME = "index.sqlite"
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class CacheKey:
    """The complete identity of a cacheable provider call."""

    provider: str
    provider_version: str
    operation: str
    query: str
    resource_name: str | None = None
    terminology_release: str | None = None
    endpoint: str | None = None
    filters: dict[str, Any] | None = None
    allowed_predicates: Sequence[str] | None = None
    max_hops: int | None = None

    def identity(self) -> dict[str, Any]:
        """Canonical identity included in both the digest and stored record."""
        return {
            "provider": self.provider,
            "provider_version": self.provider_version,
            "resource_name": self.resource_name,
            "terminology_release": self.terminology_release,
            "endpoint": self.endpoint,
            "operation": self.operation,
            "query": self.query,
            "filters": self.filters or {},
            "allowed_predicates": sorted(self.allowed_predicates or []),
            "max_hops": self.max_hops,
        }

    def digest(self) -> str:
        return stable_digest(self.identity())

    def legacy_digest(self) -> str:
        """Digest used by the pre-0.2 committed offline cassettes."""
        return stable_digest(
            {
                "provider": self.provider,
                "provider_version": self.provider_version,
                "operation": self.operation,
                "query": self.query,
                "filters": self.filters or {},
                "allowed_predicates": sorted(self.allowed_predicates or []),
                "max_hops": self.max_hops,
            }
        )


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    provider: str
    provider_version: str
    resource_name: str | None
    terminology_release: str | None
    endpoint: str | None
    operation: str
    query: str | None
    query_digest: str
    stored_at: str
    metadata: dict[str, Any]
    payload: Any
    checksum: str | None
    legacy: bool = False


class ResponseCache:
    """Content-addressed, checksummed store of provider responses."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        mode: ResourceMode | str = ResourceMode.FROZEN,
        retention_days: int | None = None,
        dir_mode: int = 0o700,
        file_mode: int = 0o600,
        store_raw_queries: bool = False,
        allow_legacy_offline_entries: bool = True,
    ) -> None:
        self.root = Path(cache_dir)
        self.mode = _coerce_mode(mode)
        if retention_days is not None and retention_days < 1:
            raise ConfigurationError("cache retention_days must be at least 1 when set")
        self.retention_days = retention_days
        self.dir_mode = _permission_mode(dir_mode, name="cache dir_mode")
        self.file_mode = _permission_mode(file_mode, name="cache file_mode")
        self.store_raw_queries = store_raw_queries
        self.allow_legacy_offline_entries = allow_legacy_offline_entries

    # -- paths and locking -------------------------------------------------

    def _path_for(self, key: CacheKey, digest: str) -> Path:
        if key.provider in {"", ".", ".."} or Path(key.provider).name != key.provider:
            raise ConfigurationError("cache provider name must be one safe path component")
        return self.root / key.provider / digest[:2] / f"{digest}.json"

    def _ensure_directory(self, path: Path) -> None:
        root = self.root.absolute()
        target = path.absolute()
        try:
            target.relative_to(root)
        except ValueError as exc:  # defensive: callers must stay inside the cache root
            raise ConfigurationError(f"cache directory {target} escapes cache root {root}") from exc
        reject_symlink_components(root)
        reject_symlink_components(target)

        missing: list[Path] = []
        current = target
        while not current.exists():
            missing.append(current)
            if current == root:
                break
            current = current.parent

        path.mkdir(parents=True, exist_ok=True, mode=self.dir_mode)
        reject_symlink_components(target)
        # ``Path.mkdir(parents=True, mode=...)`` only guarantees ``mode`` for
        # the final component. Apply the configured policy to every directory
        # this call created, without changing an existing broad directory if a
        # caller intentionally chose one as the cache root.
        for created in reversed(missing):
            os.chmod(created, self.dir_mode)

    @contextmanager
    def _entry_lock(self, path: Path) -> Iterator[None]:
        """Serialize writers in this process and, on POSIX, across processes."""
        self._ensure_directory(path.parent)
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_name = str(lock_path.absolute())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(lock_name, threading.RLock())

        with thread_lock:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                self.file_mode,
            )
            try:
                os.fchmod(descriptor, self.file_mode)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    # -- core operations ---------------------------------------------------

    def get_entry(self, key: CacheKey) -> CacheEntry | None:
        """Return a validated cache entry, retaining its access metadata."""
        self._validate_bound_key(key)
        digest = key.digest()
        path = self._path_for(key, digest)
        self._validate_entry_path(path)
        if path.is_file():
            return self._read_entry(path, key=key, expected_digest=digest, legacy=False)

        if self.mode is ResourceMode.OFFLINE and self.allow_legacy_offline_entries:
            legacy_digest = key.legacy_digest()
            legacy_path = self._path_for(key, legacy_digest)
            if legacy_path.is_file():
                return self._read_entry(
                    legacy_path,
                    key=key,
                    expected_digest=legacy_digest,
                    legacy=True,
                )
        return None

    def _validate_entry_path(self, path: Path) -> None:
        self._ensure_directory(path.parent)
        reject_symlink_components(path)

    def get(self, key: CacheKey) -> Any | None:
        """Return the cached payload, or ``None`` when no entry exists."""
        entry = self.get_entry(key)
        return None if entry is None else entry.payload

    def set(
        self,
        key: CacheKey,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Atomically store a checksummed payload and return its full entry."""
        digest = key.digest()
        path = self._path_for(key, digest)
        record: dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "key": digest,
            "provider": key.provider,
            "provider_version": key.provider_version,
            "resource_name": key.resource_name,
            "terminology_release": key.terminology_release,
            "endpoint": key.endpoint,
            "operation": key.operation,
            "query": key.query if self.store_raw_queries else None,
            "query_digest": stable_digest(key.query),
            "filters": key.filters or {},
            "allowed_predicates": sorted(key.allowed_predicates or []),
            "max_hops": key.max_hops,
            "stored_at": utc_now().isoformat(),
            "metadata": metadata or {},
            "payload": value,
        }
        record["checksum"] = _checksum(record)

        with self._entry_lock(path):
            self._atomic_write(path, record)
        return self._entry_from_record(record, path=path, key=key, legacy=False)

    def get_or_fetch_entry(
        self,
        key: CacheKey,
        fetch: Callable[[], Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[CacheEntry, bool]:
        """Resolve a call while preserving the entry metadata for provenance."""
        self._validate_bound_key(key)

        if self.mode is ResourceMode.ONLINE:
            return self.set(key, fetch(), metadata), False

        cached = self.get_entry(key)
        if cached is not None:
            return cached, True

        if self.mode is ResourceMode.CACHE_PREFER:
            return self.set(key, fetch(), metadata), False

        raise CacheMiss(
            f"no cache entry for {key.provider}.{key.operation} "
            f"[{key.digest()[:12]}] in {self.mode.value} mode; network access is disabled. "
            "Populate a release-bound snapshot in ONLINE/CACHE_PREFER mode first, or select the correct "
            f"cache directory: {self.root}"
        )

    def get_or_fetch(
        self,
        key: CacheKey,
        fetch: Callable[[], Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        """Backward-compatible payload-only wrapper around :meth:`get_or_fetch_entry`."""
        entry, cache_hit = self.get_or_fetch_entry(key, fetch, metadata=metadata)
        return entry.payload, cache_hit

    def _validate_bound_key(self, key: CacheKey) -> None:
        if self.mode is not ResourceMode.FROZEN:
            return
        missing = [
            name
            for name, value in (
                ("resource_name", key.resource_name),
                ("endpoint", key.endpoint),
                ("terminology_release", key.terminology_release),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "FROZEN cache calls must be release- and endpoint-bound; missing "
                + ", ".join(missing)
            )
        if key.operation != "version" and str(key.terminology_release).casefold() == "current":
            raise ConfigurationError(
                "FROZEN data calls require a concrete terminology release, not 'current'"
            )

    def _atomic_write(self, path: Path, record: dict[str, Any]) -> None:
        rendered = (
            json.dumps(
                record,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, self.file_mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, self.file_mode)
            _fsync_directory(path.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _read_entry(
        self,
        path: Path,
        *,
        key: CacheKey,
        expected_digest: str,
        legacy: bool,
    ) -> CacheEntry | None:
        self._validate_entry_path(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CacheCorruption(f"cache entry {path} is unreadable: {exc}") from exc
        entry = self._entry_from_record(record, path=path, key=key, legacy=legacy)
        if entry.key != expected_digest:
            raise CacheCorruption(
                f"cache entry {path} declares key {entry.key!r}, expected {expected_digest!r}"
            )
        if self._is_expired(entry):
            self._delete_expired_entry(path, entry)
            return None
        return entry

    def _delete_expired_entry(self, path: Path, entry: CacheEntry) -> None:
        """Delete an expired entry if it was not replaced after being read."""
        with self._entry_lock(path):
            if not path.is_file():
                return
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise CacheCorruption(
                    f"expired cache entry {path} could not be removed safely: {exc}"
                ) from exc
            if (
                str(current.get("key", path.stem)) != entry.key
                or current.get("checksum") != entry.checksum
                or str(current.get("stored_at", "")) != entry.stored_at
            ):
                return
            try:
                path.unlink()
                _fsync_directory(path.parent)
            except OSError as exc:
                raise CacheCorruption(
                    f"expired cache entry {path} could not be removed: {exc}"
                ) from exc

    def _entry_from_record(
        self,
        record: Any,
        *,
        path: Path,
        key: CacheKey | None,
        legacy: bool,
    ) -> CacheEntry:
        if not isinstance(record, dict):
            raise CacheCorruption(f"cache entry {path} must contain a JSON object")
        if "payload" not in record:
            raise CacheCorruption(f"cache entry {path} has no payload")

        checksum = record.get("checksum")
        if not legacy:
            if record.get("format_version") != _FORMAT_VERSION:
                raise CacheCorruption(f"cache entry {path} has an unsupported format version")
            if not isinstance(checksum, str):
                raise CacheCorruption(f"cache entry {path} has no checksum")
            unsigned = dict(record)
            unsigned.pop("checksum", None)
            if _checksum(unsigned) != checksum:
                raise CacheCorruption(f"cache entry {path} failed its checksum")

        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise CacheCorruption(f"cache entry {path} metadata must be an object")

        query_digest = record.get("query_digest")
        stored_query = record.get("query")
        if key is not None:
            expected_query_digest = stable_digest(key.query)
            if legacy:
                if stored_query != key.query:
                    raise CacheCorruption(f"legacy cache entry {path} has the wrong query")
                query_digest = expected_query_digest
                self._validate_legacy_identity(record, path=path, key=key)
            elif query_digest != expected_query_digest:
                raise CacheCorruption(f"cache entry {path} has the wrong query digest")
            self._validate_release(record, metadata=metadata, path=path, key=key, legacy=legacy)
            if not legacy:
                self._validate_current_identity(record, path=path, key=key)

        return CacheEntry(
            key=str(record.get("key", path.stem)),
            provider=str(record.get("provider", "unknown")),
            provider_version=str(record.get("provider_version", "unknown")),
            resource_name=(
                key.resource_name if legacy and key is not None else record.get("resource_name")
            ),
            terminology_release=(
                _metadata_release(metadata)
                if legacy
                else record.get("terminology_release") or _metadata_release(metadata)
            ),
            endpoint=key.endpoint if legacy and key is not None else record.get("endpoint"),
            operation=str(record.get("operation", "unknown")),
            query=stored_query if isinstance(stored_query, str) else None,
            query_digest=str(query_digest or stable_digest(stored_query or "")),
            stored_at=str(record.get("stored_at", "")),
            metadata=metadata,
            payload=record["payload"],
            checksum=checksum if isinstance(checksum, str) else None,
            legacy=legacy,
        )

    @staticmethod
    def _validate_current_identity(record: dict[str, Any], *, path: Path, key: CacheKey) -> None:
        expected = {
            "provider": key.provider,
            "provider_version": key.provider_version,
            "resource_name": key.resource_name,
            "endpoint": key.endpoint,
            "operation": key.operation,
            "filters": key.filters or {},
            "allowed_predicates": sorted(key.allowed_predicates or []),
            "max_hops": key.max_hops,
        }
        mismatches = [field for field, value in expected.items() if record.get(field) != value]
        if mismatches:
            raise CacheCorruption(
                f"cache entry {path} has mismatched identity fields: {', '.join(mismatches)}"
            )

    @staticmethod
    def _validate_legacy_identity(record: dict[str, Any], *, path: Path, key: CacheKey) -> None:
        expected = {
            "provider": key.provider,
            "provider_version": key.provider_version,
            "operation": key.operation,
            "filters": key.filters or {},
            "allowed_predicates": sorted(key.allowed_predicates or []),
            "max_hops": key.max_hops,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise CacheCorruption(
                    f"legacy cache entry {path} has {field}={record.get(field)!r}, expected {value!r}"
                )

    @staticmethod
    def _validate_release(
        record: dict[str, Any],
        *,
        metadata: dict[str, Any],
        path: Path,
        key: CacheKey,
        legacy: bool,
    ) -> None:
        expected = key.terminology_release
        if not expected or str(expected).casefold() == "current" or key.operation == "version":
            return
        actual = _metadata_release(metadata) if legacy else record.get("terminology_release")
        if actual is None:
            if legacy:
                return
            raise CacheCorruption(
                f"cache entry {path} has no terminology release; expected {expected!r}"
            )
        if str(actual) != str(expected):
            raise CacheCorruption(
                f"cache entry {path} is for terminology release {actual!r}, expected {expected!r}"
            )

    def _is_expired(self, entry: CacheEntry) -> bool:
        if self.retention_days is None or not entry.stored_at:
            return False
        try:
            stored_at = datetime.fromisoformat(entry.stored_at)
        except ValueError as exc:
            raise CacheCorruption(
                f"cache entry {entry.key} has an invalid stored_at timestamp"
            ) from exc
        if stored_at.tzinfo is None:
            stored_at = stored_at.replace(tzinfo=UTC)
        return utc_now() - stored_at > timedelta(days=self.retention_days)

    # -- inspection --------------------------------------------------------

    def entries(self) -> Iterator[CacheEntry]:
        """Walk and integrity-check every cached response."""
        if not self.root.is_dir():
            return
        for path in sorted(self.root.rglob("*.json")):
            reject_symlink_components(path)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise CacheCorruption(f"cache entry {path} is unreadable: {exc}") from exc
            legacy = record.get("format_version") != _FORMAT_VERSION
            yield self._entry_from_record(record, path=path, key=None, legacy=legacy)

    def purge_expired(self) -> int:
        """Delete entries older than the configured retention window."""
        if self.retention_days is None or not self.root.is_dir():
            return 0
        removed = 0
        for path in sorted(self.root.rglob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise CacheCorruption(f"cache entry {path} is unreadable: {exc}") from exc
            legacy = record.get("format_version") != _FORMAT_VERSION
            entry = self._entry_from_record(record, path=path, key=None, legacy=legacy)
            if self._is_expired(entry):
                self._delete_expired_entry(path, entry)
                removed += int(not path.exists())
        return removed

    def reindex(self) -> int:
        """Rebuild the derived SQLite audit index from payload files."""
        self._ensure_directory(self.root)
        index_path = self.root / _INDEX_FILENAME
        reject_symlink_components(index_path)
        with sqlite3.connect(index_path) as conn:
            conn.execute("DROP TABLE IF EXISTS cache_entries")
            conn.execute(
                """
                CREATE TABLE cache_entries (
                    key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    query_digest TEXT NOT NULL,
                    stored_at TEXT,
                    terminology_release TEXT,
                    endpoint TEXT,
                    checksum TEXT
                )
                """
            )
            count = 0
            for entry in self.entries():
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.key,
                        entry.provider,
                        entry.operation,
                        entry.query_digest,
                        entry.stored_at,
                        entry.terminology_release,
                        entry.endpoint,
                        entry.checksum,
                    ),
                )
                count += 1
            conn.execute("CREATE INDEX idx_provider_op ON cache_entries (provider, operation)")
        os.chmod(index_path, self.file_mode)
        return count


def _coerce_mode(mode: ResourceMode | str) -> ResourceMode:
    if isinstance(mode, ResourceMode):
        return mode
    try:
        return ResourceMode(str(mode).strip().upper())
    except ValueError as exc:
        permitted = ", ".join(member.value for member in ResourceMode)
        raise ConfigurationError(
            f"unknown resource mode {mode!r}; expected one of {permitted}. Network access refused."
        ) from exc


def _permission_mode(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 0o777:
        raise ConfigurationError(f"{name} must be an integer between 0 and 0o777")
    return value


def _checksum(record: dict[str, Any]) -> str:
    return stable_digest(record, length=64)


def _metadata_release(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("terminology_release")
    return None if value in (None, "") else str(value)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:  # pragma: no cover - not supported on every filesystem
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
