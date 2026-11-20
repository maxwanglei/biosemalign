from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest

from biosemalign.enums import ResourceMode
from biosemalign.exceptions import CacheCorruption, CacheMiss, ConfigurationError
from biosemalign.storage.cache import CacheKey, ResponseCache
from biosemalign.util import stable_digest, utc_now


def key(**updates):
    return replace(
        CacheKey(
            provider="synthetic",
            provider_version="1",
            operation="lookup",
            query="private source",
            resource_name="SyntheticVocabulary",
            terminology_release="synthetic-v1",
            endpoint="https://example.invalid/api",
        ),
        **updates,
    )


def entry_path(cache, cache_key):
    digest = cache_key.digest()
    return cache.root / cache_key.provider / digest[:2] / f"{digest}.json"


def rewrite_record(path, **updates):
    record = json.loads(path.read_text())
    record.update(updates)
    record.pop("checksum", None)
    record["checksum"] = stable_digest(record, length=64)
    path.write_text(json.dumps(record))


def test_release_endpoint_and_policy_are_part_of_cache_identity():
    original = key(filters={"scope": "report"}, allowed_predicates=["b", "a"], max_hops=1)
    assert original.digest() == replace(original, allowed_predicates=["a", "b"]).digest()
    for replacement in (
        replace(original, terminology_release="synthetic-v2"),
        replace(original, endpoint="https://other.invalid/api"),
        replace(original, filters={"scope": "row"}),
        replace(original, max_hops=2),
    ):
        assert replacement.digest() != original.digest()
    assert (
        original.legacy_digest()
        == replace(original, terminology_release="synthetic-v2").legacy_digest()
    )


@pytest.mark.parametrize(
    "mode, expected_calls", [(ResourceMode.CACHE_PREFER, 1), (ResourceMode.ONLINE, 2)]
)
def test_fetch_policy_and_private_replay(tmp_path, mode, expected_calls):
    cache = ResponseCache(tmp_path, mode=mode)
    calls = []

    def fetch():
        calls.append(1)
        return {"result": len(calls)}

    first, hit = cache.get_or_fetch_entry(
        key(), fetch, metadata={"terminology_release": "synthetic-v1"}
    )
    second, second_hit = cache.get_or_fetch_entry(key(), fetch)
    assert not hit
    assert second_hit is (mode is ResourceMode.CACHE_PREFER)
    assert len(calls) == expected_calls
    assert first.query is None
    assert first.query_digest == stable_digest("private source")
    assert cache.get(key()) == second.payload
    assert "private source" not in entry_path(cache, key()).read_text()
    assert entry_path(cache, key()).stat().st_mode & 0o777 == 0o600
    assert cache.root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("mode", [ResourceMode.FROZEN, ResourceMode.OFFLINE])
def test_zero_network_modes_never_fetch(tmp_path, mode):
    cache = ResponseCache(tmp_path, mode=mode)
    with pytest.raises(CacheMiss, match="network access is disabled"):
        cache.get_or_fetch(key(), lambda: pytest.fail("offline cache invoked fetch"))
    cache.set(key(), {"synthetic": True})
    assert cache.get_or_fetch(key(), lambda: pytest.fail("cache hit invoked fetch")) == (
        {"synthetic": True},
        True,
    )
    assert cache.get(replace(key(), terminology_release="synthetic-v2")) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"resource_name": None},
        {"endpoint": None},
        {"terminology_release": None},
        {"terminology_release": "current"},
    ],
)
def test_frozen_requires_concrete_release_and_endpoint(tmp_path, updates):
    with pytest.raises(ConfigurationError, match="FROZEN"):
        ResponseCache(tmp_path).get_or_fetch(key(**updates), lambda: pytest.fail("must not fetch"))


def test_version_probe_can_use_current_release_and_raw_query_is_opt_in(tmp_path):
    cache = ResponseCache(tmp_path, store_raw_queries=True)
    version_key = key(operation="version", terminology_release="current")
    cache.set(version_key, "synthetic-v1")
    assert cache.get_entry(version_key).query == "private source"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "typo"},
        {"retention_days": 0},
        {"dir_mode": True},
        {"file_mode": -1},
        {"file_mode": 0o1000},
    ],
)
def test_invalid_cache_configuration_fails_closed(tmp_path, kwargs):
    with pytest.raises(ConfigurationError):
        ResponseCache(tmp_path, **kwargs)


@pytest.mark.parametrize("provider", ["", ".", "..", "../escape", "unsafe/name"])
def test_provider_cannot_escape_cache_root(tmp_path, provider):
    with pytest.raises(ConfigurationError, match="safe path component"):
        ResponseCache(tmp_path).set(key(provider=provider), {})


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"format_version": 999}, "unsupported format"),
        ({"metadata": ["not an object"]}, "metadata must be an object"),
        ({"query_digest": "wrong"}, "wrong query digest"),
        ({"terminology_release": "synthetic-v2"}, "terminology release"),
        ({"terminology_release": None}, "no terminology release"),
        ({"provider_version": "different"}, "mismatched identity"),
        ({"key": "wrong"}, "declares key"),
    ],
)
def test_forged_or_incompatible_entry_is_rejected(tmp_path, updates, message):
    cache = ResponseCache(tmp_path)
    cache.set(key(), {"ok": True})
    rewrite_record(entry_path(cache, key()), **updates)
    with pytest.raises(CacheCorruption, match=message):
        cache.get(key())


@pytest.mark.parametrize(
    "payload, message",
    [
        ("{broken", "unreadable"),
        ("[]", "JSON object"),
        ('{"format_version":2}', "no payload"),
        ('{"format_version":2,"payload":{}}', "no checksum"),
    ],
)
def test_unreadable_or_incomplete_entries_fail(tmp_path, payload, message):
    cache = ResponseCache(tmp_path)
    cache.set(key(), {})
    entry_path(cache, key()).write_text(payload)
    with pytest.raises(CacheCorruption, match=message):
        cache.get(key())


def test_checksum_detects_payload_tampering(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.set(key(), {"observed": 1})
    path = entry_path(cache, key())
    record = json.loads(path.read_text())
    record["payload"]["observed"] = 2
    path.write_text(json.dumps(record))
    with pytest.raises(CacheCorruption, match="checksum"):
        cache.get(key())


def test_failed_fetch_and_failed_atomic_replace_preserve_prior_entry(tmp_path, monkeypatch):
    cache = ResponseCache(tmp_path, mode="ONLINE")
    cache.set(key(), {"original": True})

    def fail():
        raise RuntimeError("provider outage")

    with pytest.raises(RuntimeError, match="provider outage"):
        cache.get_or_fetch(key(), fail)
    assert cache.get(key()) == {"original": True}
    import biosemalign.storage.cache as module

    monkeypatch.setattr(
        module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure"))
    )
    with pytest.raises(OSError, match="disk failure"):
        cache.set(key(), {"replacement": True})
    assert cache.get(key()) == {"original": True}
    assert not list(tmp_path.rglob("*.tmp"))


def legacy_record(cache_key, **updates):
    record = {
        "key": cache_key.legacy_digest(),
        "provider": cache_key.provider,
        "provider_version": cache_key.provider_version,
        "operation": cache_key.operation,
        "query": cache_key.query,
        "filters": {},
        "allowed_predicates": [],
        "max_hops": None,
        "stored_at": utc_now().isoformat(),
        "metadata": {"terminology_release": "synthetic-v1"},
        "payload": {"legacy": True},
    }
    record.update(updates)
    return record


def test_legacy_is_only_replayed_in_explicit_offline_mode(tmp_path):
    cache_key = key()
    path = (
        tmp_path
        / cache_key.provider
        / cache_key.legacy_digest()[:2]
        / f"{cache_key.legacy_digest()}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(legacy_record(cache_key)))
    offline = ResponseCache(tmp_path, mode="OFFLINE")
    entry = offline.get_entry(cache_key)
    assert entry.legacy and entry.terminology_release == "synthetic-v1"
    assert entry.resource_name == cache_key.resource_name
    assert entry.endpoint == cache_key.endpoint
    assert ResponseCache(tmp_path).get(cache_key) is None
    assert (
        ResponseCache(tmp_path, mode="OFFLINE", allow_legacy_offline_entries=False).get(cache_key)
        is None
    )
    for updates, message in [
        ({"query": "wrong"}, "wrong query"),
        ({"provider_version": "wrong"}, "provider_version"),
        ({"metadata": {"terminology_release": "wrong"}}, "terminology release"),
    ]:
        path.write_text(json.dumps(legacy_record(cache_key, **updates)))
        with pytest.raises(CacheCorruption, match=message):
            offline.get(cache_key)


def test_retention_purge_and_sqlite_reindex(tmp_path):
    cache = ResponseCache(tmp_path, retention_days=2)
    old_key = key(query="old")
    cache.set(old_key, {"old": True})
    cache.set(key(), {"current": True})
    rewrite_record(
        entry_path(cache, old_key), stored_at=(utc_now() - timedelta(days=3)).isoformat()
    )
    assert cache.purge_expired() == 1
    assert cache.get(old_key) is None
    assert cache.reindex() == 1
    with closing(sqlite3.connect(tmp_path / "index.sqlite")) as conn:
        assert conn.execute(
            "SELECT terminology_release, query_digest FROM cache_entries"
        ).fetchone() == ("synthetic-v1", stable_digest("private source"))
    assert cache.reindex() == 1
    assert [entry.payload for entry in cache.entries()] == [{"current": True}]


def test_read_removes_expired_entry_and_rejects_invalid_timestamp(tmp_path):
    cache = ResponseCache(tmp_path, retention_days=1)
    cache.set(key(), {})
    rewrite_record(
        entry_path(cache, key()),
        stored_at=(utc_now() - timedelta(days=2)).replace(tzinfo=None).isoformat(),
    )
    assert cache.get(key()) is None
    cache.set(key(), {})
    rewrite_record(entry_path(cache, key()), stored_at="invalid date")
    with pytest.raises(CacheCorruption, match="stored_at"):
        cache.get(key())


def test_concurrent_writers_leave_one_complete_valid_response(tmp_path):
    cache = ResponseCache(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        entries = list(
            executor.map(
                lambda number: cache.set(key(), {"writer": number, "values": list(range(100))}),
                range(24),
            )
        )
    assert len(entries) == 24
    payload = cache.get(key())
    assert payload["writer"] in range(24)
    assert payload["values"] == list(range(100))
    assert len(list(cache.entries())) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_cache_rejects_symlink_root_and_index(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symbolic link"):
        ResponseCache(link).set(key(), {})
    cache = ResponseCache(actual)
    (actual / "index.sqlite").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ConfigurationError, match="symbolic link"):
        cache.reindex()
