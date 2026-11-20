"""Hermetic defaults: isolated configuration and zero external sockets."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from biosemalign import BioSemAlign
from biosemalign.schemas.artifact import AlignmentArtifact
from biosemalign.settings import Settings, reset_settings_cache
from biosemalign.storage.cache import ResponseCache
from tests.support import PRODUCT, make_pipeline, seed_synthetic_cache
from tests.support import synthetic_settings as build_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in os.environ:
        if name.startswith("BIOSEMALIGN_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BIOSEMALIGN_MODE", "FROZEN")
    monkeypatch.setenv("BIOSEMALIGN_CACHE_DIR", str(tmp_path / "default-cache"))
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def deny_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network"):
        return

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("external network access is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def synthetic_cache(tmp_path: Path) -> ResponseCache:
    return seed_synthetic_cache(tmp_path / "synthetic-cache")


@pytest.fixture
def synthetic_settings(synthetic_cache: ResponseCache) -> Settings:
    return build_settings(synthetic_cache.root)


@pytest.fixture
def synthetic_pipeline(tmp_path: Path) -> Iterator[BioSemAlign]:
    with make_pipeline(tmp_path / "pipeline-cache") as pipeline:
        yield pipeline


@pytest.fixture
def artifact(synthetic_pipeline: BioSemAlign) -> AlignmentArtifact:
    return synthetic_pipeline.align_artifact(synthetic_pipeline.make_request(PRODUCT, route="oral"))
