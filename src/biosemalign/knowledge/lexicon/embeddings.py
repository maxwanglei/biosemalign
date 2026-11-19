"""Dense embedding indexes over a lexicon bundle (v0.2).

Lexical retrieval misses names that share no characters — a brand and its
generic, a trade name in another language. An embedding index places every
name of a bundle in a vector space where a model has learned such
equivalences, and retrieves by cosine similarity.

An index is a separate file beside the bundle, ``embeddings-<name>.duckdb``,
with its own manifest binding it to the bundle digest and to the encoder's
weights digest. A query with a different model, or against a rebuilt bundle,
is refused rather than silently compared across incompatible spaces. Search is
an exhaustive cosine scan inside DuckDB: no extension, no approximate index,
no network.

Two encoders ship. :class:`SentenceTransformerEncoder` loads a model from a
local directory only (Hub access is forced offline). :class:`HashingEncoder`
is a dependency-free character-trigram baseline: it tolerates misspellings
but is not semantic, and exists to exercise and benchmark the dense path.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from biosemalign._version import __version__
from biosemalign.exceptions import ConfigurationError, LexiconError
from biosemalign.knowledge.lexicon.build import (
    DUCKDB_CONFIG,
    file_sha256,
    require_duckdb,
    trigrams,
)
from biosemalign.knowledge.providers.lexicon import LexiconHit, LexiconStore
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.util import stable_digest

__all__ = [
    "EMBEDDING_INDEX_VERSION",
    "EmbeddingIndex",
    "EmbeddingManifest",
    "HashingEncoder",
    "SentenceTransformerEncoder",
    "TextEncoder",
    "build_embedding_index",
]

EMBEDDING_INDEX_VERSION = "1"

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_BATCH = 512


@runtime_checkable
class TextEncoder(Protocol):
    """Maps texts to fixed-length vectors."""

    model_id: str
    weights_sha256: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEncoder:
    """Signed feature hashing of character trigrams, L2-normalized."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ConfigurationError("hashing encoder dimension must be at least 8")
        self.dimension = dimension
        self.model_id = f"hashing-trigram-{dimension}"
        self.weights_sha256 = stable_digest({"encoder": "hashing-trigram", "dimension": dimension})

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for gram in trigrams(text):
                digest = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest())
                vector[digest % self.dimension] += 1.0 if digest >> 63 else -1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class SentenceTransformerEncoder:
    """A sentence-transformers model loaded from a local directory only."""

    def __init__(self, path: str | Path, *, batch_size: int = 64) -> None:
        directory = Path(path)
        if not directory.is_dir():
            raise ConfigurationError(
                f"embedding model {directory} must be a local directory; models are never "
                "downloaded"
            )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise ConfigurationError(
                "dense retrieval with a model needs biosemalign[embeddings]"
            ) from exc
        self._model: Any = SentenceTransformer(str(directory))  # pragma: no cover
        self.batch_size = batch_size  # pragma: no cover
        self.model_id = directory.name  # pragma: no cover
        self.dimension = int(self._model.get_sentence_embedding_dimension())  # pragma: no cover
        self.weights_sha256 = _directory_digest(directory)  # pragma: no cover

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


def _directory_digest(directory: Path) -> str:  # pragma: no cover - needs a real model
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


class EmbeddingManifest(BioSemAlignModel):
    """Identity and integrity record of one embedding index."""

    index_version: str
    name: str
    model_id: str
    weights_sha256: str
    dimension: int = Field(ge=1)
    bundle_sha256: str
    vectors: int = Field(ge=0)
    index_file: str
    index_sha256: str
    builder: str
    built_at: datetime


def _paths(bundle: Path, name: str) -> tuple[Path, Path]:
    if not _NAME.match(name):
        raise LexiconError(f"embedding index name {name!r} must match {_NAME.pattern}")
    return bundle / f"embeddings-{name}.duckdb", bundle / f"embeddings-{name}.json"


def build_embedding_index(
    bundle: str | Path,
    encoder: TextEncoder,
    *,
    name: str,
    overwrite: bool = False,
) -> EmbeddingManifest:
    """Encode every name of a verified bundle into a new embedding index."""
    duckdb = require_duckdb()
    directory = Path(bundle)
    # Opened only to verify the bundle; DuckDB allows one configuration per file
    # in a process, so it is closed before the names are streamed.
    store = LexiconStore(directory)
    bundle_sha256, index_file = store.bundle_sha256, store.manifest.index_file
    store.close()
    index_path, manifest_path = _paths(directory, name)
    if manifest_path.exists() and not overwrite:
        raise LexiconError(f"embedding index {name!r} already exists; pass overwrite to replace it")
    partial = index_path.with_name(index_path.name + ".partial")
    partial.unlink(missing_ok=True)
    import pyarrow as pa

    source = duckdb.connect(
        str(directory / index_file),
        read_only=True,
        config={**DUCKDB_CONFIG, "enable_external_access": False},
    )
    target = duckdb.connect(str(partial), config=DUCKDB_CONFIG)
    count = 0
    try:
        target.execute(
            "CREATE TABLE vectors (synonym_id INTEGER, concept_key INTEGER, source_key INTEGER, "
            f"vector FLOAT[{encoder.dimension}])"
        )
        reader = source.execute(
            "SELECT s.synonym_id, s.synonym, c.concept_key, k.source_key FROM synonyms s "
            "JOIN concepts c USING (vocabulary_id, concept_code) JOIN sources k USING (source) "
            "ORDER BY s.synonym_id"
        ).to_arrow_reader(_BATCH)
        for batch in reader:
            ids, texts, concepts, sources = (column.to_pylist() for column in batch.columns)
            vectors = encoder.encode(texts)
            if any(len(vector) != encoder.dimension for vector in vectors):
                raise LexiconError(f"encoder returned vectors not of length {encoder.dimension}")
            target.register(
                "vector_batch",
                pa.table(
                    {
                        "synonym_id": pa.array(ids, pa.int32()),
                        "concept_key": pa.array(concepts, pa.int32()),
                        "source_key": pa.array(sources, pa.int32()),
                        "vector": pa.array(vectors, pa.list_(pa.float32(), encoder.dimension)),
                    }
                ),
            )
            target.execute("INSERT INTO vectors SELECT * FROM vector_batch")
            target.unregister("vector_batch")
            count += len(ids)
        target.execute("CHECKPOINT")
    finally:
        target.close()
        source.close()
    index_sha256 = file_sha256(partial)
    os.replace(partial, index_path)
    manifest = EmbeddingManifest(
        index_version=EMBEDDING_INDEX_VERSION,
        name=name,
        model_id=encoder.model_id,
        weights_sha256=encoder.weights_sha256,
        dimension=encoder.dimension,
        bundle_sha256=bundle_sha256,
        vectors=count,
        index_file=index_path.name,
        index_sha256=index_sha256,
        builder=f"biosemalign {__version__}",
        built_at=datetime.now(UTC),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


class EmbeddingIndex:
    """A verified, read-only embedding index for one bundle and one encoder."""

    def __init__(self, store: LexiconStore, name: str, encoder: TextEncoder) -> None:
        duckdb = require_duckdb()
        index_path, manifest_path = _paths(store.path, name)
        if not manifest_path.is_file():
            raise LexiconError(
                f"bundle has no embedding index {name!r}; build it with biosemalign lexicon embed"
            )
        try:
            manifest = EmbeddingManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise LexiconError(f"embedding manifest {manifest_path} is invalid: {exc}") from exc
        if manifest.bundle_sha256 != store.bundle_sha256:
            raise LexiconError(
                f"embedding index {name!r} was built for bundle {manifest.bundle_sha256[:12]}, "
                f"not {store.bundle_sha256[:12]}; rebuild it"
            )
        if (manifest.weights_sha256, manifest.dimension) != (
            encoder.weights_sha256,
            encoder.dimension,
        ):
            raise LexiconError(
                f"embedding index {name!r} was built with {manifest.model_id}; vectors from "
                f"{encoder.model_id} are not comparable with it"
            )
        if not index_path.is_file() or file_sha256(index_path) != manifest.index_sha256:
            raise LexiconError(f"embedding index {index_path} is missing, modified, or corrupted")
        self.store = store
        self.encoder = encoder
        self.manifest = manifest
        self._con = duckdb.connect(
            str(index_path),
            read_only=True,
            config={**DUCKDB_CONFIG, "enable_external_access": False},
        )

    def close(self) -> None:
        self._con.close()

    def search(self, text: str, vocabularies: Iterable[str], top_k: int) -> list[LexiconHit]:
        """The most similar concepts, one hit per concept, by its nearest name."""
        vector = self.encoder.encode([text])[0]
        if not any(vector):
            return []
        ranges = self.store.key_ranges(vocabularies)
        if not ranges:
            return []
        where = " OR ".join("concept_key BETWEEN ? AND ?" for _ in ranges)
        rows = self._con.execute(
            "SELECT concept_key, synonym_id, source_key, score FROM ("
            "SELECT concept_key, synonym_id, source_key, "
            f"array_cosine_similarity(vector, ?::FLOAT[{self.manifest.dimension}]) AS score, "
            "row_number() OVER (PARTITION BY concept_key ORDER BY "
            f"array_cosine_similarity(vector, ?::FLOAT[{self.manifest.dimension}]) DESC, "
            f"synonym_id) AS position FROM vectors WHERE {where}"
            ") WHERE position = 1 ORDER BY score DESC, concept_key LIMIT ?",
            [vector, vector, *(bound for pair in ranges for bound in pair), top_k],
        ).fetchall()
        hits: list[LexiconHit] = []
        for concept_key, synonym_id, source_key, score in rows:
            vocabulary, code = self.store.concept_for_key(int(concept_key))
            hits.append(
                LexiconHit(
                    vocabulary,
                    code,
                    None,
                    self.store.source_name(int(source_key)),
                    int(synonym_id),
                    round(float(score), 6),
                )
            )
        return hits
