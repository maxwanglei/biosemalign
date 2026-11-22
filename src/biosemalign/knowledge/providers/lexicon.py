"""Local lexicon provider: client-compiled vocabularies, served offline (v0.2).

A :class:`LexiconStore` opens one bundle read-only after verifying it: the
index must hash to the manifest's ``index_sha256``, the manifest must hash to
its own ``bundle_sha256``, and — when the deployment pins one, which FROZEN
mode requires — the bundle digest must equal the pin. Extensions are never
installed or loaded, so opening a bundle inside an enclave makes no outbound
request.

A :class:`LexiconProvider` presents one vocabulary of the bundle through the
same :class:`KnowledgeProvider` protocol RxNav implements, so validation and
enrichment work unchanged whichever serves a vocabulary. Every read counts as
a cache hit in provenance: a release-bound local snapshot is, by definition,
never a network call.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from biosemalign.enums import Authority, ConceptStatus, ResourceMode, RetrievalChannel
from biosemalign.exceptions import ConfigurationError, LexiconError, ProviderError
from biosemalign.knowledge.lexicon.build import (
    DUCKDB_CONFIG,
    file_sha256,
    require_duckdb,
    trigrams,
)
from biosemalign.knowledge.lexicon.contract import (
    LEXICON_CONTRACT_VERSION,
    MANIFEST_FILE,
    NORMALIZER_VERSION,
    LexiconManifest,
)
from biosemalign.logging import get_logger
from biosemalign.schemas.decision import ConceptValidation
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgeStatement, RetrievalEvidence
from biosemalign.schemas.provenance import ResourceProvenance
from biosemalign.util import normalize_text, tokenize

__all__ = [
    "LEXICON_PROVIDER_VERSION",
    "LexiconEdge",
    "LexiconHit",
    "LexiconProvider",
    "LexiconStore",
]

LEXICON_PROVIDER_VERSION = "0.1.0"

#: Synonyms carried on each candidate; enough for lexical features and review
#: without copying every name a large vocabulary holds for a concept.
_MAX_SYNONYMS = 25

#: BM25 parameters (Robertson & Zaragoza's conventional defaults).
_BM25_K1 = 1.2
_BM25_B = 0.75

#: Postings one ranked query may read, rarest grams first.
_POSTINGS_BUDGET = 50_000

#: Memoized lookups per kind; the bundle is immutable, so entries never go stale.
_CACHE_ENTRIES = 100_000

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LexiconEdge:
    """One relationship of a concept inside its vocabulary family."""

    subject: str
    predicate: str
    object_vocabulary: str
    object_code: str
    assertion_source: str | None = None
    subject_vocabulary: str | None = None


@dataclass(frozen=True, slots=True)
class LexiconHit:
    """One name that matched, and the concept it names."""

    vocabulary_id: str
    concept_code: str
    synonym: str | None
    source: str
    synonym_id: int
    score: float


@dataclass(frozen=True, slots=True)
class _ConceptRow:
    vocabulary_id: str
    concept_code: str
    concept_name: str
    concept_class: str | None
    status: str
    valid_start_date: date | None
    valid_end_date: date | None
    replaced_by: str | None


class LexiconStore:
    """One verified, read-only lexicon bundle."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        require_pin: bool = False,
    ) -> None:
        duckdb = require_duckdb()
        self.path = Path(path)
        manifest_path = self.path / MANIFEST_FILE
        if not manifest_path.is_file():
            raise LexiconError(f"{self.path} is not a lexicon bundle: {MANIFEST_FILE} is missing")
        try:
            self.manifest = LexiconManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise LexiconError(f"lexicon manifest {manifest_path} is invalid: {exc}") from exc

        if self.manifest.contract_version.split(".")[0] != LEXICON_CONTRACT_VERSION.split(".")[0]:
            raise LexiconError(
                f"bundle uses lexicon contract {self.manifest.contract_version}; this release "
                f"reads {LEXICON_CONTRACT_VERSION}. Rebuild the bundle."
            )
        if self.manifest.normalizer_version != NORMALIZER_VERSION:
            raise LexiconError(
                f"bundle was normalized with normalizer {self.manifest.normalizer_version}, but "
                f"this release normalizes mentions with {NORMALIZER_VERSION}; exact matches would "
                "be missed. Rebuild the bundle."
            )
        if require_pin and expected_sha256 is None:
            raise ConfigurationError(
                "FROZEN mode requires the lexicon bundle to be pinned; set "
                f"BIOSEMALIGN_LEXICON_SHA256={self.manifest.bundle_sha256} after reviewing it"
            )
        if expected_sha256 is not None and expected_sha256 != self.manifest.bundle_sha256:
            raise LexiconError(
                f"lexicon bundle {self.manifest.bundle_sha256} does not match the pinned "
                f"BIOSEMALIGN_LEXICON_SHA256 {expected_sha256}"
            )
        index = self.path / self.manifest.index_file
        if not index.is_file():
            raise LexiconError(f"lexicon index {index} is missing")
        actual = file_sha256(index)
        if actual != self.manifest.index_sha256:
            raise LexiconError(
                f"lexicon index {index} has digest {actual}, but the manifest records "
                f"{self.manifest.index_sha256}; the bundle was modified or corrupted"
            )

        self._lock = threading.Lock()
        self._con = duckdb.connect(
            str(index),
            read_only=True,
            config={**DUCKDB_CONFIG, "enable_external_access": False},
        )
        self._vocabulary_ids = {
            v.vocabulary_id.casefold(): v.vocabulary_id for v in self.manifest.vocabularies
        }
        self._concept_cache: OrderedDict[Any, Any] = OrderedDict()
        self._names_cache: OrderedDict[Any, Any] = OrderedDict()
        self._relationship_cache: OrderedDict[Any, Any] = OrderedDict()
        self._incoming_relationship_cache: OrderedDict[Any, Any] = OrderedDict()
        self._key_cache: OrderedDict[Any, Any] = OrderedDict()
        self._predicate_cache: OrderedDict[Any, Any] = OrderedDict()
        self._corpus_cache: OrderedDict[Any, Any] = OrderedDict()
        self._code_cache: OrderedDict[Any, Any] = OrderedDict()
        # Each vocabulary owns one contiguous range of concept keys, and the
        # source table is tiny; both let ranked postings stay integer-only.
        self._key_ranges = {
            str(vocabulary): (int(low), int(high))
            for vocabulary, low, high in self._rows(
                "SELECT vocabulary_id, min(concept_key), max(concept_key) FROM concepts GROUP BY 1"
            )
        }
        self._sources = {
            int(key): str(source)
            for key, source in self._rows("SELECT source_key, source FROM sources")
        }
        # Bundles built before relationships carried a source lack the column.
        self._edge_source_column = any(
            row[0] == "assertion_source" for row in self._rows("DESCRIBE relationships")
        )

    @property
    def bundle_sha256(self) -> str:
        return self.manifest.bundle_sha256

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def vocabulary_id(self, name: str) -> str | None:
        """The bundle's spelling of a vocabulary name, or None if not served."""
        return self._vocabulary_ids.get(name.casefold())

    def release(self, vocabulary_id: str) -> str | None:
        declared = self.manifest.vocabulary(vocabulary_id)
        return declared.release if declared is not None else None

    # -- queries ---------------------------------------------------------
    #
    # DuckDB answers a single-column or full-key equality from its ART index
    # in about a millisecond, but an IN list or an extra filter falls back to
    # a scan. Lookups are therefore issued per key and memoized: the bundle is
    # immutable, and batch inputs repeat the same concepts constantly.

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._con.execute(sql, list(params)).fetchall()

    @staticmethod
    def _in(values: Sequence[str]) -> str:
        """Placeholders for an ``IN`` list over a token-sorted, indexed column."""
        return "(" + ", ".join("?" for _ in values) + ")"

    def _memoized(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
        load: Callable[[], Any],
    ) -> Any:
        with self._lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        value = load()
        with self._lock:
            cache[key] = value
            if len(cache) > _CACHE_ENTRIES:
                cache.popitem(last=False)
        return value

    def exact(self, text: str, vocabularies: Sequence[str]) -> list[LexiconHit]:
        """Names equal to ``text`` after the engine's normalization."""
        normalized = normalize_text(text)
        if not normalized:
            return []
        wanted = set(vocabularies)
        rows = self._rows(
            "SELECT vocabulary_id, concept_code, synonym, source, synonym_id FROM synonyms "
            "WHERE normalized = ?",
            [normalized],
        )
        return [
            LexiconHit(vocabulary, code, synonym, source, synonym_id, score=100.0)
            for vocabulary, code, synonym, source, synonym_id in sorted(rows)
            if vocabulary in wanted
        ]

    def supports(self, method: str) -> bool:
        return method in self.manifest.indexes

    def bm25(
        self,
        text: str,
        vocabularies: Sequence[str],
        top_k: int,
        *,
        kind: str = "bm25",
    ) -> list[LexiconHit]:
        """Best-scoring concepts by BM25 over their names, one hit per concept.

        ``kind`` selects whole-word tokens (``bm25``) or character trigrams
        (``trigram``), which tolerate misspellings. The rarest query grams are
        scored first and the most frequent are dropped once a postings budget
        is spent: a gram in a large share of all names contributes almost
        nothing to the score and would dominate the cost.
        """
        if not self.supports(kind):
            raise LexiconError(f"this lexicon bundle was built without a {kind!r} index")
        grams = Counter(tokenize(text) if kind == "bm25" else trigrams(text))
        if not grams:
            return []
        n, avgdl = self._corpus(kind)
        query = sorted(grams)
        frequencies: dict[str, int] = {
            str(token): int(df)
            for token, df in self._rows(
                f"SELECT token, df FROM {kind}_stats WHERE token IN {self._in(query)}", query
            )
        }
        chosen: list[str] = []
        spent = 0
        for token in sorted(frequencies, key=lambda t: (frequencies[t], t)):
            if chosen and spent + frequencies[token] > _POSTINGS_BUDGET:
                break
            chosen.append(token)
            spent += frequencies[token]
        if not chosen:
            return []

        ranges = self.key_ranges(vocabularies)
        scores: dict[int, float] = {}
        owners: dict[int, tuple[int, int]] = {}
        for token, synonym_id, tf, length, concept_key, source_key in self._rows(
            "SELECT token, synonym_id, tf, length, concept_key, source_key "
            f"FROM {kind}_postings WHERE token IN {self._in(chosen)}",
            chosen,
        ):
            if not any(low <= concept_key <= high for low, high in ranges):
                continue
            df = frequencies[token]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            weight = grams[token] * idf * tf * (_BM25_K1 + 1)
            weight /= tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / avgdl)
            scores[synonym_id] = scores.get(synonym_id, 0.0) + weight
            owners[synonym_id] = (concept_key, source_key)

        # One hit per concept: its best-scoring name.
        best: dict[int, tuple[float, int]] = {}
        for synonym_id, score in scores.items():
            concept_key = owners[synonym_id][0]
            current = best.get(concept_key)
            if current is None or (score, -synonym_id) > (current[0], -current[1]):
                best[concept_key] = (score, synonym_id)
        ranked = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]
        hits: list[LexiconHit] = []
        for concept_key, (score, synonym_id) in ranked:
            vocabulary, code = self.concept_for_key(concept_key)
            source = self.source_name(owners[synonym_id][1])
            hits.append(LexiconHit(vocabulary, code, None, source, synonym_id, round(score, 6)))
        return hits

    def key_ranges(self, vocabularies: Iterable[str]) -> list[tuple[int, int]]:
        """Concept-key ranges of the named vocabularies (each is contiguous)."""
        return [self._key_ranges[v] for v in vocabularies if v in self._key_ranges]

    def source_name(self, source_key: int) -> str:
        return self._sources[source_key]

    def concept_for_key(self, concept_key: int) -> tuple[str, str]:
        """``(vocabulary, code)`` of a concept key."""

        def load() -> tuple[str, str]:
            row = self._rows(
                "SELECT vocabulary_id, concept_code FROM concepts WHERE concept_key = ?",
                [concept_key],
            )[0]
            return str(row[0]), str(row[1])

        code: tuple[str, str] = self._memoized(self._code_cache, concept_key, load)
        return code

    def _corpus(self, kind: str) -> tuple[int, float]:
        def load() -> tuple[int, float]:
            row = self._rows(f"SELECT n, avgdl FROM {kind}_corpus")[0]
            return int(row[0]), float(row[1]) or 1.0

        corpus: tuple[int, float] = self._memoized(self._corpus_cache, kind, load)
        return corpus

    def concepts(self, vocabulary_id: str, codes: Iterable[str]) -> dict[str, _ConceptRow]:
        found: dict[str, _ConceptRow] = {}
        for code in sorted(set(codes)):

            def load(code: str = code) -> _ConceptRow | None:
                rows = self._rows(
                    "SELECT vocabulary_id, concept_code, concept_name, concept_class, status, "
                    "valid_start_date, valid_end_date, replaced_by FROM concepts "
                    "WHERE vocabulary_id = ? AND concept_code = ?",
                    [vocabulary_id, code],
                )
                return _ConceptRow(*rows[0]) if rows else None

            row = self._memoized(self._concept_cache, (vocabulary_id, code), load)
            if row is not None:
                found[code] = row
        return found

    def names(self, vocabulary_id: str, codes: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
        """Every (name, source) of each concept, preferred name first."""
        names: dict[str, list[tuple[str, str]]] = {}
        for code in sorted(set(codes)):

            def load(code: str = code) -> list[tuple[str, str]]:
                return [
                    (str(synonym), str(source))
                    for synonym, source in self._rows(
                        "SELECT synonym, source FROM synonyms "
                        "WHERE vocabulary_id = ? AND concept_code = ? "
                        "ORDER BY synonym_type = 'preferred' DESC, synonym_id",
                        [vocabulary_id, code],
                    )
                ]

            names[code] = self._memoized(self._names_cache, (vocabulary_id, code), load)
        return names

    def family(self, vocabulary_id: str) -> tuple[str, ...]:
        """The vocabulary first, then those its relationships may cross into."""
        return self.manifest.family(vocabulary_id)

    def relationships(
        self, vocabulary_id: str, codes: Iterable[str], predicates: Sequence[str]
    ) -> list[LexiconEdge]:
        """Outgoing edges inside the subject's vocabulary family.

        Without a declared family that is the vocabulary itself. With one —
        RxNorm and RxNorm Extension, say — an RxNorm Extension product's
        ``has_ingredient`` edge to an RxNorm ingredient is returned too, with
        the object's vocabulary, so it is never read as a code of its own.
        """
        wanted = set(predicates)
        family = list(self.family(vocabulary_id))
        source = "assertion_source" if self._edge_source_column else "NULL"
        edges: list[LexiconEdge] = []
        for code in sorted(set(codes)):

            def load(code: str = code) -> list[tuple[str, str, str, str | None]]:
                return [
                    (str(predicate), str(object_vocabulary), str(obj), asserted)
                    for predicate, object_vocabulary, obj, asserted in self._rows(
                        f"SELECT predicate, vocabulary_id_2, concept_code_2, {source} "
                        "FROM relationships WHERE vocabulary_id_1 = ? AND concept_code_1 = ? "
                        "AND list_contains(?::VARCHAR[], vocabulary_id_2) "
                        "ORDER BY predicate, vocabulary_id_2, concept_code_2",
                        [vocabulary_id, code, family],
                    )
                ]

            outgoing = self._memoized(self._relationship_cache, (vocabulary_id, code), load)
            edges.extend(
                LexiconEdge(code, predicate, object_vocabulary, obj, asserted)
                for predicate, object_vocabulary, obj, asserted in outgoing
                if predicate in wanted
            )
        return edges

    def incoming_relationships(
        self, vocabulary_id: str, codes: Iterable[str], predicates: Sequence[str]
    ) -> list[LexiconEdge]:
        """Incoming edges pointing to ``(vocabulary_id, code)`` from within its family."""
        wanted = set(predicates)
        if not wanted:
            return []
        family = list(self.family(vocabulary_id))
        source = "assertion_source" if self._edge_source_column else "NULL"
        edges: list[LexiconEdge] = []
        for code in sorted(set(codes)):

            def load(code: str = code) -> list[tuple[str, str, str, str | None]]:
                return [
                    (str(subject_vocab), str(subject_code), str(predicate), asserted)
                    for subject_vocab, subject_code, predicate, asserted in self._rows(
                        f"SELECT vocabulary_id_1, concept_code_1, predicate, {source} "
                        "FROM relationships WHERE vocabulary_id_2 = ? AND concept_code_2 = ? "
                        "AND list_contains(?::VARCHAR[], vocabulary_id_1) "
                        "ORDER BY predicate, vocabulary_id_1, concept_code_1",
                        [vocabulary_id, code, family],
                    )
                ]

            incoming = self._memoized(
                self._incoming_relationship_cache, (vocabulary_id, code), load
            )
            edges.extend(
                LexiconEdge(
                    subject=subject_code,
                    predicate=predicate,
                    object_vocabulary=vocabulary_id,
                    object_code=code,
                    assertion_source=asserted,
                    subject_vocabulary=subject_vocab,
                )
                for subject_vocab, subject_code, predicate, asserted in incoming
                if predicate in wanted
            )
        return edges

    def predicates(self, vocabulary_id: str) -> frozenset[str]:
        def load() -> frozenset[str]:
            return frozenset(
                str(row[0])
                for row in self._rows(
                    "SELECT DISTINCT predicate FROM relationships WHERE vocabulary_id_1 = ? "
                    "AND list_contains(?::VARCHAR[], vocabulary_id_2)",
                    [vocabulary_id, list(self.family(vocabulary_id))],
                )
            )

        predicates: frozenset[str] = self._memoized(self._predicate_cache, vocabulary_id, load)
        return predicates

    def keys(
        self, members: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """Equivalence keys of each ``(vocabulary, code)``, including its own identity."""
        keys: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for vocabulary, code in sorted(set(members)):

            def load(vocabulary: str = vocabulary, code: str = code) -> list[tuple[str, str]]:
                return [
                    (str(key_type), str(key_value))
                    for key_type, key_value in self._rows(
                        "SELECT key_type, key_value FROM xrefs "
                        "WHERE vocabulary_id = ? AND concept_code = ?",
                        [vocabulary, code],
                    )
                ]

            xrefs = self._memoized(self._key_cache, (vocabulary, code), load)
            keys[(vocabulary, code)] = {(vocabulary, code), *xrefs}
        return keys


class LexiconProvider:
    """One vocabulary of a lexicon bundle, behind the provider protocol."""

    provider_name = "lexicon"
    provider_version = LEXICON_PROVIDER_VERSION

    def __init__(self, store: LexiconStore, vocabulary_id: str, *, mode: ResourceMode) -> None:
        canonical = store.vocabulary_id(vocabulary_id)
        if canonical is None:
            raise ConfigurationError(f"lexicon bundle does not serve vocabulary {vocabulary_id!r}")
        self.store = store
        self.vocabulary = canonical
        self.resource_name = canonical
        self.mode = mode
        self.cache_hits = 0
        self.network_calls = 0
        self.accessed_at: datetime | None = None

    # -- protocol --------------------------------------------------------

    def terminology_release(self) -> str | None:
        return self.store.release(self.vocabulary)

    def search(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[CandidateConcept]:
        """Exact name matches, or the best BM25 matches when there are none."""
        hits = self.store.exact(query, [self.vocabulary])
        channel = RetrievalChannel.NORMALIZED_LOOKUP
        if not hits:
            kind = "trigram" if self.store.supports("trigram") else "bm25"
            hits = self.store.bm25(query, [self.vocabulary], top_k, kind=kind)
            channel = RetrievalChannel.LEXICAL
        self._touch()
        return self.candidates(hits, channel=channel, query=query)[:top_k]

    def lookup(self, concept_id: str, *, version: str | None = None) -> CandidateConcept | None:
        self._check_version(version)
        candidates = self._build([concept_id])
        return candidates.get(concept_id)

    def get_relationships(
        self,
        concept_id: str,
        *,
        predicates: list[str],
        max_hops: int = 1,
    ) -> list[KnowledgeStatement]:
        """Edges inside this vocabulary's family, followed up to ``max_hops`` deep.

        Edges to vocabularies outside the declared family stay in the bundle
        but are not returned. An edge into another family member names that
        vocabulary in ``object_vocabulary``; otherwise the object is a code of
        the subject's own vocabulary.
        """
        statements: list[KnowledgeStatement] = []
        frontier = {concept_id}
        seen = {concept_id}
        for hop in range(1, max(max_hops, 1) + 1):
            edges = self.store.relationships(self.vocabulary, frontier, predicates)
            self._touch()
            rows: dict[tuple[str, str], Any] = {}
            for vocabulary in {edge.object_vocabulary for edge in edges} | {self.vocabulary}:
                codes = {e.object_code for e in edges if e.object_vocabulary == vocabulary}
                if vocabulary == self.vocabulary:
                    codes |= frontier
                for code, row in self.store.concepts(vocabulary, codes).items():
                    rows[(vocabulary, code)] = row
            for edge in edges:
                target = rows.get((edge.object_vocabulary, edge.object_code))
                own = edge.object_vocabulary == self.vocabulary
                extensions: dict[str, Any] = {
                    "target_status": target.status if target else None,
                }
                if edge.assertion_source is not None:
                    extensions["assertion_source"] = edge.assertion_source
                if hop > 1:
                    extensions["step_subject_id"] = edge.subject
                statements.append(
                    KnowledgeStatement(
                        subject_id=concept_id if hop > 1 else edge.subject,
                        predicate=edge.predicate,
                        object_id=edge.object_code,
                        object_vocabulary=None if own else edge.object_vocabulary,
                        object_class=target.concept_class if target else None,
                        object_label=target.concept_name if target else None,
                        source_resource=self.vocabulary,
                        authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
                        direct=hop == 1,
                        extensions=extensions,
                    )
                )
            # Only the subject's own vocabulary is walked further: a code is
            # meaningful only together with its vocabulary.
            objects = {e.object_code for e in edges if e.object_vocabulary == self.vocabulary}
            frontier = objects - seen
            seen |= objects
            if not frontier:
                break
        return statements

    def validate_concept(self, concept_id: str, *, version: str | None = None) -> ConceptValidation:
        self._check_version(version)
        row = self.store.concepts(self.vocabulary, [concept_id]).get(concept_id)
        self._touch()
        if row is None:
            return ConceptValidation(
                concept_id=concept_id, exists=False, vocabulary=self.vocabulary
            )
        return ConceptValidation(
            concept_id=concept_id,
            exists=True,
            status=_status(row.status),
            vocabulary=self.vocabulary,
            concept_class=row.concept_class,
            remapped_to=[row.replaced_by] if row.replaced_by else [],
        )

    # -- helpers ---------------------------------------------------------

    def available_predicates(self) -> frozenset[str]:
        return self.store.predicates(self.vocabulary)

    def candidates(
        self,
        hits: Sequence[LexiconHit],
        *,
        channel: RetrievalChannel,
        query: str,
    ) -> list[CandidateConcept]:
        """Candidates for this vocabulary's hits, in hit order, with evidence."""
        own = [hit for hit in hits if hit.vocabulary_id == self.vocabulary]
        built = self._build([hit.concept_code for hit in own])
        ordered: list[CandidateConcept] = []
        for rank, hit in enumerate(own, start=1):
            candidate = built.get(hit.concept_code)
            if candidate is None:
                continue
            candidate.retrieval_evidence.append(
                RetrievalEvidence(
                    channel=channel,
                    query=query,
                    provider=self.provider_name,
                    score=hit.score,
                    rank=rank,
                    source_atom_id=str(hit.synonym_id),
                    source_vocabulary=hit.source,
                )
            )
            if all(existing is not candidate for existing in ordered):
                ordered.append(candidate)
        return ordered

    def candidates_for_codes(
        self,
        codes: Sequence[str],
        *,
        channel: RetrievalChannel,
        query: str,
        provider_name: str | None = None,
    ) -> list[CandidateConcept]:
        """Build candidates for explicit concept codes in caller order."""
        built = self._build(codes)
        ordered: list[CandidateConcept] = []
        seen: set[str] = set()
        for rank, code in enumerate(codes, start=1):
            if code in seen:
                continue
            candidate = built.get(code)
            if candidate is None:
                continue
            seen.add(code)
            candidate.retrieval_evidence.append(
                RetrievalEvidence(
                    channel=channel,
                    query=query,
                    provider=provider_name or self.provider_name,
                    score=100.0,
                    rank=rank,
                    source_vocabulary=self.vocabulary,
                )
            )
            ordered.append(candidate)
        return ordered

    def provenance(self) -> ResourceProvenance:
        payload: dict[str, Any] = {
            "resource_name": self.resource_name,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "terminology_release": self.terminology_release(),
            "mode": self.mode,
            "cache_hits": self.cache_hits,
            "network_calls": 0,
            "content_sha256": self.store.bundle_sha256,
        }
        if self.accessed_at is not None:
            payload["accessed_at"] = self.accessed_at
        return ResourceProvenance.model_validate(payload)

    def reset_counters(self) -> None:
        self.cache_hits = 0

    def close(self) -> None:
        """The store is shared and closed by its owner."""

    def _touch(self) -> None:
        self.cache_hits += 1
        self.accessed_at = datetime.now(UTC)

    def _check_version(self, version: str | None) -> None:
        release = self.terminology_release()
        if version is not None and version != release:
            raise ProviderError(
                f"requested {self.vocabulary} release {version!r}, but the lexicon bundle "
                f"serves {release!r}"
            )

    def _build(self, codes: Sequence[str]) -> dict[str, CandidateConcept]:
        rows = self.store.concepts(self.vocabulary, codes)
        names = self.store.names(self.vocabulary, rows)
        self._touch()
        release = self.terminology_release()
        built: dict[str, CandidateConcept] = {}
        for code, row in rows.items():
            sources = sorted({source for _, source in names.get(code, [])})
            synonyms = [name for name, _ in names.get(code, []) if name != row.concept_name][
                :_MAX_SYNONYMS
            ]
            built[code] = CandidateConcept(
                candidate_id=code,
                preferred_label=row.concept_name,
                vocabulary=self.vocabulary,
                vocabulary_version=release,
                concept_class=row.concept_class,
                synonyms=synonyms,
                status=_status(row.status),
                remapped_to=[row.replaced_by] if row.replaced_by else [],
                source_vocabularies=sources,
                extensions={
                    "valid_start_date": _iso(row.valid_start_date),
                    "valid_end_date": _iso(row.valid_end_date),
                },
            )
        return built


def _status(value: str) -> ConceptStatus:
    try:
        return ConceptStatus(value)
    except ValueError:  # pragma: no cover - the builder rejects unknown statuses
        log.warning("lexicon.unknown_status", status=value)
        return ConceptStatus.UNKNOWN


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None
