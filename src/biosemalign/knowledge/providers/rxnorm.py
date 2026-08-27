"""RxNorm provider, backed by the NLM RxNav REST API.

Three behaviours here were established by probing the live API and are not
obvious from the RxNav documentation:

1. ``approximateTerm`` searches obsolete and non-RxNorm source atoms by
   default. ``TOPROL XL 50 MG`` resolves to RxCUI 220348 — an MMSL atom named
   "PROPRIETARY" with status ``NotCurrent``, whose ``properties`` and
   ``related`` endpoints both return empty. Passing ``option=1`` restricts the
   search to current concepts and yields RxCUI 866438, the correct branded
   drug. The two searches are therefore separate channels, and a candidate
   found only by the historical channel is flagged rather than mixed in.

2. ``related.json?rela=`` returns only *directly asserted* edges, while
   ``related.json?tty=`` returns the *transitive* closure. An SBD has no direct
   ``has_ingredient`` edge — the real path is SBD → SCDC → IN. Enrichment uses
   the TTY form because it answers the question in one call, and every
   statement it produces is marked ``direct=False`` so the distinction survives
   into the Knowledge Package (§6.7).

3. ``has_active_moiety`` is not a valid RxNav predicate; the API answers with a
   plain-text "Path or Query Parameter error". The RxNorm relationship carrying
   active-moiety semantics is ``has_boss``.
"""

from __future__ import annotations

from typing import Any

from biosemalign.enums import (
    Authority,
    ConceptStatus,
    EgressDataClass,
    ResourceMode,
    RetrievalChannel,
)
from biosemalign.exceptions import CacheCorruption, CacheMiss, ConfigurationError, ProviderError
from biosemalign.knowledge.providers.base import HttpJsonProvider
from biosemalign.logging import get_logger
from biosemalign.schemas.decision import ConceptValidation
from biosemalign.schemas.knowledge import CandidateConcept, KnowledgeStatement, RetrievalEvidence

__all__ = [
    "PREDICATE_TO_TTY",
    "RELA_ONLY_PREDICATES",
    "TTY_TO_PREDICATE",
    "RxNavProvider",
]

log = get_logger(__name__)

#: RxNorm term types used for drug normalization, mapped to the RxNorm
#: relationship name that describes reaching them. Only TTYs with a faithful
#: RxNorm predicate are listed; inventing official-looking predicate names for
#: the rest would misrepresent what the vocabulary actually asserts.
#:
#: These are fetched through ``related.json?tty=``, which walks the transitive
#: closure — one request reaches an ingredient from a branded drug even though
#: RxNorm asserts no direct edge between them.
TTY_TO_PREDICATE: dict[str, str] = {
    "IN": "has_ingredient",
    "PIN": "has_precise_ingredient",
    "MIN": "has_ingredients",
    "BN": "has_tradename",
    "DF": "has_dose_form",
    "SCDC": "consists_of",
}

PREDICATE_TO_TTY: dict[str, str] = {v: k for k, v in TTY_TO_PREDICATE.items()}

#: Predicates RxNav serves only through ``related.json?rela=``. ``BOSS`` is a
#: real RxNorm term type but ``related.json?tty=BOSS`` returns HTTP 400, while
#: ``rela=has_boss`` is accepted — so active-moiety information is reachable,
#: just through the other endpoint. Edges from this path are directly asserted,
#: so they are recorded with ``direct=True``.
RELA_ONLY_PREDICATES: frozenset[str] = frozenset({"has_boss", "boss_of"})

_STATUS_MAP: dict[str, ConceptStatus] = {
    "active": ConceptStatus.ACTIVE,
    "notcurrent": ConceptStatus.NOT_CURRENT,
    "remapped": ConceptStatus.REMAPPED,
    "quantified": ConceptStatus.QUANTIFIED,
    "obsolete": ConceptStatus.OBSOLETE,
    "never active": ConceptStatus.NEVER_ACTIVE,
}


def _parse_status(raw: str | None) -> ConceptStatus:
    if not raw:
        return ConceptStatus.UNKNOWN
    return _STATUS_MAP.get(raw.strip().casefold(), ConceptStatus.UNKNOWN)


class RxNavProvider(HttpJsonProvider):
    """Candidate retrieval, enrichment, and validation against RxNorm."""

    provider_name = "rxnav"
    provider_version = "0.1.0"
    vocabulary = "RxNorm"

    def __init__(self, *, expected_release: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.cache.mode is ResourceMode.FROZEN and not expected_release:
            self.close()
            raise ConfigurationError(
                "FROZEN RxNorm access requires BIOSEMALIGN_RXNAV_TERMINOLOGY_RELEASE; "
                "pin the release used to populate this cache snapshot"
            )
        self._expected_release = expected_release
        # A FROZEN run trusts only its configured snapshot identity and never
        # resolves the mutable ``current`` version endpoint. ONLINE still
        # verifies an expected release against the endpoint before data calls.
        self._release: str | None = (
            expected_release if self.cache.mode is ResourceMode.FROZEN else None
        )

    # -- release -----------------------------------------------------------

    def terminology_release(self) -> str | None:
        """The RxNorm release date, pinned into provenance for every decision."""
        if self._release is None:
            try:
                payload = self.get_json("version.json", {}, operation="version", query="current")
                release = (
                    str(payload.get("version")) if payload and payload.get("version") else None
                )
                if release is None:
                    raise ProviderError(
                        "RxNav version response did not contain a release identifier"
                    )
                if self._expected_release is not None and release != self._expected_release:
                    raise ProviderError(
                        f"RxNav served release {release!r}, expected {self._expected_release!r}"
                    )
                self._release = release
            except (CacheMiss, CacheCorruption):
                raise
            except ProviderError as exc:
                log.warning("rxnav.version_unavailable", error=str(exc))
                raise
        return self._release

    @property
    def known_terminology_release(self) -> str | None:
        """Return already-known release identity without cache or network I/O."""
        return self._release or self._expected_release

    def _metadata(self) -> dict[str, Any]:
        return {
            "terminology_release": self.terminology_release(),
            "vocabulary": self.vocabulary,
        }

    def _cache_terminology_release(self, operation: str) -> str | None:
        if operation == "version":
            return self._expected_release or "current"
        return self.terminology_release()

    def _validated_release(self, requested: str | None) -> str:
        release = self.terminology_release()
        if release is None:  # defensive: RxNav's implementation now raises instead
            raise ProviderError("RxNorm release could not be established")
        if requested is not None and requested != release:
            raise ProviderError(
                f"RxNav cannot serve requested release {requested!r}; active snapshot is {release!r}"
            )
        return release

    # -- retrieval channels ------------------------------------------------

    def find_by_name(self, name: str) -> list[str]:
        """Exact then normalized lookup (``search=2``).

        Returns bare RxCUIs. This channel fails on strength-bearing reported
        names such as "TOPROL XL 50 MG", which is expected and is why the
        approximate channels exist.
        """
        payload = self.get_json(
            "rxcui.json",
            {"name": name, "search": 2},
            operation="find_by_name",
            query=name,
            metadata=self._metadata(),
            egress_class=EgressDataClass.SENSITIVE_TEXT,
        )
        return list((payload or {}).get("idGroup", {}).get("rxnormId", []) or [])

    def approximate(
        self, term: str, *, max_entries: int = 10, current_only: bool = True
    ) -> list[dict[str, Any]]:
        """RxNav approximate matching.

        ``current_only`` maps to ``option=1``. Leaving it off searches obsolete
        and non-RxNorm source atoms, which is occasionally the only way to
        resolve a withdrawn product named in a historical FAERS report — but
        those hits cannot be enriched and must never be silently treated as
        current concepts.
        """
        option = 1 if current_only else 0
        payload = self.get_json(
            "approximateTerm.json",
            {"term": term, "maxEntries": max_entries, "option": option},
            operation="approximate_current" if current_only else "approximate_historical",
            query=term,
            filters={"maxEntries": max_entries, "option": option},
            metadata=self._metadata(),
            egress_class=EgressDataClass.SENSITIVE_TEXT,
        )
        candidates = (payload or {}).get("approximateGroup", {}).get("candidate") or []
        return [c for c in candidates if c.get("rxcui")]

    def spelling_suggestions(self, name: str) -> list[str]:
        """Alternative spellings, used as a last-resort retrieval channel."""
        payload = self.get_json(
            "spellingsuggestions.json",
            {"name": name},
            operation="spelling_suggestions",
            query=name,
            metadata=self._metadata(),
            egress_class=EgressDataClass.SENSITIVE_TEXT,
        )
        group = (payload or {}).get("suggestionGroup", {}) or {}
        suggestions = (group.get("suggestionList") or {}).get("suggestion") or []
        return list(suggestions)

    # -- concept access ----------------------------------------------------

    def _properties(self, rxcui: str) -> dict[str, Any]:
        payload = self.get_json(
            f"rxcui/{rxcui}/properties.json",
            {},
            operation="properties",
            query=rxcui,
            metadata=self._metadata(),
        )
        return (payload or {}).get("properties", {}) or {}

    def _history_status(self, rxcui: str) -> dict[str, Any]:
        payload = self.get_json(
            f"rxcui/{rxcui}/historystatus.json",
            {},
            operation="history_status",
            query=rxcui,
            metadata=self._metadata(),
        )
        return (payload or {}).get("rxcuiStatusHistory", {}) or {}

    def lookup(self, concept_id: str, *, version: str | None = None) -> CandidateConcept | None:
        """Fetch one RxNorm concept.

        Falls back to ``historystatus`` when ``properties`` is empty. That is
        not a defensive nicety: an obsolete concept returns ``{}`` from
        ``properties`` while ``historystatus`` still carries its name, term
        type, and source, and dropping it here is precisely how a dead-end
        candidate becomes invisible.
        """
        release = self._validated_release(version)
        properties = self._properties(concept_id)
        history = self._history_status(concept_id)
        meta = history.get("metaData", {}) or {}
        attributes = history.get("attributes", {}) or {}

        name = properties.get("name") or attributes.get("name")
        if not name:
            return None

        tty = properties.get("tty") or attributes.get("tty")
        status = _parse_status(meta.get("status"))
        source = meta.get("source")

        remapped = [
            str(c.get("remappedRxCui"))
            for c in (history.get("derivedConcepts", {}) or {}).get("remappedConcept", []) or []
            if c.get("remappedRxCui")
        ]

        synonyms = [s for s in (properties.get("synonym"),) if s]

        return CandidateConcept(
            candidate_id=str(concept_id),
            preferred_label=str(name),
            vocabulary=self.vocabulary,
            vocabulary_version=release,
            concept_class=tty,
            synonyms=synonyms,
            status=status,
            remapped_to=remapped,
            source_vocabularies=[source] if source else [],
            extensions={
                "is_multiple_ingredient": attributes.get("isMultipleIngredient") == "YES",
                "is_branded": attributes.get("isBranded") == "YES",
                "suppress": properties.get("suppress"),
                "active_start_date": meta.get("activeStartDate") or None,
                "active_end_date": meta.get("activeEndDate") or None,
            },
        )

    def validate_concept(self, concept_id: str, *, version: str | None = None) -> ConceptValidation:
        """Confirm an identifier exists in RxNorm and report its status (§8.1).

        Absence is not signalled by an empty response. RxNav answers for an
        entirely fabricated RxCUI with a fully-formed envelope carrying
        ``status: "UNKNOWN"``, ``source: "NONE"`` and an empty name — so
        checking merely that a response came back would report every invented
        identifier as valid, defeating the check this method exists to perform.
        """
        self._validated_release(version)
        history = self._history_status(concept_id)
        meta = history.get("metaData", {}) or {}
        attributes = history.get("attributes", {}) or {}

        known = bool(attributes.get("name")) and meta.get("source") not in (None, "", "NONE")
        if not known:
            return ConceptValidation(
                concept_id=str(concept_id),
                exists=False,
                vocabulary=self.vocabulary,
                message=f"RxCUI {concept_id} is not known to RxNorm",
            )

        remapped = [
            str(c.get("remappedRxCui"))
            for c in (history.get("derivedConcepts", {}) or {}).get("remappedConcept", []) or []
            if c.get("remappedRxCui")
        ]
        return ConceptValidation(
            concept_id=str(concept_id),
            exists=True,
            status=_parse_status(meta.get("status")),
            vocabulary=self.vocabulary,
            concept_class=attributes.get("tty"),
            remapped_to=remapped,
        )

    # -- relationships -----------------------------------------------------

    def get_relationships(
        self,
        concept_id: str,
        *,
        predicates: list[str],
        max_hops: int = 1,
    ) -> list[KnowledgeStatement]:
        """Retrieve relationships, restricted to the requested predicates.

        Two paths, because RxNav serves them through different endpoints. Most
        predicates translate to term types and come back from one ``tty=``
        request as the transitive closure (``direct=False``). A few — notably
        ``has_boss`` — are rejected by ``tty=`` and must be fetched one at a
        time through ``rela=``, which returns directly asserted edges
        (``direct=True``).

        Unsupported predicates are filtered before this method. A failure of a
        supported path propagates so incomplete evidence cannot look like a
        successful enrichment.
        """
        wanted = set(predicates)
        statements: list[KnowledgeStatement] = []

        ttys = sorted(PREDICATE_TO_TTY[p] for p in wanted if p in PREDICATE_TO_TTY)
        if ttys:
            statements.extend(self._related_by_tty(concept_id, ttys, wanted, max_hops))

        for predicate in sorted(wanted & RELA_ONLY_PREDICATES):
            statements.extend(self._related_by_rela(concept_id, predicate, max_hops))

        return statements

    def _related_by_tty(
        self,
        concept_id: str,
        ttys: list[str],
        predicates: set[str],
        max_hops: int,
    ) -> list[KnowledgeStatement]:
        payload = self.get_json(
            f"rxcui/{concept_id}/related.json",
            {"tty": " ".join(ttys)},
            operation="related_by_tty",
            query=concept_id,
            allowed_predicates=sorted(predicates),
            max_hops=max_hops,
            metadata=self._metadata(),
        )

        statements: list[KnowledgeStatement] = []
        for group in (payload or {}).get("relatedGroup", {}).get("conceptGroup") or []:
            tty = group.get("tty")
            predicate = TTY_TO_PREDICATE.get(tty or "")
            if predicate is None:
                continue
            for concept in group.get("conceptProperties") or []:
                statements.append(
                    KnowledgeStatement(
                        subject_id=str(concept_id),
                        predicate=predicate,
                        object_id=str(concept["rxcui"]),
                        object_label=concept.get("name"),
                        source_resource=self.vocabulary,
                        authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
                        direct=False,
                        evidence_type="rxnav_tty_traversal",
                        provenance_note=(
                            f"Reached via RxNav related?tty={tty}; transitive closure, "
                            "not a directly asserted RxNorm edge."
                        ),
                        extensions={"target_tty": tty},
                    )
                )
        return statements

    def _related_by_rela(
        self, concept_id: str, predicate: str, max_hops: int
    ) -> list[KnowledgeStatement]:
        payload = self.get_json(
            f"rxcui/{concept_id}/related.json",
            {"rela": predicate},
            operation="related_by_rela",
            query=concept_id,
            allowed_predicates=[predicate],
            max_hops=max_hops,
            metadata=self._metadata(),
        )

        statements: list[KnowledgeStatement] = []
        for group in (payload or {}).get("relatedGroup", {}).get("conceptGroup") or []:
            for concept in group.get("conceptProperties") or []:
                statements.append(
                    KnowledgeStatement(
                        subject_id=str(concept_id),
                        predicate=predicate,
                        object_id=str(concept["rxcui"]),
                        object_label=concept.get("name"),
                        source_resource=self.vocabulary,
                        authority=Authority.OFFICIAL_CODING_RELATIONSHIP,
                        direct=True,
                        evidence_type="rxnav_rela_assertion",
                        provenance_note=f"Directly asserted RxNorm edge via related?rela={predicate}.",
                        extensions={"target_tty": group.get("tty")},
                    )
                )
        return statements

    # -- protocol conformance ---------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[CandidateConcept]:
        """Composite search across the exact and current-approximate channels.

        The retrieval orchestrator normally drives the individual channels so it
        can honour profile settings; this method exists so ``RxNavProvider``
        satisfies :class:`~biosemalign.knowledge.providers.base.KnowledgeProvider`
        and can be exercised by the shared contract suite.
        """
        found: dict[str, CandidateConcept] = {}

        for rxcui in self.find_by_name(query):
            concept = self.lookup(rxcui)
            if concept is not None:
                concept.retrieval_evidence.append(
                    RetrievalEvidence(
                        channel=RetrievalChannel.NORMALIZED_LOOKUP,
                        query=query,
                        provider=self.provider_name,
                        score=100.0,
                        rank=1,
                        source_vocabulary="RXNORM",
                    )
                )
                found[rxcui] = concept

        for hit in self.approximate(query, max_entries=top_k, current_only=True):
            rxcui = str(hit["rxcui"])
            if rxcui in found:
                continue
            concept = self.lookup(rxcui)
            if concept is None:
                continue
            concept.retrieval_evidence.append(
                RetrievalEvidence(
                    channel=RetrievalChannel.TERMINOLOGY_APPROXIMATE_CURRENT,
                    query=query,
                    provider=self.provider_name,
                    score=float(hit.get("score", 0.0)),
                    rank=int(hit.get("rank", 0)) or None,
                    source_atom_id=str(hit.get("rxaui")) if hit.get("rxaui") else None,
                    source_vocabulary=hit.get("source"),
                )
            )
            found[rxcui] = concept
            if len(found) >= top_k:
                break

        return list(found.values())[:top_k]
