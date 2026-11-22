"""Shared state of a link derivation: the base tables, emitted rows, and the report."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BRANDED_PRODUCT_CLASSES",
    "BRAND_CLASSES",
    "INGREDIENT_CLASSES",
    "MULTI_INGREDIENT_CLASSES",
    "PRECISE_INGREDIENT_CLASSES",
    "LinkContext",
    "RuleResult",
]


def _rxnorm(*classes: str) -> tuple[str, ...]:
    """Every spelling of RxNorm classes — term types and OMOP names — from the spec."""
    from importlib import resources

    import yaml

    from biosemalign.targets.spec import TargetSpec

    text = (
        resources.files("biosemalign.domains.drug.specs")
        .joinpath("rxnorm.yaml")
        .read_text(encoding="utf-8")
    )
    spec = TargetSpec.model_validate(yaml.safe_load(text))
    return tuple(spec.spellings(classes))


# A dictionary may carry RxNorm term types or the class names Athena uses for
# the same thing; the RxNorm target spec's aliases list both.
INGREDIENT_CLASSES = _rxnorm("IN")
PRECISE_INGREDIENT_CLASSES = _rxnorm("PIN")
MULTI_INGREDIENT_CLASSES = _rxnorm("MIN")
BRAND_CLASSES = _rxnorm("BN")
#: Branded products whose ingredients speak for their brand. Packs are left
#: out: a pack may bundle several products, so its ingredients are a union.
BRANDED_PRODUCT_CLASSES = _rxnorm("SBD", "SBDC", "SBDF", "SBDG")

#: Examples kept per drop reason in the report.
EXAMPLES = 5


@dataclass(slots=True)
class RuleResult:
    """What one rule emitted and what it declined to emit, with reasons."""

    rule: str
    description: str
    overlay: str
    considered: int = 0
    linked: int = 0
    emitted: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    skipped: str | None = None


@dataclass(slots=True)
class LinkContext:
    """A DuckDB connection holding the base tables, plus what has been emitted.

    Base tables: ``base_concepts``, ``base_synonyms``, ``base_relationships``,
    ``base_xrefs``, and ``family_ingredients`` (active ingredients of the
    target family). Rules insert into ``emit_relationships`` and
    ``emit_xrefs`` and record a :class:`RuleResult`.
    """

    con: Any
    target: str
    family: list[str]
    served: dict[str, str]
    results: list[RuleResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def vocabulary(self, name: str) -> str | None:
        """The base's spelling of a vocabulary, or None when it is not served."""
        return self.served.get(name.casefold())

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        return list(self.con.execute(sql, list(params)).fetchall())

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> int:
        row = self.con.execute(sql, list(params)).fetchone()
        return int(row[0] or 0) if row else 0

    def record(
        self,
        result: RuleResult,
        *,
        outcomes: str,
        emitted_sql: str,
        emitted_params: Sequence[Any] = (),
    ) -> RuleResult:
        """Fill ``result`` from an outcome table (subject, outcome, example) and emitted rows.

        ``outcomes`` has one row per considered subject with columns
        ``subject`` (a display string) and ``outcome`` ('linked' or a drop
        reason).
        """
        result.considered = self.scalar(f"SELECT count(*) FROM {outcomes}")
        result.linked = self.scalar(f"SELECT count(*) FROM {outcomes} WHERE outcome = 'linked'")
        for reason, count in self.rows(
            f"SELECT outcome, count(*) FROM {outcomes} WHERE outcome <> 'linked' "
            "GROUP BY 1 ORDER BY 2 DESC"
        ):
            result.dropped[str(reason)] = int(count)
            result.examples[str(reason)] = [
                str(row[0])
                for row in self.rows(
                    f"SELECT subject FROM {outcomes} WHERE outcome = ? "
                    f"ORDER BY hash(subject) LIMIT {EXAMPLES}",
                    [reason],
                )
            ]
        result.emitted = self.scalar(emitted_sql, emitted_params)
        self.results.append(result)
        return result

    def skip(self, result: RuleResult, reason: str) -> RuleResult:
        result.skipped = reason
        self.results.append(result)
        return result
