"""The public pipeline facade (§15).

The surface is deliberately small: construct from a profile, align a value, get
a decision. Advanced callers can run the stages individually, which is what
makes component-level evaluation possible (§22.1) — retrieval failure and
adjudication failure are separable only if the stages can be observed
separately.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from biosemalign._version import __version__
from biosemalign.adjudication.adjudicator import build_adjudicator
from biosemalign.adjudication.backend import Adjudicator
from biosemalign.enums import ResourceMode, SourceType
from biosemalign.exceptions import ConfigurationError, UnsupportedInput
from biosemalign.input.sau_builder import SAUBuilder
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.logging import get_logger
from biosemalign.ops import OpContext
from biosemalign.pipelines.kernel import AlignmentKernel, PackageCache
from biosemalign.pipelines.workflow import Cascade, workflow_from_profile
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.resources.registry import DeploymentEnvironment
from biosemalign.routing.conformal import Calibration
from biosemalign.routing.router import DecisionRouter
from biosemalign.schemas.artifact import AlignmentArtifact, BatchAlignmentRecord
from biosemalign.schemas.decision import AlignmentDecision, AlignmentProposal, ValidationResult
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.provenance import Provenance, RunContext
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.settings import Settings, get_settings
from biosemalign.storage.cache import ResponseCache
from biosemalign.storage.crosswalk_memo import CrosswalkMemo
from biosemalign.validation.validator import Validator

__all__ = ["BatchInputError", "BioSemAlign"]

log = get_logger(__name__)

_BATCH_MAPPING_CACHE_MAX = 1024


@dataclass(frozen=True, slots=True)
class BatchInputError:
    """A source-row parse failure that can travel through the streaming pipeline."""

    error_code: str
    error_message: str


WorkflowBuilder = Callable[[OpContext], Cascade]


class BioSemAlign:
    """A configured semantic-alignment pipeline.

    The workflow — which operators run, in which tiers — comes from the
    profile, or from ``workflow``: a function that builds a :class:`Cascade`
    from this pipeline's :class:`OpContext`. A function rather than a cascade
    so that its operators use this pipeline's providers, whose usage the
    decision provenance reports. Whatever it builds, the kernel runs it, so
    validation, routing, and provenance cannot be skipped.
    """

    def __init__(
        self,
        profile: TaskProfile,
        *,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
        adjudicator: Adjudicator | None = None,
        environment: DeploymentEnvironment | str | None = None,
        workflow: WorkflowBuilder | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings or get_settings()
        self.knowledge = KnowledgeRouter(
            profile, settings=self.settings, cache=cache, environment=environment
        )
        identifier_key = (
            self.settings.identifier_hmac_key.get_secret_value()
            if self.settings.identifier_hmac_key is not None
            else None
        )
        self.sau_builder = SAUBuilder(profile, identifier_key=identifier_key)
        self.adjudicator = adjudicator or build_adjudicator(
            profile, settings=self.settings, cache=self.knowledge.cache
        )
        self.validator = Validator(profile, provider=self.knowledge.primary_provider())
        if calibration is None and profile.routing.calibration is not None:
            calibration = Calibration.load(profile.routing.calibration)
        self.decision_router = DecisionRouter(profile, calibration=calibration)

        context = OpContext(
            profile=profile,
            settings=self.settings,
            knowledge=self.knowledge,
            adjudicator=self.adjudicator,
        )
        if isinstance(workflow, Cascade):
            raise ConfigurationError(
                "pass a function that builds the cascade from an OpContext, not a built "
                "Cascade: its operators must use this pipeline's knowledge router"
            )
        self.workflow = (workflow or workflow_from_profile)(context)
        self.workflow.verify(profile, self.settings)
        self._owns_memo = False
        self.kernel = AlignmentKernel(
            profile=profile,
            workflow=self.workflow,
            knowledge=self.knowledge,
            validator=self.validator,
            router=self.decision_router,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        name_or_path: str,
        *,
        mode: ResourceMode | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> BioSemAlign:
        """Build a pipeline from a named profile (§15)."""
        settings = kwargs.pop("settings", None) or get_settings()
        if mode is not None or cache_dir is not None:
            payload = settings.model_dump()
            payload.update(
                {
                    key: value
                    for key, value in (("mode", mode), ("cache_dir", cache_dir))
                    if value is not None
                }
            )
            # model_copy(update=...) deliberately skips validation; settings
            # controlling network access must never retain an arbitrary string.
            settings = Settings.model_validate(payload)
        return cls(load_profile(name_or_path), settings=settings, **kwargs)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> BioSemAlign:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self.kernel.memo is not None and self._owns_memo:
            self.kernel.memo.close()
            self.kernel.memo = None
        self.knowledge.close()

    def attach_memo(self, memo: CrosswalkMemo | str | Path) -> CrosswalkMemo:
        """Reuse resolutions of identical inputs across runs through a persistent memo.

        Reuse is keyed on the pinned identity of every resource, so a resource
        whose release is not pinned would let one release's mapping be served
        for another; that is refused.
        """
        uses_rxnav = any(op.op_id == "rxnav" for t in self.workflow.tiers for op in t.retrievers)
        if uses_rxnav and self.settings.rxnav_terminology_release is None:
            raise ConfigurationError(
                "the crosswalk memo needs a pinned RxNorm release "
                "(BIOSEMALIGN_RXNAV_TERMINOLOGY_RELEASE); otherwise a mapping made under one "
                "release could be reused under another"
            )
        self._owns_memo = not isinstance(memo, CrosswalkMemo)
        store = (
            memo
            if isinstance(memo, CrosswalkMemo)
            else CrosswalkMemo(
                memo,
                dir_mode=self.settings.output_dir_mode,
                file_mode=self.settings.output_file_mode,
            )
        )
        lexicon = self.knowledge.lexicon
        self.kernel.memo = store
        self.kernel.memo_resources = {
            "lexicon": lexicon.bundle_sha256 if lexicon is not None else None,
            "rxnorm_release": self.settings.rxnav_terminology_release if uses_rxnav else None,
            "llm_model": self.settings.llm_model,
            "llm_model_revision": self.settings.llm_model_revision,
        }
        return store

    # -- stages (§15, "advanced users should be able to run individual stages") --

    def new_run(self, *, run_id: str | None = None) -> RunContext:
        """Start a provenance-collecting run."""
        reset_counters = getattr(self.knowledge, "reset_counters", None)
        if reset_counters is not None:
            reset_counters()
        return RunContext(
            run_id=run_id or uuid.uuid4().hex,
            package_version=__version__,
            profile_name=self.profile.profile_name,
            profile_version=self.profile.profile_version,
            identifier_hmac_key_fingerprint=(self.settings.identifier_hmac_key_fingerprint),
        )

    def build_sau(
        self, request: AlignmentRequest, run: RunContext | None = None
    ) -> list[SemanticAlignmentUnit]:
        """Module 1: interpret the input."""
        self._validate_request_profile(request)
        return self.sau_builder.build(request, run or self.new_run())

    def retrieve_knowledge(self, sau: SemanticAlignmentUnit, *, tier: int = 0) -> KnowledgePackage:
        """Module 2: retrieve candidates and build one tier's Knowledge Package."""
        return self.kernel.tier_knowledge(self.workflow.tiers[tier], sau)[1]

    def adjudicate(
        self, sau: SemanticAlignmentUnit, package: KnowledgePackage
    ) -> AlignmentProposal:
        """Module 3: select concepts and classify the relationship."""
        return self.adjudicator.adjudicate(sau, package)

    def validate(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None = None,
    ) -> ValidationResult:
        """Module 4: deterministic validation."""
        return self.validator.validate(proposal, package, sau)

    def route(
        self,
        *,
        sau: SemanticAlignmentUnit,
        package: KnowledgePackage,
        proposal: AlignmentProposal,
        validation: ValidationResult,
        run: RunContext,
    ) -> AlignmentDecision:
        """Module 5 and 6: route, then stamp provenance."""
        return self.decision_router.route(
            sau=sau,
            package=package,
            proposal=proposal,
            validation=validation,
            provenance=self._provenance(run),
        )

    # -- whole-pipeline entry points --------------------------------------

    def align(self, request: AlignmentRequest) -> AlignmentDecision:
        """Align one request, returning the single decision it produces."""
        return self.align_artifact(request).decision

    def align_artifact(self, request: AlignmentRequest) -> AlignmentArtifact:
        """Align a request that produces exactly one SAU and retain all evidence."""
        run = self.new_run()
        units = self.build_sau(request, run)
        if not units:
            raise ValueError("request produced no semantic alignment units")
        if len(units) != 1:
            raise UnsupportedInput(
                f"align() requires exactly one semantic alignment unit, but the request "
                f"produced {len(units)}; use align_all() or align_all_artifacts()"
            )
        return self._align_one_artifact(units[0], run)

    def align_all(self, request: AlignmentRequest) -> list[AlignmentDecision]:
        """Align one request, returning every decision it produces."""
        return [artifact.decision for artifact in self.align_all_artifacts(request)]

    def align_all_artifacts(self, request: AlignmentRequest) -> list[AlignmentArtifact]:
        """Align every SAU in a request and retain each complete evidence chain."""
        run = self.new_run()
        return [self._align_one_artifact(sau, run) for sau in self.build_sau(request, run)]

    def align_value(self, value: str, **context: Any) -> AlignmentDecision:
        """Align a bare value using the profile's task settings.

        The shortest path from a string to a decision, and what the CLI's
        ``align-text`` command uses.
        """
        return self.align(self.make_request(value, **context))

    def align_batch(
        self,
        rows: Iterable[dict[str, Any] | BatchInputError],
        *,
        value_column: str,
        id_columns: list[str] | None = None,
        context_columns: list[str] | None = None,
    ) -> Iterator[AlignmentDecision]:
        """Align many rows, yielding decisions as they complete.

        A generator rather than a list: a FAERS batch is large, and the caller
        should be able to stream results to disk instead of holding every
        decision in memory.
        """
        for record in self.align_batch_records(
            rows,
            value_column=value_column,
            id_columns=id_columns,
            context_columns=context_columns,
        ):
            if record.status == "success" and record.artifact is not None:
                yield record.artifact.decision

    def align_batch_records(
        self,
        rows: Iterable[dict[str, Any] | BatchInputError],
        *,
        value_column: str,
        id_columns: list[str] | None = None,
        context_columns: list[str] | None = None,
        start_row: int = 1,
        run: RunContext | None = None,
    ) -> Iterator[BatchAlignmentRecord]:
        """Align rows independently, emitting success, error, and skipped outcomes."""
        run = run or self.new_run()
        packages = PackageCache(_BATCH_MAPPING_CACHE_MAX)
        for row_number, row in enumerate(rows, start=start_row):
            source_value: str | None = None
            record_id: str | None = None
            try:
                if isinstance(row, BatchInputError):
                    yield BatchAlignmentRecord(
                        row_number=row_number,
                        status="error",
                        error_code=row.error_code,
                        error_message=row.error_message,
                        resource_provenance=self.knowledge.resource_provenance(),
                    )
                    continue
                if not isinstance(row, dict):
                    raise UnsupportedInput(f"batch row must be an object, got {type(row).__name__}")
                if value_column not in row:
                    raise UnsupportedInput(f"configured value column {value_column!r} is absent")
                value = row[value_column]
                source_value = None if value is None else str(value)
                record_id = self._record_id(row, id_columns)
                if source_value is None or not source_value.strip():
                    yield BatchAlignmentRecord(
                        row_number=row_number,
                        status="skipped",
                        source_value=source_value,
                        record_id=record_id,
                        error_code="BLANK_VALUE",
                        error_message=f"column {value_column!r} is blank",
                        resource_provenance=self.knowledge.resource_provenance(),
                    )
                    continue

                missing_context = [c for c in (context_columns or []) if c not in row]
                if missing_context:
                    raise UnsupportedInput(
                        "configured context column(s) are absent: "
                        + ", ".join(repr(column) for column in missing_context)
                    )
                context = {c: row[c] for c in (context_columns or [])}
                for column in id_columns or []:
                    if column in row:
                        context[column] = row[column]
                if record_id is not None:
                    context["record_id"] = record_id
                # Distinguish repeated source occurrences while leaving mapping_key
                # reusable across equivalent rows.
                context["source_row_number"] = row_number

                request = self.make_request(source_value, **context)
                units = self.build_sau(request, run)
                if not units:
                    raise UnsupportedInput("row produced no semantic alignment units")
                for sau in units:
                    artifact = self.kernel.align(sau, run, cache=packages)
                    yield BatchAlignmentRecord(
                        row_number=row_number,
                        status="success",
                        source_value=source_value,
                        record_id=record_id,
                        artifact=artifact,
                        resource_provenance=self.knowledge.resource_provenance(),
                    )
            except Exception as exc:  # isolate failures to one source row
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                    raise
                message = str(exc)
                if source_value:
                    message = message.replace(source_value, "<redacted>")
                yield BatchAlignmentRecord(
                    row_number=row_number,
                    status="error",
                    source_value=source_value,
                    record_id=record_id,
                    error_code=type(exc).__name__,
                    error_message=message[:500] or "alignment failed",
                    resource_provenance=self.knowledge.resource_provenance(),
                )

    def make_request(self, value: str, **context: Any) -> AlignmentRequest:
        """Build a request for a bare value using the profile's task settings."""
        return AlignmentRequest(
            source_type=self.profile.input.source_type or SourceType.STRUCTURED_FIELD,
            value=value,
            source_system=context.pop("source_system", None),
            mention_mode=self.profile.input.mention_mode,
            task=self.profile.task.operation,
            mapping_intent=self.profile.task.mapping_intent,
            target_terminology=self.profile.task.target_terminology,
            desired_granularity=self.profile.task.desired_granularity,
            context=context,
        )

    # -- internals ---------------------------------------------------------

    def _align_one(self, sau: SemanticAlignmentUnit, run: RunContext) -> AlignmentDecision:
        return self._align_one_artifact(sau, run).decision

    def _align_one_artifact(self, sau: SemanticAlignmentUnit, run: RunContext) -> AlignmentArtifact:
        return self.kernel.align(sau, run)

    def _align_with_package(
        self,
        sau: SemanticAlignmentUnit,
        package: KnowledgePackage,
        run: RunContext,
    ) -> AlignmentArtifact:
        """Adjudicate a caller-supplied package, then finalize it through the kernel."""
        if package.sau_id != sau.sau_id:
            raise ValueError("knowledge package identity does not match the SAU")
        # Adjudicators are not input or evidence authorities. Give them
        # isolated copies and retain separate canonical snapshots for
        # validation, routing, and the persisted artifact.
        canonical_sau = sau.model_copy(deep=True)
        canonical_package = package.model_copy(deep=True)
        proposal = self.adjudicate(
            canonical_sau.model_copy(deep=True), canonical_package.model_copy(deep=True)
        )
        return self.kernel.finalize(
            sau=canonical_sau,
            package=canonical_package,
            proposal=proposal,
            run=run,
            model=self.adjudicator.provenance(),
            resolution=None,
        )

    def _validate_request_profile(self, request: AlignmentRequest) -> None:
        """Fail closed when a v0.1 request contradicts its loaded profile."""
        expected: dict[str, object] = {
            "source_type": self.profile.input.source_type,
            "mention_mode": self.profile.input.mention_mode,
            "task": self.profile.task.operation,
            "mapping_intent": self.profile.task.mapping_intent,
            "target_terminology": self.profile.task.target_terminology.casefold(),
            "desired_granularity": self.profile.task.desired_granularity,
        }
        observed: dict[str, object] = {
            "source_type": request.source_type,
            "mention_mode": request.mention_mode,
            "task": request.task,
            "mapping_intent": request.mapping_intent,
            "target_terminology": request.target_terminology.casefold(),
            "desired_granularity": request.desired_granularity,
        }
        mismatches = {
            name: {"request": observed[name], "profile": expected_value}
            for name, expected_value in expected.items()
            if observed[name] != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{name}={values['request']!r} (profile requires {values['profile']!r})"
                for name, values in mismatches.items()
            )
            raise UnsupportedInput(
                f"request contradicts profile {self.profile.profile_name!r}: {details}"
            )

    @staticmethod
    def _record_id(row: dict[str, Any], id_columns: list[str] | None) -> str | None:
        if not id_columns:
            return None
        missing = [column for column in id_columns if column not in row]
        if missing:
            raise UnsupportedInput(
                "configured identifier column(s) are absent: "
                + ", ".join(repr(column) for column in missing)
            )
        values = [row[column] for column in id_columns]
        blank = [
            column
            for column, value in zip(id_columns, values, strict=True)
            if value is None or not str(value).strip()
        ]
        if blank:
            raise UnsupportedInput(
                "configured identifier column(s) are blank: "
                + ", ".join(repr(column) for column in blank)
            )
        if len(values) == 1:
            return str(values[0])
        # Escape both the escape character and separator so component
        # boundaries remain reversible without changing ordinary FAERS IDs.
        escaped = [str(value).replace("\\", "\\\\").replace("|", "\\|") for value in values]
        return "|".join(escaped)

    def _provenance(self, run: RunContext) -> Provenance:
        """Provenance for a decision routed through the stage-by-stage API."""
        return self.kernel.provenance(run, model=self.adjudicator.provenance(), resolution=None)
