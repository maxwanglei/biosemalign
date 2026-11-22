"""The public pipeline facade (§15).

The surface is deliberately small: construct from a profile, align a value, get
a decision. Advanced callers can run the stages individually, which is what
makes component-level evaluation possible (§22.1) — matching failure and
resolution failure are separable only if the stages can be observed separately.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from biosemalign._version import __version__
from biosemalign.enums import ResourceMode, SourceType
from biosemalign.exceptions import ConfigurationError, UnsupportedInput, WorkflowViolation
from biosemalign.input.sau_builder import SAUBuilder
from biosemalign.kernel import AlignmentKernel, Cascade, PackageCache
from biosemalign.kernel.guards import resolver_view
from biosemalign.knowledge.router import KnowledgeRouter
from biosemalign.logging import get_logger
from biosemalign.modules import OpContext, PassToNext, Resolver
from biosemalign.modules.expand.hierarchy import HierarchyExpander
from biosemalign.modules.model import ModelService, capture_model_calls
from biosemalign.profiles.loader import load_profile
from biosemalign.profiles.schema import TaskProfile
from biosemalign.resources.registry import DeploymentEnvironment
from biosemalign.routing.conformal import Calibration
from biosemalign.routing.router import DecisionRouter
from biosemalign.schemas.artifact import AlignmentArtifact, BatchAlignmentRecord
from biosemalign.schemas.coding import CodingArtifact
from biosemalign.schemas.decision import AlignmentDecision, AlignmentProposal, ValidationResult
from biosemalign.schemas.expansion import ExpansionSet
from biosemalign.schemas.knowledge import KnowledgePackage
from biosemalign.schemas.phenotyping import ConceptSetArtifact
from biosemalign.schemas.provenance import RunContext
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.sau import SemanticAlignmentUnit
from biosemalign.settings import Settings, get_settings
from biosemalign.storage.cache import ResponseCache
from biosemalign.storage.crosswalk_memo import CrosswalkMemo
from biosemalign.validation.validator import Validator
from biosemalign.workflows.coding import CodingCascade, CodingWorkflow, coding_workflow_from_profile
from biosemalign.workflows.mapping import workflow_from_profile
from biosemalign.workflows.phenotyping import (
    PhenotypingWorkflow,
    phenotyping_workflow_from_profile,
)

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
        resolver: Resolver | None = None,
        environment: DeploymentEnvironment | str | None = None,
        workflow: WorkflowBuilder | None = None,
        calibration: Calibration | None = None,
        model_service: ModelService | None = None,
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
        self._model_service = model_service
        self.validator = Validator(
            profile, provider=self.knowledge.primary_provider(), model_service=model_service
        )
        if calibration is None and profile.routing.calibration is not None:
            calibration = Calibration.load(profile.routing.calibration)
        self.decision_router = DecisionRouter(profile, calibration=calibration)

        context = OpContext(
            profile=profile,
            settings=self.settings,
            knowledge=self.knowledge,
            resolver=resolver,
            model_service=model_service,
        )
        self._op_context = context
        self._owns_memo = False
        self.workflow: Cascade | None = None
        self.resolver: Resolver | None = None
        self.kernel: AlignmentKernel | None = None
        self.coding_cascade: CodingCascade | None = None
        self.coding_runner: CodingWorkflow | None = None
        self.phenotyping_runner: PhenotypingWorkflow | None = None

        if profile.workflow == "mapping":
            if isinstance(workflow, Cascade):
                raise ConfigurationError(
                    "pass a function that builds the cascade from an OpContext, not a built "
                    "Cascade: its operators must use this pipeline's knowledge router"
                )
            self.workflow = (workflow or workflow_from_profile)(context)
            self.workflow.verify(profile, self.settings)
            self.resolver = resolver or self.workflow.tiers[-1].resolver
            self.kernel = AlignmentKernel(
                profile=profile,
                workflow=self.workflow,
                knowledge=self.knowledge,
                validator=self.validator,
                router=self.decision_router,
                model_services=self.model_services,
            )
        elif profile.workflow == "coding":
            if workflow is not None:
                raise ConfigurationError(
                    "custom mapping workflow builder cannot be used with workflow='coding'"
                )
            self.coding_cascade = coding_workflow_from_profile(context)
            self.coding_cascade.verify(profile, self.settings)
            self.coding_runner = CodingWorkflow(
                profile=profile,
                cascade=self.coding_cascade,
                knowledge=self.knowledge,
                model_services=self.model_services,
            )
        else:
            if workflow is not None:
                raise ConfigurationError(
                    "custom mapping workflow builder cannot be used with workflow='phenotyping'"
                )
            self.phenotyping_runner = phenotyping_workflow_from_profile(context)
            self.phenotyping_runner.verify(self.settings)

    @property
    def model_services(self) -> tuple[ModelService, ...]:
        """All distinct :class:`ModelService` instances wired into this pipeline."""
        services: list[ModelService] = []
        seen: set[int] = set()

        def add(candidate: object) -> None:
            if isinstance(candidate, ModelService) and id(candidate) not in seen:
                seen.add(id(candidate))
                services.append(candidate)

        add(self._model_service)
        add(getattr(self.resolver, "model_service", None))
        if self.workflow is not None:
            for tier in self.workflow.tiers:
                for op in [
                    *tier.cleaners,
                    *tier.matchers,
                    tier.expander,
                    tier.linker,
                    tier.resolver,
                ]:
                    if op is not None:
                        add(getattr(op, "model_service", None))
        if self.coding_cascade is not None:
            for coding_tier in self.coding_cascade.tiers:
                coding_ops: list[object] = [*coding_tier.cleaners, *coding_tier.matchers]
                if coding_tier.linker is not None:
                    coding_ops.append(coding_tier.linker)
                for coding_op in coding_ops:
                    add(getattr(coding_op, "model_service", None))
        if self.phenotyping_runner is not None:
            for svc in self.phenotyping_runner.all_model_services:
                add(svc)
        for check in self.validator.checks:
            add(getattr(check, "model_service", None))
        return tuple(services)

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
        for svc in self.model_services:
            svc.close()
        if self.kernel is not None and self.kernel.memo is not None and self._owns_memo:
            self.kernel.memo.close()
            self.kernel.memo = None
        self.knowledge.close()

    def attach_memo(self, memo: CrosswalkMemo | str | Path) -> CrosswalkMemo:
        """Reuse resolutions of identical inputs across runs through a persistent memo.

        Reuse is keyed on the pinned identity of every resource, so a resource
        whose release is not pinned would let one release's mapping be served
        for another; that is refused.
        """
        kernel = self._require_mapping_kernel()
        lexicon = self.knowledge.lexicon
        pins = self.knowledge.memo_pins(
            set(self.profile.knowledge.resources), settings=self.settings
        )
        store = (
            memo
            if isinstance(memo, CrosswalkMemo)
            else CrosswalkMemo(
                memo,
                dir_mode=self.settings.output_dir_mode,
                file_mode=self.settings.output_file_mode,
            )
        )
        if kernel.memo is not None and self._owns_memo and kernel.memo is not store:
            kernel.memo.close()
        self._owns_memo = not isinstance(memo, CrosswalkMemo)
        kernel.memo = store
        kernel.memo_resources = {
            "lexicon": lexicon.bundle_sha256 if lexicon is not None else None,
            **pins,
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
        for svc in self.model_services:
            svc.reset_counters()
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
        """Input stage: interpret the input."""
        self._validate_request_profile(request)
        return self.sau_builder.build(request, run or self.new_run())

    def build_package(self, sau: SemanticAlignmentUnit, *, tier: int = 0) -> KnowledgePackage:
        """Match and link stage: build one tier's Knowledge Package."""
        kernel = self._require_mapping_kernel()
        workflow = self._require_mapping_workflow()
        return kernel.tier_knowledge(workflow.tiers[tier], sau)[1]

    def resolve(self, sau: SemanticAlignmentUnit, package: KnowledgePackage) -> AlignmentProposal:
        """Resolve stage: select concepts and classify the relationship."""
        resolver = self._require_resolver()
        isolated_sau = resolver_view(sau, self.profile)
        result = resolver.resolve(isolated_sau, package.model_copy(deep=True))
        if isinstance(result, PassToNext):
            raise WorkflowViolation(
                f"standalone resolver {resolver.op_id!r} passed ({result.reason}); "
                "run the full workflow cascade to escalate across tiers"
            )
        return result

    def validate(
        self,
        proposal: AlignmentProposal,
        package: KnowledgePackage,
        sau: SemanticAlignmentUnit | None = None,
    ) -> ValidationResult:
        """Check stage: deterministic validation."""
        return self.validator.validate(proposal, package, sau)

    # -- whole-pipeline entry points --------------------------------------

    def run(
        self, request: AlignmentRequest
    ) -> AlignmentArtifact | CodingArtifact | ConceptSetArtifact:
        """Execute this profile's configured workflow on a single-SAU request."""
        if self.profile.workflow == "mapping":
            return self.align_artifact(request)
        if self.profile.workflow == "coding":
            return self.code_artifact(request)
        if self.profile.workflow == "phenotyping":
            return self.propose_concept_set(request)
        raise ConfigurationError(f"workflow {self.profile.workflow!r} is not supported")

    def propose_concept_set(self, request: AlignmentRequest) -> ConceptSetArtifact:
        """Propose a phenotype concept set awaiting human review and approval (§5.4, G8)."""
        runner = self._get_phenotyping_runner()
        run = self.new_run()
        units = self.build_sau(request, run)
        if not units:
            raise ValueError("request produced no semantic alignment units")
        if len(units) != 1:
            raise UnsupportedInput(
                f"propose_concept_set() requires exactly one semantic alignment unit, but "
                f"the request produced {len(units)}"
            )
        return runner.propose(units[0], run)

    def expand_code(
        self,
        code: str,
        *,
        vocabulary: str | None = None,
        level: str | None = None,
        max_depth: int = 8,
    ) -> ExpansionSet:
        """Interactive hierarchy expansion from a concept code at a target level (§5.4)."""
        target_vocab = vocabulary or self.profile.target.terminology
        concept = self.knowledge.provider_for(target_vocab).lookup(code)
        if concept is None:
            raise ValueError(f"concept {target_vocab}:{code} was not found in terminology")
        requested_level = level if level is not None else self.profile.target.level
        run = self.new_run()
        request = AlignmentRequest(
            source_type=self.profile.input.source_type or SourceType.STRUCTURED_FIELD,
            value=concept.preferred_label,
            mention_mode=self.profile.input.mention_mode,
            task=self.profile.task.operation,
            mapping_intent=self.profile.task.mapping_intent,
            target_terminology=target_vocab,
            desired_granularity=requested_level,
        )
        sau = self.sau_builder.build(request, run)[0]
        expander = HierarchyExpander(
            self.knowledge, self.profile, max_depth=max_depth, level=requested_level
        )
        expansion = expander.expand(sau, [concept])
        surviving = [
            member
            for member in expansion.members
            if all(
                (
                    step.predicate not in self.profile.knowledge.prohibited_relationships
                    if step.direction == "lexical"
                    else self.profile.knowledge.permits(step.predicate)
                )
                for step in member.path
            )
        ]
        return expansion.model_copy(update={"members": surviving})

    def code_artifact(self, request: AlignmentRequest) -> CodingArtifact:
        """Attach every matched code for a single-SAU request without deciding."""
        runner = self._get_coding_runner()
        run = self.new_run()
        units = self.build_sau(request, run)
        if not units:
            raise ValueError("request produced no semantic alignment units")
        if len(units) != 1:
            raise UnsupportedInput(
                f"code_artifact() requires exactly one semantic alignment unit, but the request "
                f"produced {len(units)}; use code_all_artifacts()"
            )
        return runner.code(units[0], run)

    def code_all_artifacts(self, request: AlignmentRequest) -> list[CodingArtifact]:
        """Attach every matched code for each SAU produced by ``request``."""
        runner = self._get_coding_runner()
        run = self.new_run()
        return [runner.code(sau, run) for sau in self.build_sau(request, run)]

    def code_text(self, text: str, **context: Any) -> CodingArtifact:
        """Interactive single-string coding using local non-model operators only (§5.4)."""
        runner = self._get_coding_runner()
        runner.cascade.verify_local_only(knowledge=self.knowledge)
        return self.code_artifact(self.make_request(text, **context))

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
        """Align a bare value using the profile's task settings."""
        return self.align(self.make_request(value, **context))

    def align_batch(
        self,
        rows: Iterable[dict[str, Any] | BatchInputError],
        *,
        value_column: str,
        id_columns: list[str] | None = None,
        context_columns: list[str] | None = None,
    ) -> Iterator[AlignmentDecision]:
        """Align many rows, yielding decisions as they complete."""
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
        kernel = self._require_mapping_kernel()
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
                context["source_row_number"] = row_number

                request = self.make_request(source_value, **context)
                units = self.build_sau(request, run)
                if not units:
                    raise UnsupportedInput("row produced no semantic alignment units")
                for sau in units:
                    artifact = kernel.align(sau, run, cache=packages)
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
            target_terminology=self.profile.target.terminology,
            desired_granularity=self.profile.target.level,
            context=context,
        )

    # -- internals ---------------------------------------------------------

    def _require_mapping_kernel(self) -> AlignmentKernel:
        if self.kernel is None:
            raise ConfigurationError(
                f"profile {self.profile.profile_name!r} uses workflow={self.profile.workflow!r}, "
                "not 'mapping'"
            )
        return self.kernel

    def _require_mapping_workflow(self) -> Cascade:
        if self.workflow is None:
            raise ConfigurationError(
                f"profile {self.profile.profile_name!r} uses workflow={self.profile.workflow!r}, "
                "not 'mapping'"
            )
        return self.workflow

    def _require_resolver(self) -> Resolver:
        if self.resolver is None:
            raise ConfigurationError(
                f"profile {self.profile.profile_name!r} uses workflow={self.profile.workflow!r}, "
                "which has no resolver"
            )
        return self.resolver

    def _get_coding_runner(self) -> CodingWorkflow:
        if self.coding_runner is None:
            cascade = coding_workflow_from_profile(self._op_context)
            cascade.verify(self.profile, self.settings)
            self.coding_cascade = cascade
            self.coding_runner = CodingWorkflow(
                profile=self.profile,
                cascade=cascade,
                knowledge=self.knowledge,
                model_services=self.model_services,
            )
        return self.coding_runner

    def _get_phenotyping_runner(self) -> PhenotypingWorkflow:
        if self.phenotyping_runner is None:
            runner = phenotyping_workflow_from_profile(self._op_context)
            runner.verify(self.settings)
            self.phenotyping_runner = runner
        return self.phenotyping_runner

    def _align_one(self, sau: SemanticAlignmentUnit, run: RunContext) -> AlignmentDecision:
        return self._align_one_artifact(sau, run).decision

    def _align_one_artifact(self, sau: SemanticAlignmentUnit, run: RunContext) -> AlignmentArtifact:
        return self._require_mapping_kernel().align(sau, run)

    def _align_with_package(
        self,
        sau: SemanticAlignmentUnit,
        package: KnowledgePackage,
        run: RunContext,
    ) -> AlignmentArtifact:
        """Resolve a caller-supplied package, then finalize it through the kernel."""
        if package.sau_id != sau.sau_id:
            raise ValueError("knowledge package identity does not match the SAU")
        kernel = self._require_mapping_kernel()
        resolver = self._require_resolver()
        canonical_sau = sau.model_copy(deep=True)
        canonical_package = package.model_copy(deep=True)
        with capture_model_calls(self.model_services) as stage_calls:
            proposal = self.resolve(canonical_sau, canonical_package)
        return kernel.finalize(
            sau=canonical_sau,
            package=canonical_package,
            proposal=proposal,
            run=run,
            model=resolver.provenance(),
            resolution=None,
            model_calls=stage_calls,
        )

    def _validate_request_profile(self, request: AlignmentRequest) -> None:
        """Fail closed when a request contradicts its loaded profile."""
        expected: dict[str, object] = {
            "source_type": self.profile.input.source_type,
            "mention_mode": self.profile.input.mention_mode,
            "task": self.profile.task.operation,
            "mapping_intent": self.profile.task.mapping_intent,
            "target_terminology": self.profile.target.terminology.casefold(),
            "desired_granularity": self.profile.target.level,
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
        escaped = [str(value).replace("\\", "\\\\").replace("|", "\\|") for value in values]
        return "|".join(escaped)
