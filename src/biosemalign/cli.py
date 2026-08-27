"""Command-line interface (§16)."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from itertools import chain, islice
from pathlib import Path
from typing import Annotated, Any

import typer

from biosemalign._version import __version__
from biosemalign.enums import ResourceMode
from biosemalign.evaluation.metrics import EvaluationPrediction
from biosemalign.exceptions import BioSemAlignError
from biosemalign.logging import configure_logging
from biosemalign.pipelines.base import BatchInputError, BioSemAlign
from biosemalign.profiles.loader import available_profiles, load_profile
from biosemalign.run_manifest import (
    build_run_manifest,
    code_tree_sha256,
    effective_settings_sha256,
    file_sha256,
    merge_resource_provenance,
)
from biosemalign.schemas.artifact import AlignmentArtifact, BatchAlignmentRecord
from biosemalign.schemas.decision import AlignmentDecision
from biosemalign.schemas.provenance import ResourceProvenance
from biosemalign.settings import get_settings
from biosemalign.storage.cache import ResponseCache
from biosemalign.storage.jsonl import atomic_text_writer, read_jsonl, read_models
from biosemalign.storage.parquet import decision_to_row, write_csv, write_parquet
from biosemalign.storage.paths import reject_symlink_components
from biosemalign.util import stable_digest, utc_now

app = typer.Typer(
    name="biosemalign",
    help="Context-aware, knowledge-guided biomedical semantic alignment.",
    no_args_is_help=True,
    add_completion=False,
)

ProfileOption = Annotated[
    str, typer.Option("--profile", "-p", help="Task profile name or path to a profile YAML.")
]
ModeOption = Annotated[
    ResourceMode | None,
    typer.Option(
        "--mode",
        help="Resource access mode: ONLINE, CACHE_PREFER, FROZEN, or legacy OFFLINE.",
    ),
]
CacheOption = Annotated[
    Path | None, typer.Option("--cache-dir", help="Provider response cache directory.")
]


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    configure_logging(level="DEBUG" if verbose else None)


@app.command()
def version() -> None:
    """Print the package version and schema version."""
    from biosemalign.schemas import SCHEMA_VERSION

    typer.echo(f"biosemalign {__version__} (core schema {SCHEMA_VERSION})")


@app.command("profiles")
def list_profiles() -> None:
    """List discoverable task profiles."""
    names = available_profiles()
    if not names:
        typer.echo("no profiles found")
        raise typer.Exit(code=1)
    for name in names:
        profile = load_profile(name)
        typer.echo(
            f"{name}  v{profile.profile_version}  "
            f"{profile.task.operation} -> {profile.task.target_terminology}"
            f" ({profile.task.desired_granularity or 'any granularity'})"
        )


@app.command("align-text")
def align_text(
    profile: ProfileOption,
    text: Annotated[str, typer.Option("--text", "-t", help="The value to align.")],
    mode: ModeOption = None,
    cache_dir: CacheOption = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the decision as JSON here.")
    ] = None,
    context: Annotated[
        list[str] | None,
        typer.Option("--context", "-c", help="Context as key=value; repeatable."),
    ] = None,
) -> None:
    """Align a single value and print the resulting decision."""
    with _pipeline(profile, mode, cache_dir) as pipeline:
        decision = pipeline.align_value(text, **_parse_context(context))
        output_dir_mode = pipeline.settings.output_dir_mode
        output_file_mode = pipeline.settings.output_file_mode

    payload = decision.model_dump(mode="json")
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output is not None:
        with atomic_text_writer(
            output, dir_mode=output_dir_mode, file_mode=output_file_mode
        ) as handle:
            handle.write(rendered + "\n")
        typer.echo(f"wrote {output}")
    else:
        typer.echo(rendered)


@app.command("align-batch")
def align_batch(
    profile: ProfileOption,
    input_path: Annotated[Path, typer.Option("--input", "-i", help="CSV or JSONL input.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output path.")],
    value_column: Annotated[str, typer.Option("--value-column", help="Column holding the value.")],
    id_columns: Annotated[
        list[str] | None, typer.Option("--id-columns", help="Identifier columns; repeatable.")
    ] = None,
    context_columns: Annotated[
        list[str] | None, typer.Option("--context-columns", help="Context columns; repeatable.")
    ] = None,
    mode: ModeOption = None,
    cache_dir: CacheOption = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Continue a matching .partial/.checkpoint run."),
    ] = False,
    checkpoint_every: Annotated[
        int,
        typer.Option("--checkpoint-every", min=1, help="Durably checkpoint after N rows."),
    ] = 100,
) -> None:
    """Align a batch table, writing a crosswalk plus auditable JSONL artifacts.

    The JSONL contains a success/error/skipped record for every source row;
    successful rows include the SAU, complete Knowledge Package, proposal,
    validation result, and routed decision.  A manifest written last is the
    commit marker for the finalized output pair.
    """
    crosswalk_path = output if output.suffix == ".parquet" else output.with_suffix(".csv")
    jsonl_path = crosswalk_path.with_suffix(".jsonl")
    partial_path = jsonl_path.with_name(jsonl_path.name + ".partial")
    checkpoint_path = jsonl_path.with_name(jsonl_path.name + ".checkpoint.json")
    manifest_path = crosswalk_path.with_suffix(".manifest.json")
    derived_paths = {
        "crosswalk": crosswalk_path.absolute(),
        "artifact JSONL": jsonl_path.absolute(),
        "partial artifact": partial_path.absolute(),
        "checkpoint": checkpoint_path.absolute(),
        "manifest": manifest_path.absolute(),
    }
    input_resolved = input_path.resolve()
    for name, path in derived_paths.items():
        try:
            reject_symlink_components(path)
        except BioSemAlignError as exc:
            raise typer.BadParameter(f"unsafe {name} path: {exc}") from exc
    resolved_outputs = {name: path.resolve() for name, path in derived_paths.items()}
    aliases = [name for name, path in resolved_outputs.items() if path == input_resolved]
    if aliases:
        raise typer.BadParameter(
            "input path must be distinct from every output; it aliases " + ", ".join(aliases)
        )
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise typer.BadParameter("derived batch output paths must be distinct")
    required_columns = [value_column, *(id_columns or []), *(context_columns or [])]
    _validate_input_schema(input_path, required_columns)
    input_hash = file_sha256(input_path)
    started_at = utc_now()

    if resume:
        state = _load_checkpoint(checkpoint_path)
        if not partial_path.is_file():
            raise typer.BadParameter(f"cannot resume: {partial_path} does not exist")
        if state.get("input_sha256") != input_hash:
            raise typer.BadParameter("cannot resume: input content changed since checkpoint")
        started_at = datetime.fromisoformat(str(state["started_at"]))
    else:
        if (
            partial_path.exists()
            or partial_path.is_symlink()
            or checkpoint_path.exists()
            or checkpoint_path.is_symlink()
        ):
            raise typer.BadParameter(
                f"partial output exists; use --resume or remove {partial_path} and "
                f"{checkpoint_path}"
            )
        state = {}
        # A previous manifest must not make in-progress replacement outputs
        # look complete.  The new manifest is written only after both files.
        manifest_path.unlink(missing_ok=True)

    with _pipeline(profile, mode, cache_dir) as pipeline:
        profile_hash = stable_digest(pipeline.profile.model_dump(mode="json"))
        if state and state.get("profile_sha256") != profile_hash:
            raise typer.BadParameter("cannot resume: effective profile changed since checkpoint")

        configuration_hash = stable_digest(
            {
                "code_tree_sha256": code_tree_sha256(),
                "settings_sha256": effective_settings_sha256(pipeline.settings),
                "input_path": str(input_resolved),
                "input_format": input_path.suffix.casefold(),
                "output_path": str(crosswalk_path.resolve()),
                "value_column": value_column,
                "id_columns": list(id_columns or []),
                "context_columns": list(context_columns or []),
            }
        )
        if state and state.get("configuration_sha256") != configuration_hash:
            raise typer.BadParameter("cannot resume: configuration changed")

        (
            processed_rows,
            counts,
            routing,
            recorded_run_id,
            partial_resources,
        ) = _scan_partial(partial_path)
        state_run_id = str(state["run_id"]) if state.get("run_id") else None
        if state_run_id and recorded_run_id and state_run_id != recorded_run_id:
            raise typer.BadParameter(
                "cannot resume: checkpoint and partial output have different run IDs"
            )
        try:
            checkpoint_rows = int(state.get("processed_rows", 0))
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter("cannot resume: checkpoint row count is invalid") from exc
        if checkpoint_rows > processed_rows:
            raise typer.BadParameter(
                "cannot resume: checkpoint is ahead of the durable partial output"
            )
        if processed_rows > checkpoint_rows:
            # A crash can occur after the partial stream is fsynced and before
            # its checkpoint is atomically advanced. The validated contiguous
            # partial is authoritative and carries a cumulative counter snapshot.
            prior_resources = partial_resources
        else:
            prior_resources = _checkpoint_resources(state, fallback=partial_resources)
        run_id = str(state_run_id or recorded_run_id or pipeline.new_run().run_id)
        run = pipeline.new_run(run_id=run_id)
        rows: Iterator[dict[str, Any] | BatchInputError] = iter(_read_rows(input_path))
        if processed_rows:
            rows = islice(rows, processed_rows, None)

        try:
            first = next(rows)
            pending_rows: Iterator[dict[str, Any] | BatchInputError] = chain((first,), rows)
        except StopIteration:
            if processed_rows == 0:
                typer.echo(f"no rows in {input_path}", err=True)
                raise typer.Exit(code=1) from None
            pending_rows = iter(())

        checkpoint = {
            "checkpoint_version": "2.0.0",
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "input_path": str(input_path.resolve()),
            "input_sha256": input_hash,
            "profile_name": pipeline.profile.profile_name,
            "profile_sha256": profile_hash,
            "configuration_sha256": configuration_hash,
            "processed_rows": processed_rows,
            "resource_provenance": [
                resource.model_dump(mode="json") for resource in prior_resources
            ],
        }
        _write_checkpoint(
            checkpoint_path,
            checkpoint,
            dir_mode=pipeline.settings.output_dir_mode,
            file_mode=pipeline.settings.output_file_mode,
        )

        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        if not resume:
            flags |= os.O_EXCL
        try:
            descriptor = os.open(partial_path, flags, pipeline.settings.output_file_mode)
        except OSError as exc:
            raise typer.BadParameter(
                f"cannot open private partial output {partial_path}: {exc}"
            ) from exc
        os.fchmod(descriptor, pipeline.settings.output_file_mode)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            for record in pipeline.align_batch_records(
                pending_rows,
                value_column=value_column,
                id_columns=list(id_columns or []),
                context_columns=list(context_columns or []),
                start_row=processed_rows + 1,
                run=run,
            ):
                record = record.model_copy(
                    update={
                        "resource_provenance": merge_resource_provenance(
                            prior_resources, record.resource_provenance
                        )
                    }
                )
                handle.write(record.model_dump_json(exclude_none=False) + "\n")
                processed_rows = max(processed_rows, record.row_number)
                counts[record.status] = counts.get(record.status, 0) + 1
                if record.artifact is not None:
                    route = record.artifact.decision.routing_status.value
                    routing[route] = routing.get(route, 0) + 1
                if processed_rows % checkpoint_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    checkpoint["processed_rows"] = processed_rows
                    checkpoint["resource_provenance"] = _merged_resource_payload(
                        prior_resources, pipeline.knowledge.resource_provenance()
                    )
                    _write_checkpoint(
                        checkpoint_path,
                        checkpoint,
                        dir_mode=pipeline.settings.output_dir_mode,
                        file_mode=pipeline.settings.output_file_mode,
                    )
            handle.flush()
            os.fsync(handle.fileno())

        checkpoint["processed_rows"] = processed_rows
        aggregate_resources = merge_resource_provenance(
            prior_resources, pipeline.knowledge.resource_provenance()
        )
        checkpoint["resource_provenance"] = [
            resource.model_dump(mode="json") for resource in aggregate_resources
        ]
        _write_checkpoint(
            checkpoint_path,
            checkpoint,
            dir_mode=pipeline.settings.output_dir_mode,
            file_mode=pipeline.settings.output_file_mode,
        )
        _require_unchanged_input(input_path, expected_sha256=input_hash)

        crosswalk_rows = _crosswalk_rows(partial_path)
        if crosswalk_path.suffix == ".parquet":
            write_parquet(
                crosswalk_path,
                crosswalk_rows,
                dir_mode=pipeline.settings.output_dir_mode,
                file_mode=pipeline.settings.output_file_mode,
            )
        else:
            write_csv(
                crosswalk_path,
                crosswalk_rows,
                dir_mode=pipeline.settings.output_dir_mode,
                file_mode=pipeline.settings.output_file_mode,
            )
        # Keep the durable partial until the manifest commit marker exists.
        # A crash during finalization can then be resumed deterministically.
        with (
            partial_path.open(encoding="utf-8") as source,
            atomic_text_writer(
                jsonl_path,
                dir_mode=pipeline.settings.output_dir_mode,
                file_mode=pipeline.settings.output_file_mode,
            ) as destination,
        ):
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)

        _require_unchanged_input(input_path, expected_sha256=input_hash)

        manifest = build_run_manifest(
            pipeline=pipeline,
            run_id=run_id,
            started_at=started_at,
            input_sha256=input_hash,
            output_paths=[crosswalk_path, jsonl_path],
            record_counts=counts,
            resources=aggregate_resources,
        )
        with atomic_text_writer(
            manifest_path,
            dir_mode=pipeline.settings.output_dir_mode,
            file_mode=pipeline.settings.output_file_mode,
        ) as handle:
            handle.write(manifest.model_dump_json(indent=2) + "\n")
        partial_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        _fsync_directory(jsonl_path.parent)

    typer.echo(f"processed {processed_rows} source rows")
    typer.echo(f"  crosswalk: {crosswalk_path}")
    typer.echo(f"  full artifacts: {jsonl_path}")
    typer.echo(f"  run manifest: {manifest_path}")
    _print_batch_summary(counts, routing)
    if counts.get("error", 0):
        typer.secho(
            f"batch completed with {counts['error']} row error(s); outputs were finalized",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def inspect(
    profile: ProfileOption,
    text: Annotated[str, typer.Option("--text", "-t", help="The value to inspect.")],
    show_knowledge: Annotated[bool, typer.Option("--show-knowledge")] = False,
    show_validation: Annotated[bool, typer.Option("--show-validation")] = False,
    mode: ModeOption = None,
    cache_dir: CacheOption = None,
) -> None:
    """Run one value through the pipeline and explain each stage."""
    with _pipeline(profile, mode, cache_dir) as pipeline:
        run = pipeline.new_run()
        request = pipeline.make_request(text)
        sau = pipeline.build_sau(request, run)[0]

        typer.secho(f"\nSAU {sau.sau_id}", bold=True)
        typer.echo(f"  mention:   {sau.mention.text!r}")
        typer.echo(f"  types:     {[t.value for t in sau.mention.semantic_types]}")
        typer.echo(f"  readiness: {sau.readiness.status.value}")
        if sau.readiness.warnings:
            for warning in sau.readiness.warnings:
                typer.echo(f"    ! {warning}")

        package = pipeline.retrieve_knowledge(sau)
        typer.secho(f"\nCandidates ({len(package.candidates)})", bold=True)
        for index, candidate in enumerate(package.candidates, start=1):
            channels = ", ".join(sorted(c.value for c in candidate.channels))
            typer.echo(
                f"  {index}. {candidate.candidate_id} [{candidate.concept_class or '?'}] "
                f"{candidate.preferred_label}"
            )
            typer.echo(
                f"       status={candidate.status.value} score={candidate.best_score:.1f} "
                f"sources={candidate.source_vocabularies} via {channels}"
            )
            if show_knowledge:
                for statement in candidate.relationships:
                    typer.echo(
                        f"       - {statement.predicate} -> {statement.object_id} "
                        f"({statement.object_label})"
                    )

        if show_knowledge:
            typer.secho("\nCoding policies", bold=True)
            for policy in package.coding_policies:
                typer.echo(f"  [{policy.policy_id}] {policy.rule}")

        proposal = pipeline.adjudicate(sau, package)
        validation = pipeline.validate(proposal, package, sau)
        decision = pipeline.route(
            sau=sau, package=package, proposal=proposal, validation=validation, run=run
        )

        typer.secho("\nDecision", bold=True)
        typer.echo(f"  outcome:      {decision.decision.value}")
        typer.echo(f"  relationship: {decision.relationship_to_source.value}")
        for concept in decision.selected_concepts:
            typer.echo(f"  selected:     {concept.concept_id}  {concept.label}")
        if decision.source_level_concept:
            src = decision.source_level_concept
            typer.echo(f"  source level: {src.concept_id}  {src.label}")
        for loss in decision.semantic_loss:
            typer.echo(f"  loss [{loss.kind.value}]: {loss.source_value or loss.description}")
        typer.echo(f"  validation:   {decision.validation.status.value}")
        typer.echo(f"  routing:      {decision.routing_status.value}")
        for reason in decision.routing_reasons:
            typer.echo(f"    - {reason}")

        if show_validation:
            typer.secho("\nValidator findings", bold=True)
            if not decision.validation.findings:
                typer.echo("  (none)")
            for finding in decision.validation.findings:
                typer.echo(f"  [{finding.severity.value}] {finding.code}: {finding.message}")

        typer.secho("\nProvenance", bold=True)
        for resource in decision.provenance.resources:
            typer.echo(
                f"  {resource.resource_name} release={resource.terminology_release} "
                f"mode={resource.mode.value} cache_hits={resource.cache_hits} "
                f"network={resource.network_calls}"
            )
        model = decision.provenance.model
        if model:
            typer.echo(f"  adjudicator: {model.backend} ({model.model_name})")


@app.command()
def evaluate(
    gold: Annotated[Path, typer.Option("--gold", help="Gold mappings as JSONL.")],
    predictions: Annotated[Path, typer.Option("--predictions", help="Decisions as JSONL.")],
    report: Annotated[
        Path | None, typer.Option("--report", help="Write the report as JSON here.")
    ] = None,
) -> None:
    """Compare predictions against a case-ID keyed gold standard (§22)."""
    from biosemalign.evaluation.metrics import (
        GoldMapping,
        evaluate_predictions,
    )

    try:
        gold_cases = list(read_models(gold, GoldMapping))
        prediction_items = list(_read_prediction_decisions(predictions))
        evaluation = evaluate_predictions(gold_cases, prediction_items)
    except ValueError as exc:
        typer.secho(f"invalid evaluation input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    payload = evaluation.as_dict()

    typer.echo(json.dumps(payload, indent=2))
    if report is not None:
        settings = get_settings()
        with atomic_text_writer(
            report,
            dir_mode=settings.output_dir_mode,
            file_mode=settings.output_file_mode,
        ) as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"wrote {report}", err=True)

    if not evaluation.complete:
        typer.secho(
            "evaluation is incomplete: "
            f"{len(evaluation.missing_prediction_ids)} missing prediction(s), "
            f"{len(evaluation.unexpected_prediction_ids)} unexpected prediction(s)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("cache-info")
def cache_info(cache_dir: CacheOption = None) -> None:
    """Summarize the provider response cache."""
    settings = get_settings()
    cache = ResponseCache(cache_dir or settings.cache_dir)
    counts: dict[str, int] = {}
    releases: set[str] = set()
    total = 0
    for entry in cache.entries():
        total += 1
        counts[f"{entry.provider}.{entry.operation}"] = (
            counts.get(f"{entry.provider}.{entry.operation}", 0) + 1
        )
        release = entry.metadata.get("terminology_release")
        if release:
            releases.add(str(release))

    typer.echo(f"cache: {cache.root}")
    typer.echo(f"entries: {total}")
    if releases:
        typer.echo(f"terminology releases: {', '.join(sorted(releases))}")
    for name in sorted(counts):
        typer.echo(f"  {name}: {counts[name]}")


# -- helpers ---------------------------------------------------------------


def _pipeline(profile: str, mode: ResourceMode | None, cache_dir: Path | None) -> BioSemAlign:
    try:
        return BioSemAlign.from_profile(
            profile, mode=mode, cache_dir=str(cache_dir) if cache_dir else None
        )
    except BioSemAlignError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _parse_context(pairs: list[str] | None) -> dict[str, str]:
    context: dict[str, str] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator:
            typer.secho(f"context must be key=value, got {pair!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        context[key.strip()] = value.strip()
    return context


def _read_rows(path: Path) -> Iterator[dict[str, Any] | BatchInputError]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield BatchInputError(
                        error_code="JSONDecodeError",
                        error_message=f"invalid JSON at source line {line_number}: {exc.msg}",
                    )
                    continue
                yield payload
        return
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=10_000):
            yield from batch.to_pylist()
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _validate_input_schema(path: Path, required_columns: list[str]) -> None:
    """Fail before output creation when a table's configured columns are absent."""
    duplicates = sorted(
        {column for column in required_columns if required_columns.count(column) > 1}
    )
    if duplicates:
        raise typer.BadParameter(
            "value, identifier, and context columns must be distinct; repeated: "
            + ", ".join(repr(column) for column in duplicates)
        )

    suffix = path.suffix.casefold()
    available: list[str] | None = None
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    available = list(payload)
                    break
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        available = list(pq.ParquetFile(path).schema_arrow.names)
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            available = next(reader, [])
        if len(available) != len(set(available)):
            raise typer.BadParameter("input CSV contains duplicate column names")

    if available is None:
        return
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise typer.BadParameter(
            "configured input column(s) are absent: "
            + ", ".join(repr(column) for column in missing)
        )


def _require_unchanged_input(path: Path, *, expected_sha256: str) -> None:
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise typer.BadParameter(
            "input became unavailable during batch processing; partial output was retained"
        ) from exc
    if actual != expected_sha256:
        raise typer.BadParameter(
            "input content changed during batch processing; partial output was retained"
        )


def _write_checkpoint(
    path: Path,
    payload: dict[str, Any],
    *,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
) -> None:
    with atomic_text_writer(path, dir_mode=dir_mode, file_mode=file_mode) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise typer.BadParameter(f"cannot resume: {path} does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"cannot resume: invalid checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != "2.0.0":
        raise typer.BadParameter(f"cannot resume: unsupported checkpoint {path}")
    return payload


def _scan_partial(
    path: Path,
) -> tuple[
    int,
    dict[str, int],
    dict[str, int],
    str | None,
    list[ResourceProvenance],
]:
    processed_rows = 0
    counts: dict[str, int] = {}
    routing: dict[str, int] = {}
    run_ids: set[str] = set()
    resources: list[ResourceProvenance] = []
    if not path.is_file():
        return processed_rows, counts, routing, None, resources

    expected_row = 1
    for record in _read_partial_records(path):
        if record.row_number != expected_row:
            raise typer.BadParameter(
                "cannot resume: partial output row sequence is not contiguous; "
                f"expected {expected_row}, found {record.row_number}"
            )
        processed_rows = record.row_number
        expected_row += 1
        counts[record.status] = counts.get(record.status, 0) + 1
        if record.resource_provenance:
            next_resources = [
                resource.model_copy(deep=True) for resource in record.resource_provenance
            ]
            _validate_resource_progress(resources, next_resources, row_number=record.row_number)
            resources = next_resources
        if record.artifact is not None:
            run_ids.add(record.artifact.run_id)
            route = record.artifact.decision.routing_status.value
            routing[route] = routing.get(route, 0) + 1
    if len(run_ids) > 1:
        raise typer.BadParameter(f"cannot resume: partial output contains run IDs {run_ids}")
    return processed_rows, counts, routing, next(iter(run_ids), None), resources


def _validate_resource_progress(
    previous: list[ResourceProvenance],
    current: list[ResourceProvenance],
    *,
    row_number: int,
) -> None:
    before = {
        (resource.resource_name.casefold(), resource.provider_name.casefold()): resource
        for resource in previous
    }
    after = {
        (resource.resource_name.casefold(), resource.provider_name.casefold()): resource
        for resource in current
    }
    if before.keys() - after.keys():
        raise typer.BadParameter(
            f"cannot resume: resource provenance disappears at row {row_number}"
        )
    for identity, earlier in before.items():
        later = after[identity]
        if (
            earlier.provider_version != later.provider_version
            or earlier.terminology_release != later.terminology_release
            or earlier.api_version != later.api_version
            or earlier.mode is not later.mode
        ):
            raise typer.BadParameter(
                f"cannot resume: resource identity changes at row {row_number}"
            )
        if later.cache_hits < earlier.cache_hits or later.network_calls < earlier.network_calls:
            raise typer.BadParameter(
                f"cannot resume: resource counters decrease at row {row_number}"
            )


def _read_partial_records(path: Path) -> Iterator[BatchAlignmentRecord]:
    try:
        yield from read_models(path, BatchAlignmentRecord)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            "cannot resume: partial output is corrupt or violates the batch contract"
        ) from exc


def _checkpoint_resources(
    state: dict[str, Any], *, fallback: list[ResourceProvenance]
) -> list[ResourceProvenance]:
    raw = state.get("resource_provenance")
    if raw is None:
        return fallback
    if not isinstance(raw, list):
        raise typer.BadParameter("cannot resume: checkpoint resource provenance is invalid")
    try:
        return [ResourceProvenance.model_validate(item) for item in raw]
    except ValueError as exc:
        raise typer.BadParameter(
            f"cannot resume: checkpoint resource provenance is invalid: {exc}"
        ) from exc


def _merged_resource_payload(
    prior: list[ResourceProvenance], current: list[ResourceProvenance]
) -> list[dict[str, Any]]:
    try:
        merged = merge_resource_provenance(prior, current)
    except ValueError as exc:
        raise typer.BadParameter(f"cannot continue mixed-resource run: {exc}") from exc
    return [resource.model_dump(mode="json") for resource in merged]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:  # pragma: no cover - unsupported on some filesystems
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _crosswalk_rows(path: Path) -> Iterator[dict[str, Any]]:
    for record in read_models(path, BatchAlignmentRecord):
        if record.status == "success" and record.artifact is not None:
            yield decision_to_row(record.artifact.decision)


def _read_prediction_decisions(
    path: Path,
) -> Iterator[AlignmentDecision | EvaluationPrediction]:
    """Read legacy decisions while retaining context from complete artifacts."""
    for payload in read_jsonl(path):
        if "status" in payload and "row_number" in payload:
            record = BatchAlignmentRecord.model_validate(payload)
            if record.status == "success" and record.artifact is not None:
                yield EvaluationPrediction.from_artifact(record.artifact)
            continue
        if "artifact_id" in payload and "sau" in payload:
            yield EvaluationPrediction.from_artifact(AlignmentArtifact.model_validate(payload))
            continue
        yield AlignmentDecision.model_validate(payload)


def _print_summary(decisions: list[AlignmentDecision]) -> None:
    routing: dict[str, int] = {}
    for decision in decisions:
        routing[decision.routing_status.value] = routing.get(decision.routing_status.value, 0) + 1
    typer.echo("  routing: " + ", ".join(f"{k}={v}" for k, v in sorted(routing.items())))


def _print_batch_summary(counts: dict[str, int], routing: dict[str, int]) -> None:
    typer.echo("  outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if routing:
        typer.echo("  routing: " + ", ".join(f"{k}={v}" for k, v in sorted(routing.items())))


def main() -> None:  # pragma: no cover - console-script entry point
    try:
        app()
    except BioSemAlignError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
