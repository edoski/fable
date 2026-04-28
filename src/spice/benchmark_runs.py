# pyright: strict

"""Benchmark run-state and ledger collection helpers."""

from __future__ import annotations

import csv
import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from .config.benchmarks import BenchmarkPlanEntry
from .config.models import EvaluateConfig, WorkflowTask
from .config.resolution import WorkflowConfig, hydrate_model_workflow_config
from .core.errors import (
    ConfigResolutionError,
    SelectorResolutionError,
    SpiceOperatorError,
)
from .core.validation import validate_path_segment
from .execution.slurm_ssh import (
    ensure_execution_success,
    load_execution_target,
    run_execution_command,
)
from .modeling.results import LoadedEvaluationSummary, LoadedTrainingSummary
from .modeling.tuning import apply_study_best_params
from .storage.artifact import list_evaluation_summaries, load_training_summary
from .storage.layout import resolve_workflow_paths
from .storage.roots import ArtifactSelector, StudySelector
from .storage.sync import pull_artifact_from_cluster, pull_study_from_cluster

BENCHMARK_RUNS_ROOT = Path("outputs") / "benchmarks" / "runs"
BENCHMARK_LEDGER_PATH = Path("benchmarks") / "results.csv"
PLAN_FILENAME = "plan.jsonl"
SUBMISSION_FILENAME = "submission.jsonl"
METADATA_FILENAME = "metadata.json"

LEDGER_COLUMNS = (
    "recorded_at_utc",
    "git_commit",
    "execution_ref",
    "artifact_id",
    "evaluation_storage_id",
    "chain",
    "dataset",
    "surface",
    "features",
    "model",
    "problem",
    "prediction",
    "objective",
    "evaluation",
    "delay_seconds",
    "variant",
    "study",
    "sample_count",
    "total_events",
    "profit_over_baseline",
    "cost_over_optimum",
    "baseline_cost_over_optimum",
    "total_loss",
    "offset_accuracy",
    "macro_f1",
    "classification_loss",
    "regression_loss",
    "log_fee_mae",
    "log_fee_mse",
    "exact_optimum_hit_rate",
    "notes",
)


class BenchmarkRunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str
    created_at_utc: str
    target: str
    git_commit: str


class BenchmarkSubmissionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workflow: WorkflowTask
    job_id: str
    execution_ref: str
    git_commit: str
    dependency: str | None
    log_path: str


class BenchmarkCollectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workflow: WorkflowTask
    status: Literal["ready", "skipped", "missing"]
    row: dict[str, str] | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkPlanEntry:
    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    depends_on: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    selection: dict[str, object]
    config: WorkflowConfig


@dataclass(frozen=True, slots=True)
class CollectedEvaluationState:
    evaluation: LoadedEvaluationSummary
    training: LoadedTrainingSummary | None


def create_benchmark_run_dir(
    name: str,
    *,
    target: str,
    git_commit: str,
    runs_root: Path = BENCHMARK_RUNS_ROOT,
) -> Path:
    safe_name = validate_path_segment(name, label="benchmark name")
    created_at = _utc_now()
    base_dir = runs_root / safe_name / _timestamp_for_path(created_at)
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir.with_name(f"{base_dir.name}-{suffix}")
    run_dir.mkdir(parents=True)
    metadata = BenchmarkRunMetadata(
        benchmark=safe_name,
        created_at_utc=_format_datetime(created_at),
        target=target,
        git_commit=git_commit,
    )
    _write_json(run_dir / METADATA_FILENAME, metadata.model_dump(mode="json"))
    (run_dir / "collections").mkdir()
    return run_dir


def latest_benchmark_run_dir(
    name: str,
    *,
    runs_root: Path = BENCHMARK_RUNS_ROOT,
) -> Path:
    safe_name = validate_path_segment(name, label="benchmark name")
    root = runs_root / safe_name
    if not root.is_dir():
        raise SpiceOperatorError(f"No benchmark run directory found for {safe_name}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise SpiceOperatorError(f"No benchmark run directory found for {safe_name}")
    return candidates[-1]


def write_plan_jsonl(run_dir: Path, entries: list[BenchmarkPlanEntry]) -> None:
    _write_jsonl(run_dir / PLAN_FILENAME, [entry.to_json_dict() for entry in entries])


def append_submission_jsonl(run_dir: Path, record: BenchmarkSubmissionRecord) -> None:
    _append_jsonl(run_dir / SUBMISSION_FILENAME, record.model_dump(mode="json"))


def load_plan_jsonl(run_dir: Path) -> list[LoadedBenchmarkPlanEntry]:
    entries: list[LoadedBenchmarkPlanEntry] = []
    for payload in _read_jsonl(run_dir / PLAN_FILENAME):
        workflow = WorkflowTask(str(payload["workflow"]))
        config_payload = _mapping(payload["config"], label="config")
        entries.append(
            LoadedBenchmarkPlanEntry(
                run_id=str(payload["run_id"]),
                case_id=str(payload["case_id"]),
                step_id=str(payload["step_id"]),
                workflow=workflow,
                depends_on=tuple(
                    str(value) for value in _sequence(payload.get("depends_on", []))
                ),
                external_dependencies=tuple(
                    str(value)
                    for value in _sequence(payload.get("external_dependencies", []))
                ),
                selection=dict(_mapping(payload.get("selection", {}), label="selection")),
                config=hydrate_model_workflow_config(workflow, config_payload),
            )
        )
    return entries


def load_submission_jsonl(run_dir: Path) -> dict[str, BenchmarkSubmissionRecord]:
    records: dict[str, BenchmarkSubmissionRecord] = {}
    for payload in _read_jsonl(run_dir / SUBMISSION_FILENAME):
        record = BenchmarkSubmissionRecord.model_validate(payload)
        records[record.run_id] = record
    return records


def resolve_remote_git_commit(target_name: str) -> str:
    target = load_execution_target(target_name)
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    result = ensure_execution_success(
        run_execution_command(target, f"cd {repo_root} && git rev-parse HEAD"),
        action=f"read remote git commit for {target_name}",
    )
    commit = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not commit:
        raise SpiceOperatorError(f"read remote git commit for {target_name} returned no output")
    return commit


def collect_benchmark_run(
    *,
    run_dir: Path,
    target_name: str,
    ledger_path: Path = BENCHMARK_LEDGER_PATH,
    write: bool = False,
) -> list[BenchmarkCollectionRecord]:
    plan = load_plan_jsonl(run_dir)
    submissions = load_submission_jsonl(run_dir)
    existing_keys = _read_ledger_keys(ledger_path)
    records: list[BenchmarkCollectionRecord] = []
    collector_time = _utc_now()
    for entry in plan:
        if entry.workflow is not WorkflowTask.EVALUATE:
            continue
        if not isinstance(entry.config, EvaluateConfig):
            raise ConfigResolutionError(f"benchmark run {entry.run_id} is not an evaluate config")
        submission = submissions.get(entry.run_id)
        if submission is None:
            records.append(
                BenchmarkCollectionRecord(
                    run_id=entry.run_id,
                    workflow=entry.workflow,
                    status="missing",
                    reason="missing submission record",
                )
            )
            continue
        try:
            state = _pull_and_load_evaluation_state(
                entry.config,
                target_name=target_name,
            )
        except SelectorResolutionError as exc:
            records.append(
                BenchmarkCollectionRecord(
                    run_id=entry.run_id,
                    workflow=entry.workflow,
                    status="missing",
                    reason=str(exc),
                )
            )
            continue
        if state is None:
            records.append(
                BenchmarkCollectionRecord(
                    run_id=entry.run_id,
                    workflow=entry.workflow,
                    status="missing",
                    reason="evaluation summary not found",
                )
            )
            continue
        row = _ledger_row(
            entry=entry,
            state=state,
            submission=submission,
            collector_time=collector_time,
        )
        key = _ledger_key(row)
        if key in existing_keys:
            records.append(
                BenchmarkCollectionRecord(
                    run_id=entry.run_id,
                    workflow=entry.workflow,
                    status="skipped",
                    reason="ledger row already exists",
                    row=row,
                )
            )
            continue
        existing_keys.add(key)
        records.append(
            BenchmarkCollectionRecord(
                run_id=entry.run_id,
                workflow=entry.workflow,
                status="ready",
                row=row,
            )
        )
    _write_collection_jsonl(run_dir, records)
    if write:
        missing = [record for record in records if record.status == "missing"]
        if missing:
            raise SpiceOperatorError(
                f"Refusing partial benchmark ledger write: {len(missing)} evaluation rows missing"
            )
        _append_ledger_rows(
            ledger_path,
            [cast(dict[str, str], record.row) for record in records if record.status == "ready"],
        )
    return records


def _pull_and_load_evaluation_state(
    config: EvaluateConfig,
    *,
    target_name: str,
) -> CollectedEvaluationState | None:
    active_config = config
    study_id: str | None = None
    if config.artifact.variant.value == "tuned":
        study_paths = resolve_workflow_paths(config)
        if study_paths.study_id is None:
            raise ConfigResolutionError("tuned evaluation has no study identity")
        _pull_study_once(
            config,
            target_name=target_name,
            study_id=study_paths.study_id,
        )
        applied = apply_study_best_params(config)
        active_config = cast(EvaluateConfig, applied.config)
        study_id = applied.study_id
    paths = resolve_workflow_paths(active_config, study_id=study_id)
    if paths.artifact_id is None or paths.artifact_state_db is None:
        raise ConfigResolutionError("evaluation has no artifact identity")
    if active_config.evaluation is None:
        raise ConfigResolutionError("evaluation workflow requires evaluation")
    _pull_artifact_once(
        active_config,
        target_name=target_name,
        artifact_id=paths.artifact_id,
    )
    training_summary = load_training_summary(paths.artifact_state_db)
    summaries = [
        summary
        for summary in list_evaluation_summaries(paths.artifact_state_db)
        if summary.runtime.delay_seconds == active_config.delay_seconds
        and summary.runtime.evaluation_id == active_config.evaluation.id
    ]
    if not summaries:
        return None
    if len(summaries) > 1:
        raise SpiceOperatorError(
            f"Multiple evaluation summaries match artifact {paths.artifact_id}"
        )
    return CollectedEvaluationState(evaluation=summaries[0], training=training_summary)


def _pull_study_once(
    config: EvaluateConfig,
    *,
    target_name: str,
    study_id: str,
) -> None:
    pull_study_from_cluster(
        storage_root=config.storage.root,
        target_name=target_name,
        selector=StudySelector(
            study_id=study_id,
            chain_name=config.chain.name,
            dataset_name=config.dataset.name,
        ),
        replace=True,
    )


def _pull_artifact_once(
    config: EvaluateConfig,
    *,
    target_name: str,
    artifact_id: str,
) -> None:
    pull_artifact_from_cluster(
        storage_root=config.storage.root,
        target_name=target_name,
        selector=ArtifactSelector(
            artifact_id=artifact_id,
            chain_name=config.chain.name,
            dataset_name=config.dataset.name,
        ),
        replace=True,
    )


def _ledger_row(
    *,
    entry: LoadedBenchmarkPlanEntry,
    state: CollectedEvaluationState,
    submission: BenchmarkSubmissionRecord,
    collector_time: datetime,
) -> dict[str, str]:
    summary = state.evaluation
    manifest = summary.manifest
    runtime = summary.runtime
    metrics = (
        {}
        if state.training is None
        else dict(state.training.runtime.test_metrics.values)
    )
    metrics.update(runtime.metrics.values)
    recorded_at = (
        _format_datetime(datetime.fromtimestamp(summary.recorded_at, UTC))
        if summary.recorded_at > 0
        else _format_datetime(collector_time)
    )
    row = {
        "recorded_at_utc": recorded_at,
        "git_commit": submission.git_commit,
        "execution_ref": submission.execution_ref,
        "artifact_id": manifest.artifact_id,
        "evaluation_storage_id": summary.evaluation_id,
        "chain": manifest.chain_name,
        "dataset": manifest.dataset_name,
        "surface": str(entry.selection.get("surface", "")),
        "features": manifest.features_id,
        "model": manifest.model.id,
        "problem": manifest.problem_id,
        "prediction": manifest.prediction_id,
        "objective": str(entry.selection.get("objective", manifest.objective.id)),
        "evaluation": runtime.evaluation_id,
        "delay_seconds": str(runtime.delay_seconds),
        "variant": manifest.variant.value,
        "study": "" if manifest.study is None else manifest.study.name,
        "sample_count": str(runtime.sample_count),
        "total_events": str(runtime.total_events),
        "notes": "",
    }
    for column in LEDGER_COLUMNS:
        if column in row:
            continue
        row[column] = _metric_cell(metrics.get(column))
    return {column: row[column] for column in LEDGER_COLUMNS}


def _append_ledger_rows(ledger_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    _ensure_ledger_header(ledger_path)
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_COLUMNS), lineterminator="\n")
        for row in rows:
            writer.writerow({column: row[column] for column in LEDGER_COLUMNS})


def _read_ledger_keys(ledger_path: Path) -> set[tuple[str, str]]:
    if not ledger_path.exists():
        return set()
    _ensure_ledger_header(ledger_path)
    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {_ledger_key(row) for row in reader}


def _ensure_ledger_header(ledger_path: Path) -> None:
    if not ledger_path.exists():
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(LEDGER_COLUMNS)
        return
    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if tuple(header or ()) != LEDGER_COLUMNS:
        raise SpiceOperatorError(f"Benchmark ledger header mismatch: {ledger_path}")


def _ledger_key(row: dict[str, str]) -> tuple[str, str]:
    return row["artifact_id"], row["evaluation_storage_id"]


def _metric_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write_collection_jsonl(
    run_dir: Path,
    records: list[BenchmarkCollectionRecord],
) -> None:
    collection_dir = run_dir / "collections"
    collection_dir.mkdir(exist_ok=True)
    path = collection_dir / f"{_timestamp_for_path(_utc_now())}.jsonl"
    _write_jsonl(path, [record.model_dump(mode="json") for record in records])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SpiceOperatorError(f"Missing benchmark run file: {path}")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SpiceOperatorError(f"Invalid JSONL object in {path}")
        rows.append(cast(dict[str, object], payload))
    return rows


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SpiceOperatorError(f"benchmark {label} must be an object")
    return cast(dict[str, object], value)


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise SpiceOperatorError("benchmark JSONL field must be a list")
    return cast(list[object], value)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp_for_path(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
