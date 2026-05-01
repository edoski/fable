# pyright: strict

"""Benchmark run-state files."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from ..config.models import WorkflowTask
from ..config.workflow_snapshots import (
    hydrate_workflow_config_snapshot,
    workflow_config_snapshot_payload,
)
from ..core.errors import SpiceOperatorError
from ..core.validation import validate_path_segment
from .models import BenchmarkPlanEntry, LoadedBenchmarkPlanEntry

BENCHMARK_RUNS_ROOT = Path("outputs") / "benchmarks" / "runs"
PLAN_FILENAME = "plan.jsonl"
SUBMISSION_FILENAME = "submission.jsonl"
METADATA_FILENAME = "metadata.json"
COLLECTION_FILENAME = "collection.json"


class BenchmarkRunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str
    created_at_utc: str
    target: str


class BenchmarkSubmissionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workflow: WorkflowTask
    job_id: str
    execution_ref: str
    git_commit: str
    dependency: str | None
    log_path: str


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    metadata: BenchmarkRunMetadata
    plan: tuple[LoadedBenchmarkPlanEntry, ...]
    submissions: dict[str, BenchmarkSubmissionRecord]
    has_collection: bool


def create_benchmark_run_dir(
    name: str,
    *,
    target: str,
    runs_root: Path = BENCHMARK_RUNS_ROOT,
) -> Path:
    safe_name = validate_path_segment(name, label="benchmark name")
    created_at = utc_now()
    base_dir = runs_root / safe_name / timestamp_for_path(created_at)
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir.with_name(f"{base_dir.name}-{suffix}")
    run_dir.mkdir(parents=True)
    metadata = BenchmarkRunMetadata(
        benchmark=safe_name,
        created_at_utc=format_datetime(created_at),
        target=target,
    )
    write_json(run_dir / METADATA_FILENAME, metadata.model_dump(mode="json"))
    return run_dir


def load_benchmark_run(run_dir: Path) -> BenchmarkRun:
    return BenchmarkRun(
        run_dir=run_dir,
        metadata=load_run_metadata(run_dir),
        plan=tuple(load_plan_jsonl(run_dir)),
        submissions=load_submission_jsonl(run_dir),
        has_collection=collection_snapshot_path(run_dir).is_file(),
    )


def load_run_metadata(run_dir: Path) -> BenchmarkRunMetadata:
    return BenchmarkRunMetadata.model_validate(read_json(run_dir / METADATA_FILENAME))


def write_plan_jsonl(run_dir: Path, entries: list[BenchmarkPlanEntry]) -> None:
    write_jsonl(run_dir / PLAN_FILENAME, [plan_entry_json_dict(entry) for entry in entries])


def plan_entry_json_dict(entry: BenchmarkPlanEntry) -> dict[str, object]:
    return {
        "run_id": entry.run_id,
        "case_id": entry.case_id,
        "step_id": entry.step_id,
        "workflow": entry.workflow.value,
        "depends_on": list(entry.depends_on),
        "external_dependencies": list(entry.external_dependencies),
        "dimension_labels": dict(entry.dimension_labels),
        "selection": dict(entry.selection),
        "artifact_from": entry.artifact_from,
        "config": workflow_config_snapshot_payload(entry.config),
    }


def append_submission_jsonl(run_dir: Path, record: BenchmarkSubmissionRecord) -> None:
    append_jsonl(run_dir / SUBMISSION_FILENAME, record.model_dump(mode="json"))


def load_plan_jsonl(run_dir: Path) -> list[LoadedBenchmarkPlanEntry]:
    entries: list[LoadedBenchmarkPlanEntry] = []
    for payload in read_jsonl(run_dir / PLAN_FILENAME):
        workflow = WorkflowTask(str(payload["workflow"]))
        config_payload = mapping_payload(payload["config"], label="config")
        entries.append(
            LoadedBenchmarkPlanEntry(
                run_id=str(payload["run_id"]),
                case_id=str(payload["case_id"]),
                step_id=str(payload["step_id"]),
                workflow=workflow,
                depends_on=tuple(
                    str(value) for value in sequence_payload(payload.get("depends_on", []))
                ),
                external_dependencies=tuple(
                    str(value)
                    for value in sequence_payload(payload.get("external_dependencies", []))
                ),
                dimension_labels={
                    str(key): str(value)
                    for key, value in mapping_payload(
                        payload.get("dimension_labels", {}),
                        label="dimension_labels",
                    ).items()
                },
                selection=dict(mapping_payload(payload.get("selection", {}), label="selection")),
                artifact_from=_optional_string(payload.get("artifact_from")),
                config=hydrate_workflow_config_snapshot(workflow, config_payload),
            )
        )
    return entries


def load_submission_jsonl(run_dir: Path) -> dict[str, BenchmarkSubmissionRecord]:
    records: dict[str, BenchmarkSubmissionRecord] = {}
    path = run_dir / SUBMISSION_FILENAME
    if not path.is_file():
        return records
    for payload in read_jsonl(path):
        record = BenchmarkSubmissionRecord.model_validate(payload)
        records[record.run_id] = record
    return records


def collection_snapshot_path(run_dir: Path) -> Path:
    return run_dir / COLLECTION_FILENAME


def write_collection_snapshot(run_dir: Path, snapshot: object) -> None:
    payload = cast(BaseModel, snapshot).model_dump(mode="json")
    write_json_atomic(collection_snapshot_path(run_dir), payload)


def load_collection_snapshot(run_dir: Path):
    from .result_records import BenchmarkCollectionSnapshot

    return BenchmarkCollectionSnapshot.model_validate(read_json(collection_snapshot_path(run_dir)))


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SpiceOperatorError(f"Missing benchmark run file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpiceOperatorError(f"Invalid JSON object in {path}")
    return cast(dict[str, object], payload)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
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


def mapping_payload(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SpiceOperatorError(f"benchmark {label} must be an object")
    return cast(dict[str, object], value)


def sequence_payload(value: object) -> list[object]:
    if not isinstance(value, list):
        raise SpiceOperatorError("benchmark JSONL field must be a list")
    return cast(list[object], value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp_for_path(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
