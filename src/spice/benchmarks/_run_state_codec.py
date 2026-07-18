# pyright: strict

"""Benchmark run-state JSON and JSONL codec."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from ..config.models import WorkflowTask
from ..core.errors import SpiceOperatorError
from .plan_materialization import BenchmarkPlanEntry

PLAN_FILENAME = "plan.jsonl"
SUBMISSION_FILENAME = "submission.jsonl"
METADATA_FILENAME = "metadata.json"
COLLECTION_FILENAME = "collection.json"

RecordT = TypeVar("RecordT")


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


_METADATA_ADAPTER = TypeAdapter(BenchmarkRunMetadata)
_PLAN_RECORD_ADAPTER = TypeAdapter(BenchmarkPlanEntry)
_SUBMISSION_RECORD_ADAPTER = TypeAdapter(BenchmarkSubmissionRecord)


def load_run_metadata(run_dir: Path) -> BenchmarkRunMetadata:
    return _read_json_model(run_dir / METADATA_FILENAME, _METADATA_ADAPTER)


def load_plan_jsonl(run_dir: Path) -> list[BenchmarkPlanEntry]:
    return _read_jsonl_model(run_dir / PLAN_FILENAME, _PLAN_RECORD_ADAPTER)


def load_submission_jsonl(run_dir: Path) -> dict[str, BenchmarkSubmissionRecord]:
    records: dict[str, BenchmarkSubmissionRecord] = {}
    path = run_dir / SUBMISSION_FILENAME
    if not path.is_file():
        return records
    for record in _read_jsonl_model(path, _SUBMISSION_RECORD_ADAPTER):
        records[record.run_id] = record
    return records


def collection_snapshot_path(run_dir: Path) -> Path:
    return run_dir / COLLECTION_FILENAME


def load_collection_snapshot(run_dir: Path):
    from .result_records import BenchmarkCollectionSnapshot

    return _read_json_model(
        collection_snapshot_path(run_dir),
        TypeAdapter(BenchmarkCollectionSnapshot),
    )


def _read_json_model(path: Path, adapter: TypeAdapter[RecordT]) -> RecordT:
    if not path.is_file():
        raise SpiceOperatorError(f"Missing benchmark run file: {path}")
    try:
        return adapter.validate_json(path.read_text(encoding="utf-8"), strict=True)
    except ValidationError as exc:
        raise SpiceOperatorError(f"Invalid benchmark run file {path}: {exc}") from exc


def _read_jsonl_model(path: Path, adapter: TypeAdapter[RecordT]) -> list[RecordT]:
    if not path.is_file():
        raise SpiceOperatorError(f"Missing benchmark run file: {path}")
    rows: list[RecordT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(adapter.validate_json(line, strict=True))
        except ValidationError as exc:
            raise SpiceOperatorError(
                f"Invalid benchmark run file {path}:{line_number}: {exc}"
            ) from exc
    return rows
