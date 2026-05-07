# pyright: strict

"""Benchmark run-state JSON and JSONL codec."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from ..config.models import WorkflowTask
from ..config.workflow_snapshots import (
    hydrate_workflow_config_snapshot,
    workflow_config_snapshot_payload,
)
from ..core.errors import SpiceOperatorError
from .plan_materialization import (
    BenchmarkDependencyLedger,
    BenchmarkPlanEntry,
    BenchmarkRootFacts,
    BenchmarkRootLedger,
    BenchmarkSelectionLedger,
)

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


class BenchmarkPlanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    dependencies: BenchmarkDependencyLedger
    dimension_labels: dict[str, str]
    selection: BenchmarkSelectionLedger
    root_facts: BenchmarkRootFacts
    root_ledger: BenchmarkRootLedger
    config: dict[str, object]

    @classmethod
    def from_entry(cls, entry: BenchmarkPlanEntry) -> BenchmarkPlanRecord:
        return cls(
            run_id=entry.run_id,
            case_id=entry.case_id,
            step_id=entry.step_id,
            workflow=entry.workflow,
            dependencies=entry.dependencies,
            dimension_labels=dict(entry.dimension_labels),
            selection=entry.selection,
            root_facts=entry.root_facts,
            root_ledger=entry.root_ledger,
            config=workflow_config_snapshot_payload(entry.config),
        )

    def to_entry(self) -> BenchmarkPlanEntry:
        return BenchmarkPlanEntry(
            run_id=self.run_id,
            case_id=self.case_id,
            step_id=self.step_id,
            workflow=self.workflow,
            dependencies=self.dependencies,
            dimension_labels=dict(self.dimension_labels),
            selection=self.selection,
            root_facts=self.root_facts,
            root_ledger=self.root_ledger,
            config=hydrate_workflow_config_snapshot(self.workflow, self.config),
        )


_METADATA_ADAPTER = TypeAdapter(BenchmarkRunMetadata)
_PLAN_RECORD_ADAPTER = TypeAdapter(BenchmarkPlanRecord)
_SUBMISSION_RECORD_ADAPTER = TypeAdapter(BenchmarkSubmissionRecord)


def write_run_metadata(run_dir: Path, metadata: BenchmarkRunMetadata) -> None:
    write_json(run_dir / METADATA_FILENAME, _model_json_payload(metadata))


def load_run_metadata(run_dir: Path) -> BenchmarkRunMetadata:
    return _read_json_model(run_dir / METADATA_FILENAME, _METADATA_ADAPTER)


def write_plan_jsonl(run_dir: Path, entries: list[BenchmarkPlanEntry]) -> None:
    write_jsonl(
        run_dir / PLAN_FILENAME,
        [_model_json_payload(BenchmarkPlanRecord.from_entry(entry)) for entry in entries],
    )


def load_plan_jsonl(run_dir: Path) -> list[BenchmarkPlanEntry]:
    return [
        record.to_entry()
        for record in _read_jsonl_model(run_dir / PLAN_FILENAME, _PLAN_RECORD_ADAPTER)
    ]


def append_submission_jsonl(run_dir: Path, record: BenchmarkSubmissionRecord) -> None:
    append_jsonl(run_dir / SUBMISSION_FILENAME, _model_json_payload(record))


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


def write_collection_snapshot(run_dir: Path, snapshot: object) -> None:
    payload = _model_json_payload(cast(BaseModel, snapshot))
    write_json_atomic(collection_snapshot_path(run_dir), payload)


def load_collection_snapshot(run_dir: Path):
    from .result_records import BenchmarkCollectionSnapshot

    return _read_json_model(
        collection_snapshot_path(run_dir),
        TypeAdapter(BenchmarkCollectionSnapshot),
    )


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


def _model_json_payload(model: BaseModel) -> dict[str, object]:
    payload = model.model_dump(mode="json")
    return cast(dict[str, object], payload)
