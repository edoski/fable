# pyright: strict

"""Benchmark run-state JSON and JSONL codec."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

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
ModelT = TypeVar("ModelT", bound=BaseModel)


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


def write_run_metadata(run_dir: Path, metadata: BenchmarkRunMetadata) -> None:
    write_json(run_dir / METADATA_FILENAME, metadata.model_dump(mode="json"))


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
        "dependencies": {
            "local_run_ids": list(entry.dependencies.local_run_ids),
            "external_slurm_dependencies": list(
                entry.dependencies.external_slurm_dependencies
            ),
            "artifact_from_run_id": entry.dependencies.artifact_from_run_id,
        },
        "dimension_labels": dict(entry.dimension_labels),
        "selection": entry.selection.model_dump(mode="json", exclude_none=True),
        "root_facts": entry.root_facts.model_dump(mode="json", exclude_none=True),
        "root_ledger": entry.root_ledger.model_dump(mode="json", exclude_none=True),
        "config": workflow_config_snapshot_payload(entry.config),
    }


def load_plan_jsonl(run_dir: Path) -> list[BenchmarkPlanEntry]:
    entries: list[BenchmarkPlanEntry] = []
    for payload in read_jsonl(run_dir / PLAN_FILENAME):
        workflow = WorkflowTask(string_payload(payload.get("workflow"), label="workflow"))
        config_payload = mapping_payload(payload["config"], label="config")
        entries.append(
            BenchmarkPlanEntry(
                run_id=string_payload(payload.get("run_id"), label="run_id"),
                case_id=string_payload(payload.get("case_id"), label="case_id"),
                step_id=string_payload(payload.get("step_id"), label="step_id"),
                workflow=workflow,
                dependencies=_benchmark_dependency_ledger(
                    mapping_payload(
                        payload.get("dependencies", {}),
                        label="dependencies",
                    )
                ),
                dimension_labels=string_mapping_payload(
                    mapping_payload(
                        payload.get("dimension_labels", {}),
                        label="dimension_labels",
                    ),
                    label="dimension_labels",
                ),
                selection=_plan_model_payload(
                    BenchmarkSelectionLedger,
                    mapping_payload(payload.get("selection", {}), label="selection"),
                    label="selection",
                ),
                root_facts=_plan_model_payload(
                    BenchmarkRootFacts,
                    mapping_payload(
                        required_payload(payload, "root_facts"),
                        label="root_facts",
                    ),
                    label="root_facts",
                ),
                root_ledger=_plan_model_payload(
                    BenchmarkRootLedger,
                    mapping_payload(
                        required_payload(payload, "root_ledger"),
                        label="root_ledger",
                    ),
                    label="root_ledger",
                ),
                config=hydrate_workflow_config_snapshot(workflow, config_payload),
            )
        )
    return entries


def append_submission_jsonl(run_dir: Path, record: BenchmarkSubmissionRecord) -> None:
    append_jsonl(run_dir / SUBMISSION_FILENAME, record.model_dump(mode="json"))


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


def _benchmark_dependency_ledger(payload: dict[str, object]) -> BenchmarkDependencyLedger:
    return BenchmarkDependencyLedger(
        local_run_ids=string_tuple_payload(
            payload.get("local_run_ids", []),
            label="dependencies.local_run_ids",
        ),
        external_slurm_dependencies=string_tuple_payload(
            payload.get("external_slurm_dependencies", []),
            label="dependencies.external_slurm_dependencies",
        ),
        artifact_from_run_id=_optional_string(
            payload.get("artifact_from_run_id"),
            label="dependencies.artifact_from_run_id",
        ),
    )


def _plan_model_payload(
    model_type: type[ModelT],
    payload: dict[str, object],
    *,
    label: str,
) -> ModelT:
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise SpiceOperatorError(f"Invalid benchmark {label}: {exc}") from exc


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


def required_payload(payload: dict[str, object], key: str) -> object:
    if key not in payload:
        raise SpiceOperatorError(f"benchmark {key} is required")
    return payload[key]


def sequence_payload(value: object) -> list[object]:
    if not isinstance(value, list):
        raise SpiceOperatorError("benchmark JSONL field must be a list")
    return cast(list[object], value)


def string_payload(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise SpiceOperatorError(f"benchmark {label} must be a string")
    return value


def string_tuple_payload(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(string_payload(item, label=f"{label} item") for item in sequence_payload(value))


def string_mapping_payload(value: dict[str, object], *, label: str) -> dict[str, str]:
    strings: dict[str, str] = {}
    for key, item in value.items():
        strings[key] = string_payload(item, label=f"{label}.{key}")
    return strings


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return string_payload(value, label=label)
