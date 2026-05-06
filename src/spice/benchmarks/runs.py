# pyright: strict

"""Benchmark run-state files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..core.validation import validate_path_segment
from .plan_materialization import BenchmarkPlanEntry
from .run_state_codec import (
    BenchmarkRunMetadata,
    BenchmarkSubmissionRecord,
    append_submission_jsonl,
    collection_snapshot_path,
    load_collection_snapshot,
    load_plan_jsonl,
    load_run_metadata,
    load_submission_jsonl,
    write_collection_snapshot,
    write_plan_jsonl,
    write_run_metadata,
)

BENCHMARK_RUNS_ROOT = Path("outputs") / "benchmarks" / "runs"

__all__ = [
    "BENCHMARK_RUNS_ROOT",
    "BenchmarkRun",
    "BenchmarkRunMetadata",
    "BenchmarkSubmissionRecord",
    "append_submission_jsonl",
    "collection_snapshot_path",
    "create_benchmark_run_dir",
    "format_datetime",
    "load_benchmark_run",
    "load_collection_snapshot",
    "load_plan_jsonl",
    "load_run_metadata",
    "load_submission_jsonl",
    "timestamp_for_path",
    "utc_now",
    "write_collection_snapshot",
    "write_plan_jsonl",
]


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    metadata: BenchmarkRunMetadata
    plan: tuple[BenchmarkPlanEntry, ...]
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
    write_run_metadata(run_dir, metadata)
    return run_dir


def load_benchmark_run(run_dir: Path) -> BenchmarkRun:
    return BenchmarkRun(
        run_dir=run_dir,
        metadata=load_run_metadata(run_dir),
        plan=tuple(load_plan_jsonl(run_dir)),
        submissions=load_submission_jsonl(run_dir),
        has_collection=collection_snapshot_path(run_dir).is_file(),
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp_for_path(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
