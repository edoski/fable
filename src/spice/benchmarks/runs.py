# pyright: strict

"""Benchmark run-state files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ._run_state_codec import (
    BenchmarkRunMetadata,
    BenchmarkSubmissionRecord,
    collection_snapshot_path,
    load_collection_snapshot,
    load_plan_jsonl,
    load_run_metadata,
    load_submission_jsonl,
)
from .plan_materialization import BenchmarkPlanEntry

if TYPE_CHECKING:
    from .result_records import BenchmarkCollectionSnapshot

BENCHMARK_RUNS_ROOT = Path("outputs") / "benchmarks" / "runs"

__all__ = [
    "BENCHMARK_RUNS_ROOT",
    "BenchmarkRun",
    "BenchmarkRunMetadata",
    "BenchmarkSubmissionRecord",
    "format_datetime",
    "has_benchmark_collection_snapshot",
    "load_benchmark_collection_snapshot",
    "load_benchmark_run",
]


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    metadata: BenchmarkRunMetadata
    plan: tuple[BenchmarkPlanEntry, ...]
    submissions: dict[str, BenchmarkSubmissionRecord]
    has_collection: bool


def load_benchmark_run(run_dir: Path) -> BenchmarkRun:
    return BenchmarkRun(
        run_dir=run_dir,
        metadata=load_run_metadata(run_dir),
        plan=tuple(load_plan_jsonl(run_dir)),
        submissions=load_submission_jsonl(run_dir),
        has_collection=collection_snapshot_path(run_dir).is_file(),
    )


def load_benchmark_collection_snapshot(run_dir: Path) -> BenchmarkCollectionSnapshot:
    return load_collection_snapshot(run_dir)


def has_benchmark_collection_snapshot(run_dir: Path) -> bool:
    return collection_snapshot_path(run_dir).is_file()


def format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
