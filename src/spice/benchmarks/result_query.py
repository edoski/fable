# pyright: strict

"""Read-only Benchmark Result Query interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .result_store import BENCHMARK_RESULT_INDEX_PATH, list_indexed_results


@dataclass(frozen=True, slots=True)
class BenchmarkResultQuery:
    benchmark: str | None = None
    chain: str | None = None
    model: str | None = None
    evaluation: str | None = None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResultSummary:
    run_id: str
    artifact_id: str
    evaluation_storage_id: str
    chain: str
    model: str
    evaluation: str
    delay_seconds: int
    sample_count: int
    total_events: int


def list_benchmark_results(
    query: BenchmarkResultQuery,
    *,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
) -> list[BenchmarkResultSummary]:
    return [
        BenchmarkResultSummary(
            run_id=row.run_id,
            artifact_id=row.artifact_id,
            evaluation_storage_id=row.evaluation_storage_id,
            chain=row.chain_name,
            model=row.model_id,
            evaluation=row.evaluation_id,
            delay_seconds=row.delay_seconds,
            sample_count=row.sample_count,
            total_events=row.total_events,
        )
        for row in list_indexed_results(
            index_path,
            benchmark=query.benchmark,
            chain=query.chain,
            model=query.model,
            evaluation=query.evaluation,
            limit=query.limit,
        )
    ]
