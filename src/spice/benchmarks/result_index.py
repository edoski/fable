# pyright: strict

"""Benchmark Result Index rebuild and update operations."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from ..core.files import remove_path
from .result_records import BenchmarkCollectionSnapshot
from .result_store import (
    BENCHMARK_RESULT_INDEX_PATH,
    ensure_result_index,
    index_counts,
    upsert_collection_snapshot,
)
from .runs import BENCHMARK_RUNS_ROOT, collection_snapshot_path, load_collection_snapshot


def upsert_benchmark_collection_snapshot(
    snapshot: BenchmarkCollectionSnapshot,
    *,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
) -> None:
    upsert_collection_snapshot(index_path, snapshot)


def rebuild_benchmark_result_index(
    *,
    runs_root: Path = BENCHMARK_RUNS_ROOT,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
) -> dict[str, int]:
    temp_path = index_path.parent / f".{index_path.name}.rebuild.{uuid4().hex}.tmp"
    remove_path(temp_path)
    try:
        ensure_result_index(temp_path)
        for run_dir in _benchmark_run_dirs(runs_root):
            if not collection_snapshot_path(run_dir).is_file():
                continue
            upsert_collection_snapshot(temp_path, load_collection_snapshot(run_dir))
        index_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, index_path)
        return index_counts(index_path)
    except Exception:
        remove_path(temp_path)
        raise


def benchmark_result_index_counts(
    *,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
) -> dict[str, int]:
    return index_counts(index_path)


def _benchmark_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        candidate
        for benchmark_dir in runs_root.iterdir()
        if benchmark_dir.is_dir()
        for candidate in benchmark_dir.iterdir()
        if candidate.is_dir()
    )
