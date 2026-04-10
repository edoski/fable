"""Utilities for filling missing gas_limit values in block parquet files."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import polars as pl

from ..core.console import NullReporter, Reporter
from ..data.io import iter_block_files, write_block_file

FetchGasLimits = Callable[[list[int]], dict[int, int]]


def _ensure_gas_limit_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "gas_limit" in frame.columns:
        return frame
    return frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("gas_limit"))


def _missing_gas_limit_expr() -> pl.Expr:
    return pl.col("gas_limit").is_null() | (pl.col("gas_limit") == 0)


def _missing_block_numbers(frame: pl.DataFrame) -> list[int]:
    return (
        frame.filter(_missing_gas_limit_expr())
        .select(pl.col("block_number").cast(pl.Int64))
        .get_column("block_number")
        .to_list()
    )


def enrich_frame_with_gas_limit(
    frame: pl.DataFrame,
    *,
    fetch_gas_limits: FetchGasLimits,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
    reporter: Reporter | None = None,
    completed_files: int = 0,
    total_files: int = 1,
    completed_blocks: int = 0,
    total_blocks: int = 0,
) -> tuple[pl.DataFrame, int]:
    reporter = reporter or NullReporter()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_methods_per_second <= 0:
        raise ValueError("max_methods_per_second must be positive")

    frame = _ensure_gas_limit_column(frame)
    missing_block_numbers = _missing_block_numbers(frame)
    if not missing_block_numbers:
        return frame.with_columns(pl.col("gas_limit").cast(pl.Int64)), 0

    lookup: dict[int, int] = {}
    fetched_count = 0
    for offset in range(0, len(missing_block_numbers), batch_size):
        batch = missing_block_numbers[offset : offset + batch_size]
        started_at = time.monotonic()
        lookup.update(fetch_gas_limits(batch))
        fetched_count += len(batch)
        reporter.update_enrich(
            completed_files=completed_files,
            completed_blocks=completed_blocks + fetched_count,
            total_files=total_files,
            total_blocks=total_blocks,
        )
        elapsed = time.monotonic() - started_at
        target_elapsed = len(batch) / max_methods_per_second
        if elapsed < target_elapsed:
            time.sleep(target_elapsed - elapsed)

    lookup_frame = pl.DataFrame(
        {
            "block_number": list(lookup.keys()),
            "fetched_gas_limit": list(lookup.values()),
        }
    )
    enriched = (
        frame.join(lookup_frame, on="block_number", how="left")
        .with_columns(pl.coalesce("gas_limit", "fetched_gas_limit").alias("gas_limit"))
        .drop("fetched_gas_limit")
        .with_columns(pl.col("gas_limit").cast(pl.Int64))
    )
    remaining_missing = enriched.filter(_missing_gas_limit_expr()).height
    if remaining_missing:
        raise RuntimeError(f"Missing gas_limit remained after enrichment: {remaining_missing}")
    return enriched, fetched_count


def count_missing_gas_limits(path: Path) -> tuple[int, int]:
    files = iter_block_files(path)
    total_missing = 0
    for file_path in files:
        frame = _ensure_gas_limit_column(pl.read_parquet(file_path))
        total_missing += len(_missing_block_numbers(frame))
    return len(files), total_missing


def enrich_path(
    input_path: Path,
    output_path: Path,
    *,
    fetch_gas_limits: FetchGasLimits,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
    reporter: Reporter | None = None,
) -> list[Path]:
    reporter = reporter or NullReporter()
    files = iter_block_files(input_path)
    total_files, total_blocks = count_missing_gas_limits(input_path)
    reporter.start_enrich(total_files=total_files, total_blocks=total_blocks)

    written_files: list[Path] = []
    completed_files = 0
    completed_blocks = 0
    if input_path.is_file():
        frame = pl.read_parquet(input_path)
        enriched, fetched_blocks = enrich_frame_with_gas_limit(
            frame,
            fetch_gas_limits=fetch_gas_limits,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
            reporter=reporter,
            completed_files=0,
            total_files=1,
            completed_blocks=0,
            total_blocks=total_blocks,
        )
        write_block_file(output_path, enriched)
        reporter.update_enrich(
            completed_files=1,
            completed_blocks=fetched_blocks,
            total_files=1,
            total_blocks=total_blocks,
        )
        reporter.finish_enrich(output_dir=output_path.parent)
        return [output_path]

    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        destination = output_path / relative_path
        frame = pl.read_parquet(file_path)
        enriched, fetched_blocks = enrich_frame_with_gas_limit(
            frame,
            fetch_gas_limits=fetch_gas_limits,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
            reporter=reporter,
            completed_files=completed_files,
            total_files=total_files,
            completed_blocks=completed_blocks,
            total_blocks=total_blocks,
        )
        write_block_file(destination, enriched)
        completed_files += 1
        completed_blocks += fetched_blocks
        reporter.update_enrich(
            completed_files=completed_files,
            completed_blocks=completed_blocks,
            total_files=total_files,
            total_blocks=total_blocks,
        )
        written_files.append(destination)

    reporter.finish_enrich(output_dir=output_path)
    return written_files
