"""Utilities for filling missing gas_limit values in block files."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from spice_temporal.contracts import EnrichedBlockRow, RawBlockRow
from spice_temporal.io import (
    build_enriched_block_row,
    has_missing_gas_limit,
    iter_block_files,
    load_rows,
    parse_enriched_block_row,
    write_rows,
)

FetchGasLimits = Callable[[list[int]], dict[int, int]]


def enrich_rows_with_gas_limit(
    rows: list[RawBlockRow],
    *,
    fetch_gas_limits: FetchGasLimits,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> list[EnrichedBlockRow]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_methods_per_second <= 0:
        raise ValueError("max_methods_per_second must be positive")

    missing_block_numbers = [
        int(row["block_number"])
        for row in rows
        if has_missing_gas_limit(row)
    ]
    if not missing_block_numbers:
        return [parse_enriched_block_row(row) for row in rows]

    lookup: dict[int, int] = {}
    for offset in range(0, len(missing_block_numbers), batch_size):
        batch = missing_block_numbers[offset : offset + batch_size]
        start = time.monotonic()
        lookup.update(fetch_gas_limits(batch))
        elapsed = time.monotonic() - start
        target_elapsed = len(batch) / max_methods_per_second
        if elapsed < target_elapsed:
            time.sleep(target_elapsed - elapsed)

    enriched_rows: list[EnrichedBlockRow] = []
    for row in rows:
        block_number = int(row["block_number"])
        if has_missing_gas_limit(row):
            gas_limit = lookup[block_number]
        else:
            gas_limit = parse_enriched_block_row(row)["gas_limit"]
        enriched_rows.append(build_enriched_block_row(row, gas_limit=gas_limit))
    return enriched_rows


def enrich_file(
    input_path: Path,
    output_path: Path,
    *,
    fetch_gas_limits: FetchGasLimits,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> None:
    rows = load_rows(input_path)
    enriched = enrich_rows_with_gas_limit(
        rows,
        fetch_gas_limits=fetch_gas_limits,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )
    write_rows(output_path, enriched)


def enrich_path(
    input_path: Path,
    output_path: Path,
    *,
    fetch_gas_limits: FetchGasLimits,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> list[Path]:
    files = iter_block_files(input_path)
    written_files: list[Path] = []
    if input_path.is_file():
        enrich_file(
            input_path,
            output_path,
            fetch_gas_limits=fetch_gas_limits,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
        )
        return [output_path]

    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        destination = output_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        enrich_file(
            file_path,
            destination,
            fetch_gas_limits=fetch_gas_limits,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
        )
        written_files.append(destination)
    return written_files
