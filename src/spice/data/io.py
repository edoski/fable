"""Parquet-only block dataset IO."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .block_schema import (
    ENRICHED_BLOCK_COLUMNS,
    validate_enriched_block_frame,
    validate_enriched_block_schema,
)


def _is_hidden_relative_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def iter_block_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported block file format: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Block dataset path does not exist: {path}")

    files = [
        candidate
        for candidate in path.rglob("*.parquet")
        if candidate.is_file() and not _is_hidden_relative_path(candidate.relative_to(path))
    ]
    if not files:
        raise ValueError(f"No parquet block files found under {path}")
    return sorted(files)


def scan_block_dataset(path: Path, *, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    files = iter_block_files(path)
    frame = pl.scan_parquet([str(file_path) for file_path in files])
    return frame.select(list(columns)) if columns is not None else frame


def read_block_dataset(path: Path, *, columns: Sequence[str] | None = None) -> pl.DataFrame:
    return scan_block_dataset(path, columns=columns).collect()


def write_block_file(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def load_enriched_block_frame(path: Path) -> pl.DataFrame:
    files = iter_block_files(path)
    for file_path in files:
        validate_enriched_block_schema(
            pl.read_parquet_schema(file_path),
            context=f"Block file {file_path}",
        )

    frame = read_block_dataset(path, columns=ENRICHED_BLOCK_COLUMNS).sort("block_number")
    validate_enriched_block_frame(frame)
    return frame
