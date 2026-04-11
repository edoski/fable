"""Canonical raw dataset normalization for cryo pulls."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

from ..data.io import read_block_dataset, write_block_file
from ..data.validation import validate_exact_window_frame


def _trim_frame_to_window(
    frame: pl.DataFrame,
    *,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> pl.DataFrame:
    timestamps = frame["timestamp"].cast(pl.Int64).to_list()
    in_window = [
        expected_start_timestamp <= timestamp < expected_end_timestamp for timestamp in timestamps
    ]
    if not any(in_window):
        raise ValueError("Raw block dataset produced no rows inside the requested timestamp window")

    start_index = next(index for index, value in enumerate(in_window) if value)
    end_index = len(in_window) - 1 - next(
        index for index, value in enumerate(reversed(in_window)) if value
    )
    if any(not value for value in in_window[start_index : end_index + 1]):
        raise ValueError("Detected out-of-window timestamps inside the requested block window")

    return frame.slice(start_index, end_index - start_index + 1)


def _validate_contiguous_block_numbers(frame: pl.DataFrame) -> None:
    block_numbers = frame["block_number"].cast(pl.Int64).to_list()
    for left, right in zip(block_numbers, block_numbers[1:], strict=False):
        if right == left:
            raise ValueError("Detected duplicate block_number rows in raw block dataset")
        if right != left + 1:
            raise ValueError("Detected non-contiguous block_number rows in raw block dataset")


def _validate_chain_id(frame: pl.DataFrame, *, expected_chain_id: int) -> None:
    chain_ids = sorted(set(frame["chain_id"].cast(pl.Int64).to_list()))
    if len(chain_ids) != 1:
        raise ValueError("Raw block dataset must contain exactly one chain_id")
    if chain_ids[0] != expected_chain_id:
        raise ValueError(
            f"Raw block dataset chain_id mismatch: expected {expected_chain_id}, got {chain_ids[0]}"
        )


def _rewrite_chunked_dataset(
    frame: pl.DataFrame,
    *,
    output_dir: Path,
    chain_name: str,
    chunk_size: int,
) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for offset in range(0, frame.height, chunk_size):
        chunk = frame.slice(offset, min(chunk_size, frame.height - offset))
        start_block = int(chunk["block_number"][0])
        end_block = int(chunk["block_number"][-1])
        destination = output_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet"
        write_block_file(destination, chunk)
        written.append(destination)
    return written


def normalize_raw_dataset(
    input_path: Path,
    output_path: Path,
    *,
    chain_name: str,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
    chunk_size: int,
) -> list[Path]:
    frame = read_block_dataset(input_path).sort("block_number")
    required_columns = ["block_number", "timestamp", "chain_id"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Raw block dataset is missing required columns: " + ", ".join(missing)
        )

    _validate_chain_id(frame, expected_chain_id=expected_chain_id)
    _validate_contiguous_block_numbers(frame)
    trimmed = _trim_frame_to_window(
        frame,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    report = validate_exact_window_frame(
        trimmed,
        dataset_path=output_path,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    if report.status != "clean":
        raise ValueError("; ".join(report.errors))

    return _rewrite_chunked_dataset(
        trimmed,
        output_dir=output_path,
        chain_name=chain_name,
        chunk_size=chunk_size,
    )
