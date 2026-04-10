from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from spice.core.constants import EVALUATION_START_TS


def build_test_config(output_root: Path) -> dict[str, Any]:
    return {
        "output_root": str(output_root),
        "max_delay_seconds": [36],
        "lookback_seconds": 120,
        "target_anchor_count": 48,
        "pull": {
            "requests_per_second": 10,
            "max_concurrent_requests": 2,
            "max_concurrent_chunks": 1,
            "chunk_size": 1000,
        },
        "split": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
        },
        "training": {
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "effective_batch_size": 8,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0001,
            "gradient_clip_norm": 1.0,
            "alpha": 1.0,
            "beta": 0.25,
            "device": "cpu",
            "seed": 2026,
        },
        "simulation": {
            "window_seconds": 600,
            "arrival_rate_per_second": 0.02,
            "repetitions": 3,
            "seed": 2026,
        },
        "chains": [
            {
                "name": "ethereum",
                "chain_id": 1,
                "block_time_seconds": 12.0,
                "history_days": 1,
            }
        ],
    }


def write_config(path: Path, *, output_root: Path) -> None:
    path.write_text(yaml.safe_dump(build_test_config(output_root)), encoding="utf-8")


def make_block_rows(
    count: int,
    *,
    start_block: int,
    start_timestamp: int,
    chain_id: int = 1,
    block_time_seconds: int = 12,
    include_gas_limit: bool = True,
    missing_gas_limit_blocks: set[int] | None = None,
) -> list[dict[str, int | None]]:
    missing_gas_limit_blocks = missing_gas_limit_blocks or set()
    rows: list[dict[str, int | None]] = []
    for offset in range(count):
        block_number = start_block + offset
        timestamp = start_timestamp + offset * block_time_seconds
        base_fee = int(
            1_000_000_000
            + 150_000_000 * math.sin(block_number / 7.0)
            + 60_000_000 * math.cos(block_number / 3.0)
        )
        gas_limit = None if block_number in missing_gas_limit_blocks else 30_000_000
        row: dict[str, int | None] = {
            "block_number": block_number,
            "timestamp": timestamp,
            "base_fee_per_gas": max(base_fee, 1),
            "gas_used": int(18_000_000 + 2_000_000 * math.sin(block_number / 5.0)),
            "chain_id": chain_id,
        }
        if include_gas_limit:
            row["gas_limit"] = gas_limit
        rows.append(row)
    return rows


def make_history_rows(count: int = 320) -> list[dict[str, int | None]]:
    return make_block_rows(
        count,
        start_block=1,
        start_timestamp=EVALUATION_START_TS - count * 12,
        include_gas_limit=True,
    )


def make_evaluation_rows(
    count: int = 180,
    *,
    start_block: int = 10_001,
) -> list[dict[str, int | None]]:
    return make_block_rows(
        count,
        start_block=start_block,
        start_timestamp=EVALUATION_START_TS,
        include_gas_limit=True,
    )


def write_parquet_rows(path: Path, rows: list[dict[str, int | None]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    return path


def write_dataset_dir(dataset_dir: Path, rows: list[dict[str, int | None]]) -> Path:
    return write_parquet_rows(dataset_dir / "blocks.parquet", rows)


def write_raw_chunk(
    dataset_dir: Path,
    *,
    chain_name: str,
    rows: list[dict[str, int | None]],
) -> Path:
    assert rows
    assert rows[0]["block_number"] is not None
    assert rows[-1]["block_number"] is not None
    start_block = int(rows[0]["block_number"])
    end_block = int(rows[-1]["block_number"])
    return write_parquet_rows(
        dataset_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet",
        rows,
    )
