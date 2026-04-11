from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from hydra import compose, initialize_config_dir

from spice.core.config import ExperimentConfig, coerce_config
from spice.core.constants import EVALUATION_START_TS

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "src" / "spice" / "conf"


def compose_experiment(config_name: str, *, overrides: list[str] | None = None) -> ExperimentConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return coerce_config(cfg, task=config_name)


def base_overrides(tmp_path: Path) -> list[str]:
    return [
        f"runtime.output_root={tmp_path / 'artifacts'}",
        "tracking.enabled=false",
        "training.device=cpu",
        "training.max_epochs=1",
        "training.effective_batch_size=8",
        "training.early_stopping_patience=1",
        "training.log_every_n_steps=1",
        "simulation.window_seconds=600",
        "simulation.arrival_rate_per_second=0.02",
        "simulation.repetitions=3",
        "pull.enrich_batch_size=1000",
        "pull.max_methods_per_second=1000000",
        "lookback_seconds=120",
        "target_anchor_count=48",
    ]


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
