from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from spice.config import EvaluateConfig, TrainConfig, TuneConfig
from spice.features import compile_feature_contract
from spice.temporal.contracts import compile_problem_contract

ModelWorkflowConfig = TrainConfig | TuneConfig | EvaluateConfig


def synthetic_block_interval_seconds(chain_name: str) -> int:
    return {
        "ethereum": 12,
        "polygon": 2,
        "avalanche": 2,
    }.get(chain_name, 12)


def required_dataset_blocks(config: ModelWorkflowConfig) -> int:
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
    required_blocks = (
        math.ceil(
            (
                contract.required_history_seconds
                + contract.max_delay_seconds
                + block_interval_seconds
            )
            / block_interval_seconds
        )
        + contract.warmup_rows
        + contract.sample_count
        + 8
    )
    return max(64, required_blocks)


def make_block_rows(
    count: int,
    *,
    start_block: int,
    start_timestamp: int,
    chain_id: int = 1,
    block_interval_seconds: int = 12,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for offset in range(count):
        block_number = start_block + offset
        timestamp = start_timestamp + offset * block_interval_seconds
        base_fee = int(
            1_000_000_000
            + 150_000_000 * math.sin(block_number / 2.0)
            + 150_000_000 * math.cos(block_number / 2.5)
            + 50_000_000 * math.sin(block_number / 7.0)
        )
        rows.append(
            {
                "block_number": block_number,
                "timestamp": timestamp,
                "base_fee_per_gas": max(base_fee, 1),
                "gas_used": int(18_000_000 + 2_000_000 * math.sin(block_number / 5.0)),
                "gas_limit": 30_000_000,
                "chain_id": chain_id,
            }
        )
    return rows


def make_history_rows(config: ModelWorkflowConfig) -> list[dict[str, int]]:
    block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
    count = required_dataset_blocks(config)
    return make_block_rows(
        count,
        start_block=1,
        start_timestamp=config.evaluation_window_start_timestamp - count * block_interval_seconds,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=block_interval_seconds,
    )


def make_evaluation_rows(
    config: EvaluateConfig,
    *,
    count: int | None = None,
    start_block: int = 10_001,
) -> list[dict[str, int]]:
    resolved_count = count
    if resolved_count is None:
        feature_contract = compile_feature_contract(feature_set=config.feature_set)
        contract = compile_problem_contract(
            problem=config.problem,
            feature_contract=feature_contract,
            chain_runtime=config.chain.runtime,
        )
        block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
        resolved_count = max(
            64,
            math.ceil(
                (
                    contract.required_history_seconds
                    + config.delay_seconds
                    + block_interval_seconds
                )
                / block_interval_seconds
            )
            + contract.warmup_rows
            + 8,
        )
    return make_block_rows(
        resolved_count,
        start_block=start_block,
        start_timestamp=config.evaluation_window_start_timestamp,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=synthetic_block_interval_seconds(config.chain.name),
    )


def write_dataset_dir(dataset_dir: Path, rows: list[dict[str, int]]) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "blocks.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path
