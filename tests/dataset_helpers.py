from __future__ import annotations

import math

from spice.config import TrainConfig, TuneConfig
from spice.features import compile_feature_contract
from spice.temporal.contracts import compile_problem_contract

TrainTuneWorkflowConfig = TrainConfig | TuneConfig
BlockRowValue = int | float | None


def synthetic_block_interval_seconds(chain_name: str) -> int:
    return {
        "ethereum": 12,
        "polygon": 2,
        "avalanche": 2,
    }.get(chain_name, 12)


def required_dataset_blocks(config: TrainTuneWorkflowConfig) -> int:
    feature_contract = compile_feature_contract(features=config.features)
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
) -> list[dict[str, BlockRowValue]]:
    rows: list[dict[str, BlockRowValue]] = []
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
                "tx_count": 128,
                "block_size_bytes": None,
                "blob_gas_used": None,
                "excess_blob_gas": None,
                "priority_fee_p10": 1_000_000,
                "priority_fee_p50": 2_000_000,
                "priority_fee_p90": 4_000_000,
                "priority_fee_spread": 3_000_000,
                "chain_id": chain_id,
            }
        )
    return rows


def make_history_rows(config: TrainTuneWorkflowConfig) -> list[dict[str, BlockRowValue]]:
    block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
    count = required_dataset_blocks(config)
    return make_block_rows(
        count,
        start_block=1,
        start_timestamp=config.evaluation_window_start_timestamp - count * block_interval_seconds,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=block_interval_seconds,
    )
