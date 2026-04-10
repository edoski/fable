from __future__ import annotations

from pathlib import Path

import yaml

from spice_temporal.constants import EVALUATION_START_TS
from spice_temporal.records import BlockRecord


def make_block(
    block_number: int,
    *,
    timestamp: int | None = None,
    base_fee_per_gas: int | None = None,
    gas_used: int | None = None,
    gas_limit: int = 30_000_000,
    chain_id: int = 1,
) -> BlockRecord:
    return BlockRecord(
        block_number=block_number,
        timestamp=timestamp if timestamp is not None else 1_700_000_000 + 12 * block_number,
        base_fee_per_gas=(
            base_fee_per_gas
            if base_fee_per_gas is not None
            else 100 + (block_number % 5)
        ),
        gas_used=gas_used if gas_used is not None else 15_000_000 + block_number,
        gas_limit=gas_limit,
        chain_id=chain_id,
    )


def make_history_block(index: int) -> BlockRecord:
    return make_block(
        index,
        timestamp=EVALUATION_START_TS - 12 * (420 - index),
        base_fee_per_gas=100 + ((index // 3) % 7),
    )


def make_evaluation_block(index: int) -> BlockRecord:
    return make_block(
        1_000 + index,
        timestamp=EVALUATION_START_TS + 12 * index,
        base_fee_per_gas=120 + ((index // 5) % 9),
        gas_used=15_100_000 + (index % 1000),
    )


def build_test_config(*, output_root: Path | None = None) -> dict[str, object]:
    return {
        "output_root": str(output_root or Path("./artifacts/test")),
        "max_delay_seconds": [36],
        "lookback_seconds": 600,
        "target_anchor_count": 64,
        "pull": {
            "requests_per_second": 10,
            "max_concurrent_requests": 2,
            "max_concurrent_chunks": 1,
        },
        "split": {
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
        },
        "training": {
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "effective_batch_size": 8,
            "max_epochs": 2,
            "early_stopping_patience": 2,
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


def write_config(path: Path, *, output_root: Path | None = None) -> None:
    path.write_text(yaml.safe_dump(build_test_config(output_root=output_root)), encoding="utf-8")
