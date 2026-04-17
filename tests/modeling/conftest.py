from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from spice.config import TrainConfig, WorkflowTask
from spice.features import compile_feature_contract
from spice.temporal.contracts import compile_problem_contract

PRESET = "icdcs_2026"
TEST_EVALUATION_DATE = date(2025, 11, 9)


@pytest.fixture
def model_workflow_override():
    def _override(
        *,
        sample_count: int = 24,
        lookback_seconds: int = 120,
        max_delay_seconds: int = 36,
        delay_seconds: int | None = None,
        compiler_id: str = "estimated_block",
    ) -> dict[str, object]:
        feature_set_name = (
            "time_native_baseline" if compiler_id == "timestamp_native" else "icdcs_2026"
        )
        return {
            "chain": "ethereum",
            "model": "lstm",
            "dataset": {
                "evaluation_date": TEST_EVALUATION_DATE.isoformat(),
            },
            "feature_set": feature_set_name,
            "problem": {
                "id": "test_problem",
                "lookback_seconds": lookback_seconds,
                "sample_count": sample_count,
                "max_delay_seconds": max_delay_seconds,
                "compiler": {"id": compiler_id},
            },
            "delay_seconds": max_delay_seconds if delay_seconds is None else delay_seconds,
            "training": {
                "device": "cpu",
                "batch_size": 8,
                "max_epochs": 1,
                "log_every_n_steps": 1,
                "precision": "fp32",
                "compile": "off",
                "early_stopping": {
                    "patience": 1,
                    "min_delta": 0.0,
                },
            },
            "evaluation": {
                "evaluator": {
                    "id": "poisson_replay",
                    "window_seconds": 600,
                    "arrival_rate_per_second": 0.02,
                    "repetitions": 3,
                    "seed": 2026,
                }
            },
            "tuning": {
                "trial_count": 2,
                "enable_pruning": False,
            },
        }

    return _override


@pytest.fixture
def load_test_train_config(tmp_path: Path, load_workflow_config):
    def _load(
        tmp_path_arg: Path | None = None,
        *,
        override: dict[str, object] | None = None,
    ) -> TrainConfig:
        workspace = tmp_path if tmp_path_arg is None else tmp_path_arg
        return cast(
            TrainConfig,
            load_workflow_config(
                WorkflowTask.TRAIN,
                workspace=workspace,
                preset=PRESET,
                override=override,
            ),
        )

    return _load


@pytest.fixture
def seed_history_dataset():
    def _seed(config: TrainConfig) -> Path:
        return _write_dataset_dir(config.paths.history_dir, make_history_rows(config))

    return _seed


def synthetic_block_interval_seconds(chain_name: str) -> int:
    return {
        "ethereum": 12,
        "polygon": 2,
        "avalanche": 2,
    }.get(chain_name, 12)


def required_dataset_blocks(config: TrainConfig) -> int:
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


def make_history_rows(config: TrainConfig) -> list[dict[str, int]]:
    block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
    count = required_dataset_blocks(config)
    return make_block_rows(
        count,
        start_block=1,
        start_timestamp=config.evaluation_window_start_timestamp - count * block_interval_seconds,
        chain_id=config.chain.runtime.chain_id,
        block_interval_seconds=block_interval_seconds,
    )


def _write_dataset_dir(dataset_dir: Path, rows: list[dict[str, int]]) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "blocks.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path
