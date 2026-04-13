from __future__ import annotations

import math
import shutil
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import yaml

from spice.config import (
    AcquireConfig,
    SimulateConfig,
    TrainConfig,
    TuneConfig,
    load_acquire_config,
    load_simulate_config,
    load_train_config,
    load_tune_config,
)
from spice.planning.contracts import resolve_task_contract

PRESET = "icdcs_2026"
TEST_EVALUATION_DATE = date(2025, 11, 9)
TEST_WINDOW_START_TIMESTAMP = int(
    datetime.combine(TEST_EVALUATION_DATE, time.min, tzinfo=UTC).timestamp()
)
TEST_WINDOW_END_TIMESTAMP = int(
    datetime.combine(TEST_EVALUATION_DATE + timedelta(days=1), time.min, tzinfo=UTC).timestamp()
)
CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "spice" / "conf"


def deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def write_override(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    name: str = "override.yaml",
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def isolate_conf_root(tmp_path: Path, monkeypatch) -> Path:
    from spice.config import registry

    target = tmp_path / "conf"
    shutil.copytree(CONF_ROOT, target)
    monkeypatch.setattr(registry, "_CONF_ROOT", target)
    return target


def model_workflow_override(
    *,
    sample_count: int = 24,
    lookback_seconds: int = 120,
    max_supported_delay_seconds: int = 36,
    requested_delay_seconds: int | None = None,
) -> dict[str, object]:
    return {
        "chain": "ethereum",
        "model": "lstm",
        "dataset": {
            "evaluation_date": TEST_EVALUATION_DATE.isoformat(),
        },
        "task": {
            "id": "test_task",
            "lookback_seconds": lookback_seconds,
            "sample_count": sample_count,
            "max_supported_delay_seconds": max_supported_delay_seconds,
        },
        "execution": {
            "id": "test_execution",
            "requested_delay_seconds": (
                max_supported_delay_seconds
                if requested_delay_seconds is None
                else requested_delay_seconds
            ),
        },
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
        "simulation": {
            "window_seconds": 600,
            "arrival_rate_per_second": 0.02,
            "repetitions": 3,
            "seed": 2026,
        },
        "tuning": {
            "trial_count": 2,
            "enable_pruning": False,
        },
    }


def tune_override() -> dict[str, object]:
    return {
        "tuning_space": {
            "training": {
                "learning_rate": [0.0001, 0.0003],
                "weight_decay": [0.0, 0.01],
            },
            "model": {
                "id": "lstm",
                "hidden_size": [64, 128],
                "dropout": [0.0, 0.1],
            },
        }
    }


def acquire_override(
    *,
    sample_count: int = 4,
    lookback_seconds: int = 24,
    max_supported_delay_seconds: int = 12,
) -> dict[str, object]:
    return {
        "chain": "ethereum",
        "dataset": {
            "evaluation_date": TEST_EVALUATION_DATE.isoformat(),
        },
        "task": {
            "id": "acquire_test_task",
            "lookback_seconds": lookback_seconds,
            "sample_count": sample_count,
            "max_supported_delay_seconds": max_supported_delay_seconds,
        },
        "acquisition": {
            "chunk_size": 64,
            "rpc": {
                "batch_size": 16,
                "concurrency": 8,
                "min_batch_size": 8,
                "concurrency_rungs": [8],
            },
        },
    }


def load_test_acquire_config(
    tmp_path: Path,
    *,
    override: dict[str, object] | None = None,
    chain: str | None = None,
    provider: str | None = None,
) -> AcquireConfig:
    config_path = (
        None
        if override is None
        else write_override(tmp_path, override, name="acquire.yaml")
    )
    return load_acquire_config(
        preset=PRESET,
        config_path=config_path,
        chain=chain,
        provider=provider,
        storage_root=tmp_path / "outputs",
    )


def load_test_train_config(
    tmp_path: Path,
    *,
    override: dict[str, object] | None = None,
) -> TrainConfig:
    config_path = (
        None
        if override is None
        else write_override(tmp_path, override, name="train.yaml")
    )
    return load_train_config(
        preset=PRESET,
        config_path=config_path,
        storage_root=tmp_path / "outputs",
    )


def load_test_tune_config(
    tmp_path: Path,
    *,
    override: dict[str, object] | None = None,
) -> TuneConfig:
    config_path = (
        None
        if override is None
        else write_override(tmp_path, override, name="tune.yaml")
    )
    return load_tune_config(
        preset=PRESET,
        config_path=config_path,
        storage_root=tmp_path / "outputs",
    )


def load_test_simulate_config(
    tmp_path: Path,
    *,
    override: dict[str, object] | None = None,
) -> SimulateConfig:
    config_path = (
        None
        if override is None
        else write_override(tmp_path, override, name="simulate.yaml")
    )
    return load_simulate_config(
        preset=PRESET,
        config_path=config_path,
        storage_root=tmp_path / "outputs",
    )


def synthetic_block_interval_seconds(chain_name: str) -> int:
    return {
        "ethereum": 12,
        "polygon": 2,
        "avalanche": 2,
    }.get(chain_name, 12)


def required_history_seconds(config: AcquireConfig) -> int:
    return resolve_task_contract(
        task=config.task,
        feature_set=config.feature_set,
    ).required_history_seconds


def required_dataset_blocks(config: TrainConfig | TuneConfig | SimulateConfig) -> int:
    contract = resolve_task_contract(
        task=config.task,
        feature_set=config.feature_set,
    )
    block_interval_seconds = synthetic_block_interval_seconds(config.chain.name)
    required_seconds = (
        contract.required_history_seconds
        + contract.max_supported_delay_seconds
        + contract.sample_count * block_interval_seconds
    )
    return max(64, math.ceil(required_seconds / block_interval_seconds) + 1)


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


def make_history_rows(config: TrainConfig | TuneConfig | SimulateConfig) -> list[dict[str, int]]:
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
    config: SimulateConfig,
    *,
    count: int = 64,
    start_block: int = 10_001,
) -> list[dict[str, int]]:
    return make_block_rows(
        count,
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


def seed_history_dataset(config: TrainConfig | TuneConfig | SimulateConfig) -> Path:
    return write_dataset_dir(config.paths.history_dir, make_history_rows(config))


def seed_evaluation_dataset(config: SimulateConfig) -> Path:
    return write_dataset_dir(config.paths.evaluation_dir, make_evaluation_rows(config))
