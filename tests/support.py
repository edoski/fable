from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from spice.core.config import (
    ChainConfig,
    ChainName,
    CompileMode,
    ExperimentConfig,
    LstmModelConfig,
    ModelConfig,
    ModelFamily,
    ProviderConfig,
    RpcProviderName,
    TrainingConfig,
    TrainingPrecision,
    TransformerLstmModelConfig,
    TransformerModelConfig,
    load_params_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_EVALUATION_DATE = date(2025, 11, 9)
TEST_WINDOW_START_TIMESTAMP = int(
    datetime.combine(TEST_EVALUATION_DATE, time.min, tzinfo=UTC).timestamp()
)
TEST_WINDOW_END_TIMESTAMP = int(
    datetime.combine(TEST_EVALUATION_DATE + timedelta(days=1), time.min, tzinfo=UTC).timestamp()
)


def compose_experiment(config_name: str, *, overrides: list[str] | None = None) -> ExperimentConfig:
    return load_params_config(
        config_name,
        params_path=REPO_ROOT / "params.yaml",
        overrides=overrides,
    )


def base_overrides(tmp_path: Path) -> list[str]:
    return [
        f"runtime.output_root={tmp_path / 'artifacts'}",
        "tracking.enabled=false",
        "training.device=cpu",
        "training.max_epochs=1",
        "training.batch_size=8",
        "training.early_stopping.patience=1",
        "training.log_every_n_steps=1",
        "simulation.window_seconds=600",
        "simulation.arrival_rate_per_second=0.02",
        "simulation.repetitions=3",
        "acquisition.rpc_batch_size=256",
        f"evaluation.date={TEST_EVALUATION_DATE}",
        "dataset.temporal.lookback_seconds=120",
        "dataset.sampling.sample_count=48",
        "acquisition.history_sample_budget=48",
    ]


def make_chain_config(*, uses_poa_extra_data: bool = False) -> ChainConfig:
    return ChainConfig(
        name=ChainName.ETHEREUM,
        chain_id=1,
        block_time_seconds=12.0,
        uses_poa_extra_data=uses_poa_extra_data,
    )


def make_provider_config(
    endpoint: str = "https://rpc.example.test",
    *,
    name: RpcProviderName = RpcProviderName.DIRECT,
) -> ProviderConfig:
    reference = "$ETHEREUM_RPC_URL" if name is RpcProviderName.DIRECT else endpoint
    return ProviderConfig(
        name=name,
        endpoints={ChainName.ETHEREUM: endpoint},
        references={ChainName.ETHEREUM: reference},
        timeout_seconds=30.0,
        retry_count=5,
        backoff_factor=0.125,
    )


def make_model_config(*, family: ModelFamily = ModelFamily.LSTM) -> ModelConfig:
    if family is ModelFamily.TRANSFORMER:
        return TransformerModelConfig(
            family=family,
            dropout=0.1,
            d_model=128,
            nhead=4,
            transformer_layers=2,
            feedforward_dim=512,
            head_hidden_dim=64,
        )
    if family is ModelFamily.TRANSFORMER_LSTM:
        return TransformerLstmModelConfig(
            family=family,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            d_model=128,
            nhead=4,
            transformer_layers=2,
            head_hidden_dim=64,
        )
    return LstmModelConfig(
        family=family,
        input_projection_dim=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        head_hidden_dim=64,
    )


def make_training_config() -> TrainingConfig:
    return TrainingConfig(
        learning_rate=3e-4,
        weight_decay=1e-2,
        batch_size=8,
        max_epochs=1,
        early_stopping={"patience": 8, "min_delta": 1e-4},
        gradient_clip_norm=1.0,
        action_loss_weight=1.0,
        fee_loss_weight=0.25,
        device="cpu",
        seed=2026,
        deterministic=True,
        log_every_n_steps=10,
        precision=TrainingPrecision.FP32,
        compile=CompileMode.OFF,
    )


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
        start_timestamp=TEST_WINDOW_START_TIMESTAMP - count * 12,
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
        start_timestamp=TEST_WINDOW_START_TIMESTAMP,
        include_gas_limit=True,
    )


def write_parquet_rows(path: Path, rows: list[dict[str, int | None]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    return path


def write_dataset_dir(dataset_dir: Path, rows: list[dict[str, int | None]]) -> Path:
    return write_parquet_rows(dataset_dir / "blocks.parquet", rows)
