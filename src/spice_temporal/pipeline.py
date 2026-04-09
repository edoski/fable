"""End-to-end orchestration for a single training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spice_temporal.config import ChainConfig, ModelConfig, SplitConfig, TrainingConfig
from spice_temporal.datasets import (
    average_interblock_seconds,
    build_supervised_examples,
    chronological_split,
    estimate_horizon_blocks,
)
from spice_temporal.features import build_feature_rows
from spice_temporal.io import load_block_records
from spice_temporal.models import build_model
from spice_temporal.normalization import StandardScaler, fit_standard_scaler, transform_examples
from spice_temporal.records import BlockRecord, SupervisedExample
from spice_temporal.training import EpochMetrics, TrainingResult, evaluate_model, train_model


@dataclass(slots=True)
class PreparedDataset:
    train_examples: list[SupervisedExample]
    validation_examples: list[SupervisedExample]
    test_examples: list[SupervisedExample]
    scaler: StandardScaler
    lookback_steps: int
    horizon_blocks: int
    n_features: int
    n_classes: int


@dataclass(slots=True)
class SingleRunResult:
    prepared: PreparedDataset
    training_result: TrainingResult
    test_metrics: EpochMetrics


def derive_lookback_steps(lookback_seconds: int, nominal_block_time_seconds: float) -> int:
    if nominal_block_time_seconds <= 0:
        raise ValueError("nominal_block_time_seconds must be positive")
    return max(1, round(lookback_seconds / nominal_block_time_seconds))


def prepare_dataset(
    blocks: list[BlockRecord],
    *,
    chain: ChainConfig,
    window_seconds: int,
    lookback_seconds: int,
    split_config: SplitConfig,
) -> PreparedDataset:
    feature_rows = build_feature_rows(blocks)
    lookback_steps = derive_lookback_steps(lookback_seconds, chain.nominal_block_time_seconds)
    horizon_blocks = estimate_horizon_blocks(
        window_seconds,
        average_interblock_seconds([block.timestamp for block in blocks]),
    )
    examples = build_supervised_examples(
        feature_rows,
        lookback_steps=lookback_steps,
        horizon_blocks=horizon_blocks,
    )
    split = chronological_split(examples, split_config)
    scaler = fit_standard_scaler(split.train)
    train_examples = transform_examples(split.train, scaler)
    validation_examples = transform_examples(split.validation, scaler)
    test_examples = transform_examples(split.test, scaler)
    return PreparedDataset(
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        scaler=scaler,
        lookback_steps=lookback_steps,
        horizon_blocks=horizon_blocks,
        n_features=len(train_examples[0].inputs[0]),
        n_classes=len(train_examples[0].future_log_fees),
    )


def run_single_training(
    block_file: Path,
    *,
    chain: ChainConfig,
    window_seconds: int,
    lookback_seconds: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    split_config: SplitConfig,
) -> SingleRunResult:
    blocks = load_block_records(block_file)
    prepared = prepare_dataset(
        blocks,
        chain=chain,
        window_seconds=window_seconds,
        lookback_seconds=lookback_seconds,
        split_config=split_config,
    )
    model = build_model(prepared.n_features, prepared.n_classes, model_config)
    training_result = train_model(
        model,
        train_examples=prepared.train_examples,
        validation_examples=prepared.validation_examples,
        config=training_config,
    )
    test_metrics = evaluate_model(
        model,
        examples=prepared.test_examples,
        training_config=training_config,
    )
    return SingleRunResult(
        prepared=prepared,
        training_result=training_result,
        test_metrics=test_metrics,
    )
