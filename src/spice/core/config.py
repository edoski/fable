"""Strict experiment configuration models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictModel(BaseModel):
    """Shared strict base model for repo configuration and persisted schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


class ChainName(StrEnum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"


class ModelFamily(StrEnum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    TRANSFORMER_LSTM = "transformer_lstm"


class BlockSegment(StrEnum):
    HISTORY = "history"
    EVALUATION = "evaluation"


class ChainConfig(StrictModel):
    name: ChainName
    chain_id: PositiveInt
    block_time_seconds: PositiveFloat
    history_days: PositiveInt


class SplitConfig(StrictModel):
    train_fraction: float = 0.8
    validation_fraction: float = 0.1

    @model_validator(mode="after")
    def _validate_fractions(self) -> SplitConfig:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be greater than 0 and less than 1")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be non-negative and less than 1")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        return self


class TrainingConfig(StrictModel):
    learning_rate: PositiveFloat = 3e-4
    weight_decay: float = 1e-2
    effective_batch_size: PositiveInt = 64
    max_epochs: PositiveInt = 50
    early_stopping_patience: PositiveInt = 8
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: PositiveFloat = 1.0
    alpha: PositiveFloat = 1.0
    beta: PositiveFloat = 0.25
    device: str = "auto"
    seed: int = 2026

    @model_validator(mode="after")
    def _validate_non_negative(self) -> TrainingConfig:
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        return self


class PullConfig(StrictModel):
    requests_per_second: PositiveInt = 10
    max_concurrent_requests: PositiveInt = 2
    max_concurrent_chunks: PositiveInt = 1
    chunk_size: PositiveInt = 1000


class SimulationConfig(StrictModel):
    window_seconds: PositiveInt = 7_200
    arrival_rate_per_second: PositiveFloat = 0.05
    repetitions: PositiveInt = 50
    seed: int = 2026

    @model_validator(mode="after")
    def _validate_seed(self) -> SimulationConfig:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        return self


class ModelConfig(StrictModel):
    family: ModelFamily
    input_projection_dim: PositiveInt = 128
    hidden_size: PositiveInt = 128
    num_layers: PositiveInt = 2
    dropout: float = 0.1
    d_model: PositiveInt = 128
    nhead: PositiveInt = 4
    transformer_layers: PositiveInt = 2
    feedforward_dim: PositiveInt = 512
    head_hidden_dim: PositiveInt = 64

    @model_validator(mode="after")
    def _validate_dropout(self) -> ModelConfig:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")
        return self


class ExperimentConfig(StrictModel):
    output_root: Path
    max_delay_seconds: list[PositiveInt] = Field(default_factory=lambda: [12, 24, 36])
    lookback_seconds: PositiveInt = 600
    target_anchor_count: PositiveInt = 400_000
    pull: PullConfig = Field(default_factory=PullConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    chains: list[ChainConfig] = Field(
        default_factory=lambda: [
            ChainConfig(
                name=ChainName.ETHEREUM,
                chain_id=1,
                block_time_seconds=12.0,
                history_days=60,
            ),
            ChainConfig(
                name=ChainName.POLYGON,
                chain_id=137,
                block_time_seconds=2.0,
                history_days=10,
            ),
            ChainConfig(
                name=ChainName.AVALANCHE,
                chain_id=43114,
                block_time_seconds=1.6,
                history_days=10,
            ),
        ]
    )

    @model_validator(mode="after")
    def _validate_unique_chains(self) -> ExperimentConfig:
        if not self.max_delay_seconds:
            raise ValueError("max_delay_seconds must be non-empty")
        if not self.chains:
            raise ValueError("chains must be non-empty")
        names = [chain.name for chain in self.chains]
        if len(names) != len(set(names)):
            raise ValueError("chains must contain unique chain names")
        return self

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Config at {path} must contain a top-level mapping")
        return cls.model_validate(raw)

    def resolve_chain(self, chain_name: ChainName | str) -> ChainConfig:
        wanted = ChainName(chain_name)
        for chain in self.chains:
            if chain.name is wanted:
                return chain
        raise ValueError(f"Unknown chain: {wanted}")
