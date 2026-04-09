"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ModelFamily = Literal["lstm", "transformer", "transformer_lstm"]
ChainName = Literal["ethereum", "polygon", "avalanche"]


@dataclass(slots=True)
class ChainConfig:
    name: ChainName
    chain_id: int
    nominal_block_time_seconds: float
    history_days_hint: int


@dataclass(slots=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    effective_batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    alpha: float = 1.0
    beta: float = 0.25
    device: str = "auto"
    seed: int = 2026


@dataclass(slots=True)
class PullConfig:
    requests_per_second: int
    max_concurrent_requests: int
    max_concurrent_chunks: int


@dataclass(slots=True)
class ModelConfig:
    family: ModelFamily
    input_projection_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 512
    head_hidden_dim: int = 64


@dataclass(slots=True)
class ExperimentConfig:
    output_root: Path
    window_seconds: list[int] = field(default_factory=lambda: [12, 24, 36])
    lookback_seconds: int = 600
    target_anchor_count: int = 400_000
    anchor_buffer: int = 20_000
    pull: PullConfig = field(
        default_factory=lambda: PullConfig(
            requests_per_second=10,
            max_concurrent_requests=2,
            max_concurrent_chunks=1,
        )
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    chains: list[ChainConfig] = field(
        default_factory=lambda: [
            ChainConfig(
                name="ethereum",
                chain_id=1,
                nominal_block_time_seconds=12.0,
                history_days_hint=70,
            ),
            ChainConfig(
                name="polygon",
                chain_id=137,
                nominal_block_time_seconds=2.0,
                history_days_hint=20,
            ),
            ChainConfig(
                name="avalanche",
                chain_id=43114,
                nominal_block_time_seconds=1.0,
                history_days_hint=20,
            ),
        ]
    )

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if "pull" not in raw:
            raise ValueError(f"Config at {path} must define a pull section")

        pull = PullConfig(**raw["pull"])
        split = SplitConfig(**raw.get("split", {}))
        training = TrainingConfig(**raw.get("training", {}))
        default_chains = cls(output_root=Path(".")).chains
        chains = [ChainConfig(**item) for item in raw.get("chains", [])] or default_chains
        return cls(
            output_root=Path(raw["output_root"]),
            window_seconds=list(raw.get("window_seconds", [12, 24, 36])),
            lookback_seconds=raw.get("lookback_seconds", 600),
            target_anchor_count=raw.get("target_anchor_count", 400_000),
            anchor_buffer=raw.get("anchor_buffer", 20_000),
            pull=pull,
            split=split,
            training=training,
            chains=chains,
        )
