"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import yaml

ROOT_CONFIG_KEYS = {
    "output_root",
    "max_delay_seconds",
    "lookback_seconds",
    "target_anchor_count",
    "pull",
    "split",
    "training",
    "simulation",
    "chains",
}

SectionT = TypeVar("SectionT")


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


@dataclass(slots=True)
class ChainConfig:
    name: ChainName
    chain_id: int
    block_time_seconds: float
    history_days: int

    def __post_init__(self) -> None:
        self.name = ChainName(self.name)
        if self.chain_id <= 0:
            raise ValueError("chain_id must be positive")
        if self.block_time_seconds <= 0:
            raise ValueError("block_time_seconds must be positive")
        if self.history_days <= 0:
            raise ValueError("history_days must be positive")


@dataclass(slots=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be greater than 0 and less than 1")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be non-negative and less than 1")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be less than 1")


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

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(slots=True)
class PullConfig:
    requests_per_second: int
    max_concurrent_requests: int
    max_concurrent_chunks: int

    def __post_init__(self) -> None:
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if self.max_concurrent_chunks <= 0:
            raise ValueError("max_concurrent_chunks must be positive")


@dataclass(slots=True)
class SimulationConfig:
    window_seconds: int = 7_200
    arrival_rate_per_second: float = 0.05
    repetitions: int = 50
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.arrival_rate_per_second <= 0:
            raise ValueError("arrival_rate_per_second must be positive")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


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

    def __post_init__(self) -> None:
        self.family = ModelFamily(self.family)
        if self.input_projection_dim <= 0:
            raise ValueError("input_projection_dim must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.nhead <= 0:
            raise ValueError("nhead must be positive")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if self.head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive")


@dataclass(slots=True)
class ExperimentConfig:
    output_root: Path
    max_delay_seconds: list[int] = field(default_factory=lambda: [12, 24, 36])
    lookback_seconds: int = 600
    target_anchor_count: int = 400_000
    pull: PullConfig = field(
        default_factory=lambda: PullConfig(
            requests_per_second=10,
            max_concurrent_requests=2,
            max_concurrent_chunks=1,
        )
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    chains: list[ChainConfig] = field(
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

    def __post_init__(self) -> None:
        if not self.max_delay_seconds:
            raise ValueError("max_delay_seconds must be non-empty")
        if any(value <= 0 for value in self.max_delay_seconds):
            raise ValueError("max_delay_seconds must contain only positive values")
        if self.lookback_seconds <= 0:
            raise ValueError("lookback_seconds must be positive")
        if self.target_anchor_count <= 0:
            raise ValueError("target_anchor_count must be positive")
        if not self.chains:
            raise ValueError("chains must be non-empty")

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ValueError(f"Config at {path} must contain a top-level mapping")

        unknown = sorted(set(raw) - ROOT_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Config at {path} contains unknown top-level keys: {unknown}")

        for field_name in ("output_root", "max_delay_seconds", "pull", "simulation", "chains"):
            if field_name not in raw:
                raise ValueError(f"Config at {path} must define {field_name}")

        pull = _load_section(PullConfig, raw["pull"], path=path, section="pull")
        split = _load_section(SplitConfig, raw.get("split", {}), path=path, section="split")
        training = _load_section(
            TrainingConfig,
            raw.get("training", {}),
            path=path,
            section="training",
        )
        simulation = _load_section(
            SimulationConfig,
            raw["simulation"],
            path=path,
            section="simulation",
        )
        chains_raw = raw["chains"]
        if not isinstance(chains_raw, list) or not chains_raw:
            raise ValueError(f"Config at {path} must define a non-empty chains list")
        chains = [
            _load_section(ChainConfig, item, path=path, section=f"chains[{index}]")
            for index, item in enumerate(chains_raw)
        ]
        return cls(
            output_root=Path(raw["output_root"]),
            max_delay_seconds=list(raw["max_delay_seconds"]),
            lookback_seconds=raw.get("lookback_seconds", 600),
            target_anchor_count=raw.get("target_anchor_count", 400_000),
            pull=pull,
            split=split,
            training=training,
            simulation=simulation,
            chains=chains,
        )


def _load_section(
    section_type: type[SectionT],
    raw: Any,
    *,
    path: Path,
    section: str,
) -> SectionT:
    if not isinstance(raw, dict):
        raise ValueError(f"Config section {section} in {path} must be a mapping")
    try:
        return section_type(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid config section {section} in {path}: {exc}") from exc
