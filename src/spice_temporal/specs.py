"""High-level workflow specifications shared by the API layer."""

from __future__ import annotations

from dataclasses import dataclass

from spice_temporal.config import (
    ChainConfig,
    ModelConfig,
    SimulationConfig,
    SplitConfig,
    TrainingConfig,
)


@dataclass(slots=True)
class TrainingSpec:
    chain: ChainConfig
    model: ModelConfig
    max_delay_seconds: int
    lookback_seconds: int
    target_anchor_count: int
    split: SplitConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        if self.max_delay_seconds <= 0:
            raise ValueError("max_delay_seconds must be positive")
        if self.lookback_seconds <= 0:
            raise ValueError("lookback_seconds must be positive")
        if self.target_anchor_count <= 0:
            raise ValueError("target_anchor_count must be positive")


@dataclass(slots=True)
class SimulationSpec:
    window_seconds: int
    arrival_rate_per_second: float
    repetitions: int
    seed: int

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.arrival_rate_per_second <= 0:
            raise ValueError("arrival_rate_per_second must be positive")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @classmethod
    def from_config(cls, config: SimulationConfig) -> SimulationSpec:
        return cls(
            window_seconds=config.window_seconds,
            arrival_rate_per_second=config.arrival_rate_per_second,
            repetitions=config.repetitions,
            seed=config.seed,
        )
