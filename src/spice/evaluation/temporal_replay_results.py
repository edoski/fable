"""Evaluation-private temporal replay result ABI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..metrics import WindowMetricSummary
from ._temporal_replay_metric_catalog import validate_temporal_replay_metric_values
from .contracts import EvaluationMetadataValue


@dataclass(frozen=True, slots=True)
class TemporalReplayRunResult:
    n_events: int
    metrics: Mapping[str, float]
    fee_sums: Mapping[str, float]
    event_metric_sums: Mapping[str, float]
    metadata: dict[str, EvaluationMetadataValue]

    def __post_init__(self) -> None:
        validate_temporal_replay_metric_values(self.metrics)


@dataclass(frozen=True, slots=True)
class TemporalReplayResult:
    metrics: Mapping[str, float]
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: tuple[TemporalReplayRunResult, ...]

    def __post_init__(self) -> None:
        validate_temporal_replay_metric_values(self.metrics)
