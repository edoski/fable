"""Evaluation-private temporal replay result ABI."""

from __future__ import annotations

from dataclasses import dataclass

from ..prediction.base import MetricSet, WindowMetricSummary
from .contracts import EvaluationMetadataValue, EvaluationRun, EvaluationSummary

PROFIT_OVER_BASELINE = "profit_over_baseline"
COST_OVER_OPTIMUM = "cost_over_optimum"
BASELINE_COST_OVER_OPTIMUM = "baseline_cost_over_optimum"
EXACT_OPTIMUM_HIT_RATE = "exact_optimum_hit_rate"
REALIZED_FEE_SUM = "realized_fee_sum"
BASELINE_FEE_SUM = "baseline_fee_sum"
OPTIMUM_FEE_SUM = "optimum_fee_sum"
TEMPORAL_RATIO_METRIC_IDS = (
    PROFIT_OVER_BASELINE,
    COST_OVER_OPTIMUM,
    BASELINE_COST_OVER_OPTIMUM,
    EXACT_OPTIMUM_HIT_RATE,
)


@dataclass(frozen=True, slots=True)
class TemporalReplayMetrics:
    profit_over_baseline: float
    cost_over_optimum: float
    baseline_cost_over_optimum: float
    exact_optimum_hit_rate: float
    realized_fee_sum: float
    baseline_fee_sum: float
    optimum_fee_sum: float

    def values(self) -> dict[str, float]:
        return {
            PROFIT_OVER_BASELINE: self.profit_over_baseline,
            COST_OVER_OPTIMUM: self.cost_over_optimum,
            BASELINE_COST_OVER_OPTIMUM: self.baseline_cost_over_optimum,
            EXACT_OPTIMUM_HIT_RATE: self.exact_optimum_hit_rate,
            REALIZED_FEE_SUM: self.realized_fee_sum,
            BASELINE_FEE_SUM: self.baseline_fee_sum,
            OPTIMUM_FEE_SUM: self.optimum_fee_sum,
        }


@dataclass(frozen=True, slots=True)
class TemporalReplayEventMetricSums:
    profit_over_baseline: float
    cost_over_optimum: float
    baseline_cost_over_optimum: float
    exact_optimum_hit_rate: float

    def value(self, metric_id: str) -> float:
        if metric_id == PROFIT_OVER_BASELINE:
            return self.profit_over_baseline
        if metric_id == COST_OVER_OPTIMUM:
            return self.cost_over_optimum
        if metric_id == BASELINE_COST_OVER_OPTIMUM:
            return self.baseline_cost_over_optimum
        if metric_id == EXACT_OPTIMUM_HIT_RATE:
            return self.exact_optimum_hit_rate
        raise KeyError(metric_id)


@dataclass(frozen=True, slots=True)
class TemporalReplayRunResult:
    n_events: int
    metrics: TemporalReplayMetrics
    event_metric_sums: TemporalReplayEventMetricSums
    metadata: dict[str, EvaluationMetadataValue]


@dataclass(frozen=True, slots=True)
class TemporalReplayWindowMetric:
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class TemporalReplayResult:
    metrics: TemporalReplayMetrics
    window_metrics: dict[str, TemporalReplayWindowMetric]
    total_events: int
    runs: tuple[TemporalReplayRunResult, ...]


def temporal_replay_result_to_summary(result: TemporalReplayResult) -> EvaluationSummary:
    return EvaluationSummary(
        metrics=MetricSet(values=result.metrics.values()),
        window_metrics={
            metric_id: WindowMetricSummary(mean=metric.mean, std=metric.std)
            for metric_id, metric in result.window_metrics.items()
        },
        total_events=result.total_events,
        runs=[
            EvaluationRun(
                n_events=run.n_events,
                metrics=run.metrics.values(),
                metadata=dict(run.metadata),
            )
            for run in result.runs
        ],
    )
