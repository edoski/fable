"""Benchmark evaluation config and one-engine runtime helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.validation import validate_path_segment
from ..core.errors import SpiceOperatorError
from ..prediction.base import MetricDescriptor, MetricSet, WindowMetricSummary
from ..prediction.contracts import DecodedOffsets
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract


class EvaluationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvaluationSampler(StrEnum):
    FULLSET = "fullset"
    UNIFORM_WINDOW = "uniform_window"
    POISSON_ARRIVALS = "poisson_arrivals"


class EvaluatorConfig(EvaluationConfigModel):
    id: str
    sampler: EvaluationSampler
    window_seconds: int | None = Field(default=None, gt=0)
    repetitions: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None, ge=0)
    arrival_rate_per_second: float | None = Field(default=None, gt=0.0)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="evaluation.id")

    @model_validator(mode="after")
    def validate_sampler_fields(self) -> Self:
        if self.sampler is EvaluationSampler.FULLSET:
            _require_absent(
                self.window_seconds,
                self.repetitions,
                self.seed,
                self.arrival_rate_per_second,
                labels=(
                    "evaluation.window_seconds",
                    "evaluation.repetitions",
                    "evaluation.seed",
                    "evaluation.arrival_rate_per_second",
                ),
            )
            return self
        if self.sampler is EvaluationSampler.UNIFORM_WINDOW:
            _require_present(
                self.window_seconds,
                self.repetitions,
                self.seed,
                labels=(
                    "evaluation.window_seconds",
                    "evaluation.repetitions",
                    "evaluation.seed",
                ),
            )
            _require_absent(
                self.arrival_rate_per_second,
                labels=("evaluation.arrival_rate_per_second",),
            )
            return self
        _require_present(
            self.window_seconds,
            self.repetitions,
            self.seed,
            self.arrival_rate_per_second,
            labels=(
                "evaluation.window_seconds",
                "evaluation.repetitions",
                "evaluation.seed",
                "evaluation.arrival_rate_per_second",
            ),
        )
        return self


def _require_present(*values: object, labels: tuple[str, ...]) -> None:
    missing = [label for label, value in zip(labels, values, strict=True) if value is None]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))


def _require_absent(*values: object, labels: tuple[str, ...]) -> None:
    unexpected = [label for label, value in zip(labels, values, strict=True) if value is not None]
    if unexpected:
        raise ValueError("Unexpected fields for evaluation sampler: " + ", ".join(unexpected))


EvaluationMetadataValue = str | int | float
IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, EvaluationMetadataValue]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    metrics: MetricSet
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: list[EvaluationRun]


RunEvaluatorFn = Callable[
    [
        CompiledProblemStore,
        CompiledRealizationPolicyContract,
        DecodedOffsets,
        IntVector,
    ],
    EvaluationSummary,
]


@dataclass(frozen=True, slots=True)
class CompiledEvaluatorContract:
    evaluation_id: str
    metric_descriptors: tuple[MetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    config_payload: dict[str, object]
    run_fn: RunEvaluatorFn

    def run(
        self,
        store: CompiledProblemStore,
        realization_policy: CompiledRealizationPolicyContract,
        decoded_offsets: DecodedOffsets,
        sample_indices: IntVector,
    ) -> EvaluationSummary:
        return self.run_fn(
            store,
            realization_policy,
            decoded_offsets,
            sample_indices,
        )


EVALUATION_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        id="profit_over_baseline",
        label="profit over baseline",
        role="primary",
    ),
    MetricDescriptor(
        id="cost_over_optimum",
        label="cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="baseline_cost_over_optimum",
        label="baseline cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="realized_fee_sum",
        label="realized fee sum",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="baseline_fee_sum",
        label="baseline fee sum",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="optimum_fee_sum",
        label="optimum fee sum",
        role="diagnostic",
    ),
)


@dataclass(frozen=True, slots=True)
class ChronologicalSampleView:
    sample_positions: IntVector
    sample_timestamps: IntVector


def sample_poisson_arrivals(
    rng: np.random.Generator,
    *,
    rate_per_second: float,
    start_timestamp: float,
    end_timestamp: float,
) -> NDArray[np.float64]:
    if rate_per_second <= 0:
        raise ValueError("rate_per_second must be positive")
    arrivals: list[float] = []
    cursor = start_timestamp
    while cursor < end_timestamp:
        cursor += rng.exponential(1.0 / rate_per_second)
        if cursor < end_timestamp:
            arrivals.append(cursor)
    return np.asarray(arrivals, dtype=np.float64)


def chronological_sample_view(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> ChronologicalSampleView:
    resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
    sample_timestamps = store.timestamps[store.anchor_rows[resolved_sample_indices]].astype(
        np.int64,
        copy=False,
    )
    order = np.argsort(sample_timestamps, kind="stable").astype(np.int64, copy=False)
    return ChronologicalSampleView(
        sample_positions=order,
        sample_timestamps=sample_timestamps[order],
    )


def select_sample_positions_for_arrivals(
    sample_timestamps: NDArray[np.int64],
    arrivals: NDArray[np.float64],
) -> NDArray[np.int64]:
    if arrivals.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_positions = np.searchsorted(sample_timestamps, arrivals, side="right") - 1
    return selected_positions[selected_positions >= 0].astype(np.int64, copy=False)


def summarize_selected_costs(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_offsets: DecodedOffsets,
    sample_indices: IntVector,
    selected_positions: IntVector,
    *,
    metadata: dict[str, str | int | float],
) -> EvaluationRun:
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")

    realized = realization_policy.realize_selections(
        store,
        decoded_offsets,
        sample_indices,
        selected_positions,
    )
    realized_logs = store.log_base_fees[realized.realized_rows]
    realized_total = float(np.exp(realized_logs.astype(np.float64, copy=False)).sum())
    baseline_total = float(
        np.exp(store.log_base_fees[realized.baseline_rows].astype(np.float64, copy=False)).sum()
    )
    optimum_logs = store.log_base_fees[realized.optimum_rows].astype(np.float64, copy=False)
    optimum_total = float(np.exp(optimum_logs).sum())
    if baseline_total <= 0.0:
        raise ValueError("baseline fee total must be positive")
    if optimum_total <= 0.0:
        raise ValueError("optimum fee total must be positive")

    return EvaluationRun(
        n_events=int(selected_positions.shape[0]),
        metrics={
            "profit_over_baseline": (baseline_total - realized_total) / baseline_total,
            "cost_over_optimum": (realized_total - optimum_total) / optimum_total,
            "baseline_cost_over_optimum": (baseline_total - optimum_total) / optimum_total,
            "realized_fee_sum": realized_total,
            "baseline_fee_sum": baseline_total,
            "optimum_fee_sum": optimum_total,
        },
        metadata={
            **dict(metadata),
            "overflow_count": int(realized.overflow_mask.sum()),
        },
    )


def summarize_runs(runs: list[EvaluationRun]) -> EvaluationSummary:
    if not runs:
        raise ValueError("evaluation produced no runs")

    realized_fee_sum = sum(run.metrics["realized_fee_sum"] for run in runs)
    baseline_fee_sum = sum(run.metrics["baseline_fee_sum"] for run in runs)
    optimum_fee_sum = sum(run.metrics["optimum_fee_sum"] for run in runs)
    if baseline_fee_sum <= 0.0:
        raise ValueError("baseline fee sum must be positive")
    if optimum_fee_sum <= 0.0:
        raise ValueError("optimum fee sum must be positive")
    return EvaluationSummary(
        metrics=MetricSet(
            values={
                "profit_over_baseline": (baseline_fee_sum - realized_fee_sum)
                / baseline_fee_sum,
                "cost_over_optimum": (realized_fee_sum - optimum_fee_sum) / optimum_fee_sum,
                "baseline_cost_over_optimum": (baseline_fee_sum - optimum_fee_sum)
                / optimum_fee_sum,
                "realized_fee_sum": realized_fee_sum,
                "baseline_fee_sum": baseline_fee_sum,
                "optimum_fee_sum": optimum_fee_sum,
            }
        ),
        window_metrics={
            "profit_over_baseline": _summarize_window_metric(
                [run.metrics["profit_over_baseline"] for run in runs]
            ),
            "cost_over_optimum": _summarize_window_metric(
                [run.metrics["cost_over_optimum"] for run in runs]
            ),
            "baseline_cost_over_optimum": _summarize_window_metric(
                [run.metrics["baseline_cost_over_optimum"] for run in runs]
            ),
        }
        if len(runs) > 1
        else {},
        total_events=sum(run.n_events for run in runs),
        runs=runs,
    )


def _summarize_window_metric(values: list[float]) -> WindowMetricSummary:
    return WindowMetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )


def _run_fullset(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_offsets: DecodedOffsets,
    sample_indices: IntVector,
) -> EvaluationSummary:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    run = summarize_selected_costs(
        store,
        realization_policy,
        decoded_offsets,
        sample_indices,
        np.arange(sample_indices.shape[0], dtype=np.int64),
        metadata={"mode": "fullset"},
    )
    return summarize_runs([run])


def _run_uniform_window(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_offsets: DecodedOffsets,
    sample_indices: IntVector,
    *,
    config: EvaluatorConfig,
) -> EvaluationSummary:
    assert config.window_seconds is not None
    assert config.repetitions is not None
    assert config.seed is not None
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    chronological_samples = chronological_sample_view(store, sample_indices)
    first_timestamp = int(chronological_samples.sample_timestamps[0])
    last_timestamp = int(chronological_samples.sample_timestamps[-1])
    rng = np.random.default_rng(config.seed)
    runs = []
    if last_timestamp - first_timestamp <= config.window_seconds:
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                np.arange(sample_indices.shape[0], dtype=np.int64),
                metadata={"mode": "fullset_fallback"},
            )
        )
        return summarize_runs(runs)
    max_start = last_timestamp - config.window_seconds
    start_intervals = _non_empty_start_intervals(
        chronological_samples.sample_timestamps,
        first_timestamp=first_timestamp,
        max_start=max_start,
        window_seconds=config.window_seconds,
    )
    if not start_intervals:
        raise ValueError("uniform_window evaluation produced no non-empty windows")
    interval_sizes = np.array(
        [end - start + 1 for start, end in start_intervals],
        dtype=np.int64,
    )
    cumulative_sizes = np.cumsum(interval_sizes, dtype=np.int64)
    total_starts = int(cumulative_sizes[-1])
    for repetition in range(1, config.repetitions + 1):
        start_timestamp = _sample_start_timestamp(
            rng,
            start_intervals=start_intervals,
            cumulative_sizes=cumulative_sizes,
            total_starts=total_starts,
        )
        end_timestamp = start_timestamp + config.window_seconds
        selected_positions = chronological_samples.sample_positions[
            np.flatnonzero(
                (chronological_samples.sample_timestamps >= start_timestamp)
                & (chronological_samples.sample_timestamps < end_timestamp)
            )
        ].astype(np.int64, copy=False)
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                selected_positions,
                metadata={
                    "mode": "windowed",
                    "window_start_timestamp": start_timestamp,
                    "window_end_timestamp": end_timestamp,
                    "repetition": repetition,
                },
            )
        )
    return summarize_runs(runs)


def _non_empty_start_intervals(
    sample_timestamps: np.ndarray,
    *,
    first_timestamp: int,
    max_start: int,
    window_seconds: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for timestamp in sample_timestamps:
        start = max(first_timestamp, int(timestamp) - window_seconds + 1)
        end = min(max_start, int(timestamp))
        if start > end:
            continue
        if intervals and start <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def _sample_start_timestamp(
    rng: np.random.Generator,
    *,
    start_intervals: list[tuple[int, int]],
    cumulative_sizes: np.ndarray,
    total_starts: int,
) -> int:
    draw = int(rng.integers(total_starts))
    interval_index = int(np.searchsorted(cumulative_sizes, draw, side="right"))
    interval_start, _ = start_intervals[interval_index]
    interval_offset = draw - int(cumulative_sizes[interval_index - 1]) if interval_index else draw
    return interval_start + interval_offset


def _run_poisson_arrivals(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_offsets: DecodedOffsets,
    sample_indices: IntVector,
    *,
    config: EvaluatorConfig,
) -> EvaluationSummary:
    assert config.window_seconds is not None
    assert config.repetitions is not None
    assert config.seed is not None
    assert config.arrival_rate_per_second is not None
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    chronological_samples = chronological_sample_view(store, sample_indices)
    first_timestamp = int(chronological_samples.sample_timestamps[0])
    last_timestamp = int(chronological_samples.sample_timestamps[-1])
    latest_start = last_timestamp - config.window_seconds
    if latest_start < first_timestamp:
        raise ValueError("Evaluation examples do not cover the requested replay window")

    rng = np.random.default_rng(config.seed)
    runs = []
    for _ in range(config.repetitions):
        window_start = float(rng.uniform(first_timestamp, latest_start))
        window_end = window_start + config.window_seconds
        arrivals = sample_poisson_arrivals(
            rng,
            rate_per_second=config.arrival_rate_per_second,
            start_timestamp=window_start,
            end_timestamp=window_end,
        )
        selected_positions = chronological_samples.sample_positions[
            select_sample_positions_for_arrivals(
                chronological_samples.sample_timestamps,
                arrivals,
            )
        ].astype(np.int64, copy=False)
        if selected_positions.size == 0:
            continue
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                selected_positions,
                metadata={
                    "window_start_timestamp": window_start,
                    "window_end_timestamp": window_end,
                    "n_arrivals": int(arrivals.shape[0]),
                },
            )
        )

    if not runs:
        raise SpiceOperatorError(
            "poisson_arrivals evaluation produced no valid arrivals; adjust the benchmark rate or window"
        )
    return summarize_runs(runs)


def coerce_evaluator_config(
    raw_config: dict[str, object] | EvaluatorConfig,
) -> EvaluatorConfig:
    if isinstance(raw_config, EvaluatorConfig):
        return raw_config
    return EvaluatorConfig.model_validate(raw_config)


def compile_evaluator_contract(
    evaluator_config: EvaluatorConfig,
) -> CompiledEvaluatorContract:
    if evaluator_config.sampler is EvaluationSampler.FULLSET:
        run_fn = _run_fullset
    elif evaluator_config.sampler is EvaluationSampler.UNIFORM_WINDOW:
        run_fn = lambda store, realization_policy, decoded_offsets, sample_indices: _run_uniform_window(
            store,
            realization_policy,
            decoded_offsets,
            sample_indices,
            config=evaluator_config,
        )
    else:
        run_fn = lambda store, realization_policy, decoded_offsets, sample_indices: _run_poisson_arrivals(
            store,
            realization_policy,
            decoded_offsets,
            sample_indices,
            config=evaluator_config,
        )
    return CompiledEvaluatorContract(
        evaluation_id=evaluator_config.id,
        metric_descriptors=EVALUATION_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config_payload=evaluator_config.model_dump(mode="json", exclude_none=True),
        run_fn=run_fn,
    )
