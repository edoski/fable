"""Internal training and simulation summaries."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..config import ArtifactVariant, StudyConfig
from .artifacts import LoadedTrainingArtifact, TrainingArtifactManifest
from .evaluation import EpochMetrics
from .pipeline import PreparedInferenceDataset, PreparedTrainingDataset, TrainingRunResult
from .simulation import SimulationSummary


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train_samples: int
    validation_samples: int
    test_samples: int


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    total_loss: float
    accuracy: float
    mean_cost_over_optimum: float
    mean_profit_over_baseline: float


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    chain: str
    dataset_id: str
    variant: ArtifactVariant
    study: StudyConfig | None
    model_id: str
    task_id: str
    max_supported_delay_seconds: int
    lookback_seconds: int
    sample_count: int
    n_blocks_available: int
    n_blocks_used: int
    split_sizes: SplitSizes
    best_epoch: int
    resolved_device: str
    resolved_precision: str
    compiled: bool
    best_validation_metrics: MetricsSummary
    test_metrics: MetricsSummary


@dataclass(frozen=True, slots=True)
class SimulationAggregateSummary:
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class SimulationRunRecord:
    window_start_timestamp: float
    window_end_timestamp: float
    n_arrivals: int
    n_events: int
    profit_over_baseline: float
    cost_over_optimum: float
    baseline_cost_over_optimum: float


@dataclass(frozen=True, slots=True)
class SimulationSummaryRecord:
    chain: str
    dataset_id: str
    variant: ArtifactVariant
    study: StudyConfig | None
    model_id: str
    task_id: str
    max_supported_delay_seconds: int
    requested_delay_seconds: int
    lookback_seconds: int
    simulation_window_seconds: int
    arrival_rate_per_second: float
    repetitions: int
    n_history_context_blocks: int
    n_evaluation_blocks: int
    sample_count: int
    profit_over_baseline: SimulationAggregateSummary
    cost_over_optimum: SimulationAggregateSummary
    baseline_cost_over_optimum: SimulationAggregateSummary
    total_events: int
    runs: list[SimulationRunRecord]


def summarize_epoch_metrics(metrics: EpochMetrics) -> MetricsSummary:
    return MetricsSummary(
        total_loss=metrics.total_loss,
        accuracy=metrics.accuracy,
        mean_cost_over_optimum=metrics.mean_cost_over_optimum,
        mean_profit_over_baseline=metrics.mean_profit_over_baseline,
    )


def iter_epoch_pairs(result: TrainingRunResult) -> Iterator[tuple[int, EpochMetrics, EpochMetrics]]:
    for index, (train_metrics, validation_metrics) in enumerate(
        zip(
            result.training_result.train_history,
            result.training_result.validation_history,
            strict=True,
        ),
        start=1,
    ):
        yield index, train_metrics, validation_metrics


def build_training_summary(
    result: TrainingRunResult,
    *,
    chain_name: str,
    dataset_id: str,
    model_id: str,
    manifest: TrainingArtifactManifest,
    prepared: PreparedTrainingDataset,
) -> TrainingSummary:
    best_validation = result.training_result.validation_history[
        result.training_result.best_epoch - 1
    ]
    return TrainingSummary(
        chain=chain_name,
        dataset_id=dataset_id,
        variant=manifest.variant,
        study=manifest.study,
        model_id=model_id,
        task_id=manifest.task_id,
        max_supported_delay_seconds=manifest.max_supported_delay_seconds,
        lookback_seconds=manifest.lookback_seconds,
        sample_count=manifest.sample_count,
        n_blocks_available=prepared.n_blocks_available,
        n_blocks_used=prepared.n_blocks_used,
        split_sizes=SplitSizes(
            train_samples=int(prepared.split_indices.train.shape[0]),
            validation_samples=int(prepared.split_indices.validation.shape[0]),
            test_samples=int(prepared.split_indices.test.shape[0]),
        ),
        best_epoch=result.training_result.best_epoch,
        resolved_device=result.training_result.resolved_device,
        resolved_precision=result.training_result.resolved_precision,
        compiled=result.training_result.compiled,
        best_validation_metrics=summarize_epoch_metrics(best_validation),
        test_metrics=summarize_epoch_metrics(result.test_metrics),
    )


def build_simulation_summary_record(
    loaded_artifact: LoadedTrainingArtifact,
    *,
    prepared: PreparedInferenceDataset,
    simulation: SimulationSummary,
    requested_delay_seconds: int,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
) -> SimulationSummaryRecord:
    manifest = loaded_artifact.manifest
    return SimulationSummaryRecord(
        chain=manifest.chain.name,
        dataset_id=manifest.dataset_id,
        variant=manifest.variant,
        study=manifest.study,
        model_id=manifest.model.id,
        task_id=manifest.task_id,
        max_supported_delay_seconds=manifest.max_supported_delay_seconds,
        requested_delay_seconds=requested_delay_seconds,
        lookback_seconds=manifest.lookback_seconds,
        simulation_window_seconds=window_seconds,
        arrival_rate_per_second=arrival_rate_per_second,
        repetitions=repetitions,
        n_history_context_blocks=prepared.n_history_context_blocks,
        n_evaluation_blocks=prepared.n_evaluation_blocks,
        sample_count=prepared.sample_count,
        profit_over_baseline=SimulationAggregateSummary(
            mean=simulation.mean_profit_over_baseline,
            std=simulation.std_profit_over_baseline,
        ),
        cost_over_optimum=SimulationAggregateSummary(
            mean=simulation.mean_cost_over_optimum,
            std=simulation.std_cost_over_optimum,
        ),
        baseline_cost_over_optimum=SimulationAggregateSummary(
            mean=simulation.mean_baseline_cost_over_optimum,
            std=simulation.std_baseline_cost_over_optimum,
        ),
        total_events=simulation.total_events,
        runs=[
            SimulationRunRecord(
                window_start_timestamp=run.window_start_timestamp,
                window_end_timestamp=run.window_end_timestamp,
                n_arrivals=run.n_arrivals,
                n_events=run.n_events,
                profit_over_baseline=run.profit_over_baseline,
                cost_over_optimum=run.cost_over_optimum,
                baseline_cost_over_optimum=run.baseline_cost_over_optimum,
            )
            for run in simulation.runs
        ],
    )
