"""Typed training and simulation summary envelopes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..config import (
    ArtifactVariant,
    ModelConfig,
    PredictionConfig,
    ProblemSpec,
    StudyConfig,
)
from ..features import FeaturePrerequisites
from ..prediction import (
    MetricDescriptor,
    MetricSet,
    PredictionSimulationRun,
    PredictionSimulationSummary,
    WindowMetricSummary,
)
from ..temporal.scaling import ScalerStats
from .pipeline import PreparedInferenceDataset, PreparedTrainingDataset, TrainingRunResult


@dataclass(frozen=True, slots=True)
class ArtifactChainMetadata:
    name: str


@dataclass(frozen=True, slots=True)
class TrainingArtifactManifest:
    artifact_id: str
    objective_id: str
    prediction: PredictionConfig
    metric_descriptors: list[MetricDescriptor]
    chain: ArtifactChainMetadata
    dataset_id: str
    dataset_name: str
    problem: ProblemSpec
    variant: ArtifactVariant
    study: StudyConfig | None
    study_id: str | None
    feature_family_id: str
    feature_prerequisites: FeaturePrerequisites
    max_candidate_slots: int
    feature_set_id: str
    feature_names: list[str]
    feature_graph_fingerprint: str
    model: ModelConfig
    scaler: ScalerStats
    compiler_runtime_metadata: dict[str, object]

    @property
    def problem_id(self) -> str:
        return self.problem.id

    @property
    def prediction_id(self) -> str:
        return self.prediction.id

    @property
    def prediction_family_id(self) -> str:
        return self.prediction.family.id

    @property
    def max_supported_delay_seconds(self) -> int:
        return self.problem.max_supported_delay_seconds

    @property
    def lookback_seconds(self) -> int:
        return self.problem.lookback_seconds

    @property
    def sample_count(self) -> int:
        return self.problem.sample_count

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train_samples: int
    validation_samples: int
    test_samples: int


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    artifact_id: str
    objective_id: str
    prediction_id: str
    prediction_family_id: str
    metric_descriptors: list[MetricDescriptor]
    chain: str
    dataset_id: str
    dataset_name: str
    variant: ArtifactVariant
    study: StudyConfig | None
    study_id: str | None
    model_id: str
    problem_id: str
    max_supported_delay_seconds: int
    lookback_seconds: int
    feature_family_id: str
    feature_prerequisites: FeaturePrerequisites
    sample_count: int
    max_candidate_slots: int
    n_rows_available: int
    n_rows_used: int
    split_sizes: SplitSizes
    best_epoch: int
    resolved_device: str
    resolved_precision: str
    compiled: bool
    representation_id: str
    storage_mode_id: str
    batch_planner_id: str
    family_execution_id: str
    best_validation_metrics: MetricSet
    test_metrics: MetricSet


@dataclass(frozen=True, slots=True)
class TrainingEpochRecord:
    epoch: int
    train_metrics: MetricSet
    validation_metrics: MetricSet


@dataclass(frozen=True, slots=True)
class SimulationSummaryRecord:
    artifact_id: str
    objective_id: str
    prediction_id: str
    prediction_family_id: str
    metric_descriptors: list[MetricDescriptor]
    chain: str
    dataset_id: str
    dataset_name: str
    variant: ArtifactVariant
    study: StudyConfig | None
    study_id: str | None
    model_id: str
    problem_id: str
    max_supported_delay_seconds: int
    requested_delay_seconds: int
    lookback_seconds: int
    feature_family_id: str
    feature_prerequisites: FeaturePrerequisites
    simulation_window_seconds: int
    arrival_rate_per_second: float
    repetitions: int
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    max_candidate_slots: int
    metrics: MetricSet
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: list[PredictionSimulationRun]


def iter_epoch_records(result: TrainingRunResult) -> Iterator[TrainingEpochRecord]:
    for index, (train_metrics, validation_metrics) in enumerate(
        zip(
            result.training_result.train_history,
            result.training_result.validation_history,
            strict=True,
        ),
        start=1,
    ):
        yield TrainingEpochRecord(
            epoch=index,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )


def build_training_summary(
    result: TrainingRunResult,
    *,
    chain_name: str,
    dataset_id: str,
    model_id: str,
    manifest: TrainingArtifactManifest,
    prepared: PreparedTrainingDataset,
    best_validation_metrics: MetricSet,
    test_metrics: MetricSet,
) -> TrainingSummary:
    return TrainingSummary(
        artifact_id=manifest.artifact_id,
        objective_id=manifest.objective_id,
        prediction_id=manifest.prediction_id,
        prediction_family_id=manifest.prediction_family_id,
        metric_descriptors=list(manifest.metric_descriptors),
        chain=chain_name,
        dataset_id=dataset_id,
        dataset_name=manifest.dataset_name,
        variant=manifest.variant,
        study=manifest.study,
        study_id=manifest.study_id,
        model_id=model_id,
        problem_id=manifest.problem_id,
        max_supported_delay_seconds=manifest.max_supported_delay_seconds,
        lookback_seconds=manifest.lookback_seconds,
        feature_family_id=manifest.feature_family_id,
        feature_prerequisites=manifest.feature_prerequisites,
        sample_count=manifest.sample_count,
        max_candidate_slots=manifest.max_candidate_slots,
        n_rows_available=prepared.n_rows_available,
        n_rows_used=prepared.n_rows_used,
        split_sizes=SplitSizes(
            train_samples=int(prepared.split_indices.train.shape[0]),
            validation_samples=int(prepared.split_indices.validation.shape[0]),
            test_samples=int(prepared.split_indices.test.shape[0]),
        ),
        best_epoch=result.training_result.best_epoch,
        resolved_device=result.training_result.resolved_device,
        resolved_precision=result.training_result.resolved_precision,
        compiled=result.training_result.compiled,
        representation_id=result.training_result.representation_id,
        storage_mode_id=result.training_result.storage_mode_id,
        batch_planner_id=result.training_result.batch_planner_id,
        family_execution_id=result.training_result.family_execution_id,
        best_validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
    )


def build_simulation_summary_record(
    manifest: TrainingArtifactManifest,
    *,
    prepared: PreparedInferenceDataset,
    simulation: PredictionSimulationSummary,
    requested_delay_seconds: int,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
) -> SimulationSummaryRecord:
    return SimulationSummaryRecord(
        artifact_id=manifest.artifact_id,
        objective_id=manifest.objective_id,
        prediction_id=manifest.prediction_id,
        prediction_family_id=manifest.prediction_family_id,
        metric_descriptors=list(manifest.metric_descriptors),
        chain=manifest.chain.name,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        variant=manifest.variant,
        study=manifest.study,
        study_id=manifest.study_id,
        model_id=manifest.model.id,
        problem_id=manifest.problem_id,
        max_supported_delay_seconds=manifest.max_supported_delay_seconds,
        requested_delay_seconds=requested_delay_seconds,
        lookback_seconds=manifest.lookback_seconds,
        feature_family_id=manifest.feature_family_id,
        feature_prerequisites=manifest.feature_prerequisites,
        simulation_window_seconds=window_seconds,
        arrival_rate_per_second=arrival_rate_per_second,
        repetitions=repetitions,
        n_history_rows=prepared.n_history_rows,
        n_evaluation_rows=prepared.n_evaluation_rows,
        sample_count=prepared.sample_count,
        max_candidate_slots=manifest.max_candidate_slots,
        metrics=simulation.metrics,
        window_metrics=dict(simulation.window_metrics),
        total_events=simulation.total_events,
        runs=list(simulation.runs),
    )
