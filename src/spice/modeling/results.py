# pyright: strict

"""Typed training and evaluation summary envelopes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..config import (
    ArtifactVariant,
    DatasetBuilderConfig,
    FeatureSetConfig,
    ModelConfig,
    PredictionConfig,
    ProblemSpec,
    StudyConfig,
)
from ..evaluation import EvaluationRun, EvaluationSummary
from ..modeling.dataset_builders import BuilderRuntimeMetadata
from ..prediction import MetricDescriptor, MetricSet, WindowMetricSummary
from ..semantics import ArtifactSemantics
from ..temporal.scaling import ScalerStats
from .pipeline import PreparedInferenceDataset, PreparedTrainingDataset, TrainingRunResult


@dataclass(frozen=True, slots=True)
class TrainingArtifactManifest:
    """Single-source persisted artifact provenance plus exact authored config payloads."""

    artifact_id: str
    dataset_builder: DatasetBuilderConfig
    prediction: PredictionConfig
    chain_name: str
    dataset_id: str
    dataset_name: str
    problem: ProblemSpec
    variant: ArtifactVariant
    study: StudyConfig | None
    study_id: str | None
    feature_set: FeatureSetConfig
    model: ModelConfig[str]
    scaler: ScalerStats
    builder_runtime_metadata: BuilderRuntimeMetadata
    semantics: ArtifactSemantics

    @property
    def problem_id(self) -> str:
        return self.semantics.problem.problem_id

    @property
    def prediction_id(self) -> str:
        return self.semantics.prediction.prediction_id

    @property
    def prediction_family_id(self) -> str:
        return self.semantics.prediction.prediction_family_id

    @property
    def feature_set_id(self) -> str:
        return self.semantics.feature.feature_set_id

    @property
    def feature_family_id(self) -> str:
        return self.semantics.feature.feature_family_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.semantics.feature.feature_names

    @property
    def max_delay_seconds(self) -> int:
        return self.semantics.problem.max_delay_seconds

    @property
    def lookback_seconds(self) -> int:
        return self.semantics.problem.lookback_seconds

    @property
    def sample_count(self) -> int:
        return self.semantics.problem.sample_count

    @property
    def feature_prerequisites(self):
        return self.semantics.feature.feature_prerequisites

    @property
    def max_candidate_slots(self) -> int:
        return self.semantics.max_candidate_slots

    @property
    def feature_graph_fingerprint(self) -> str:
        return self.semantics.feature.feature_graph_fingerprint

    @property
    def training_metric_descriptors(self):
        return self.semantics.prediction.training_metric_descriptors

    @property
    def representation_id(self) -> str:
        return self.semantics.representation.representation_id

    @property
    def dataset_builder_id(self) -> str:
        return self.semantics.dataset_builder.dataset_builder_id

    @property
    def input_normalization_id(self) -> str:
        return self.semantics.input_normalization.input_normalization_id

    @property
    def n_features(self) -> int:
        return len(self.semantics.feature.feature_names)


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train_samples: int
    validation_samples: int
    test_samples: int


@dataclass(frozen=True, slots=True)
class TrainingRuntimeSummary:
    """Runtime-only training outcomes stored separately from manifest provenance."""

    n_rows_available: int
    n_rows_used: int
    split_sizes: SplitSizes
    best_epoch: int
    resolved_device: str
    resolved_precision: str
    compiled: bool
    loader_strategy_id: str
    input_storage_mode_id: str
    target_storage_mode_id: str
    batch_planner_id: str
    best_validation_metrics: MetricSet
    test_metrics: MetricSet


@dataclass(frozen=True, slots=True)
class LoadedTrainingSummary:
    """Read model that pairs artifact provenance with one training runtime summary."""

    manifest: TrainingArtifactManifest
    runtime: TrainingRuntimeSummary


@dataclass(frozen=True, slots=True)
class TrainingEpochRecord:
    epoch: int
    train_metrics: MetricSet
    validation_metrics: MetricSet


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeSummary:
    """Runtime-only evaluation outcomes stored separately from manifest provenance."""

    delay_seconds: int
    evaluator_id: str
    evaluator_config: dict[str, object]
    metric_descriptors: tuple[MetricDescriptor, ...]
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    metrics: MetricSet
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: list[EvaluationRun]


@dataclass(frozen=True, slots=True)
class LoadedEvaluationSummary:
    """Read model that pairs artifact provenance with one evaluation runtime summary."""

    evaluation_id: str
    recorded_at: int
    manifest: TrainingArtifactManifest
    runtime: EvaluationRuntimeSummary


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


def build_training_runtime_summary(
    result: TrainingRunResult,
    *,
    prepared: PreparedTrainingDataset,
    best_validation_metrics: MetricSet,
    test_metrics: MetricSet,
) -> TrainingRuntimeSummary:
    return TrainingRuntimeSummary(
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
        loader_strategy_id=result.training_result.loader_strategy_id,
        input_storage_mode_id=result.training_result.input_storage_mode_id,
        target_storage_mode_id=result.training_result.target_storage_mode_id,
        batch_planner_id=result.training_result.batch_planner_id,
        best_validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
    )


def build_evaluation_runtime_summary(
    *,
    prepared: PreparedInferenceDataset,
    evaluation: EvaluationSummary,
    delay_seconds: int,
    evaluator_id: str,
    evaluator_config: dict[str, object],
    metric_descriptors: tuple[MetricDescriptor, ...],
) -> EvaluationRuntimeSummary:
    return EvaluationRuntimeSummary(
        delay_seconds=delay_seconds,
        evaluator_id=evaluator_id,
        evaluator_config=dict(evaluator_config),
        metric_descriptors=metric_descriptors,
        n_history_rows=prepared.n_history_rows,
        n_evaluation_rows=prepared.n_evaluation_rows,
        sample_count=prepared.sample_count,
        metrics=evaluation.metrics,
        window_metrics=dict(evaluation.window_metrics),
        total_events=evaluation.total_events,
        runs=list(evaluation.runs),
    )
