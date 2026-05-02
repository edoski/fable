# pyright: strict

"""Typed training and evaluation summary envelopes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from ..config.models import (
    ArtifactVariant,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
)
from ..evaluation import EvaluationRun, EvaluationSummary, EvaluatorConfig
from ..metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from ..modeling.dataset_builders import (
    BuilderRuntimeMetadata,
    DatasetBuilderConfig,
    PreparedInferenceDataset,
    PreparedTrainingDataset,
)
from ..modeling.families.base import ModelConfig
from ..objectives import ObjectiveConfig
from ..semantics import ArtifactSemantics
from ..temporal.scaling import ScalerStats
from .training_run import TrainingRunResult

JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class _FrozenJsonObject:
    items: tuple[tuple[str, FrozenJsonValue], ...]


@dataclass(frozen=True, slots=True)
class _FrozenJsonArray:
    items: tuple[FrozenJsonValue, ...]


FrozenJsonValue: TypeAlias = JsonScalar | _FrozenJsonObject | _FrozenJsonArray


@dataclass(frozen=True, slots=True)
class TrainingArtifactManifest:
    """Single-source persisted artifact provenance plus exact authored config payloads."""

    artifact_id: str
    dataset_builder: DatasetBuilderConfig
    prediction: PredictionConfig
    objective: ObjectiveConfig
    chain_name: str
    dataset_id: str
    dataset_name: str
    problem: ProblemSpec
    variant: ArtifactVariant
    study: StudyConfig | None
    study_id: str | None
    features: FeaturesConfig
    model: ModelConfig[str]
    split: SplitConfig
    training: TrainingConfig
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
    def features_id(self) -> str:
        return self.semantics.feature.features_id

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
    best_objective_metric_id: str
    best_objective_value: float
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
    objective_metrics: MetricSet


@dataclass(frozen=True, slots=True)
class EvaluationExecutionProvenance:
    """Execution identity for an evaluation run when launched by the remote backend."""

    execution_ref: str
    job_id: str | None
    log_path: str | None
    workflow_task: str | None
    target: str | None


@dataclass(frozen=True, slots=True)
class EvaluationConfigSnapshot:
    """Immutable JSON-ready evaluator config provenance."""

    _payload: _FrozenJsonObject

    @classmethod
    def from_config(cls, config: EvaluatorConfig) -> EvaluationConfigSnapshot:
        return cls.from_payload(config.model_dump(mode="json", exclude_none=True))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> EvaluationConfigSnapshot:
        from ..evaluation import coerce_evaluator_config

        raw_payload = dict(payload)
        config = coerce_evaluator_config(raw_payload)
        normalized = config.model_dump(mode="json", exclude_none=True)
        if raw_payload != normalized:
            raise ValueError("evaluation config snapshot payload is not canonical JSON")
        return cls(_freeze_json_object(normalized))

    def payload(self) -> JsonObject:
        return _thaw_json_object(self._payload)


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeSummary:
    """Runtime-only evaluation outcomes stored separately from manifest provenance."""

    delay_seconds: int
    evaluation_id: str
    evaluation_config: EvaluationConfigSnapshot
    metric_descriptors: tuple[MetricDescriptor, ...]
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    metrics: MetricSet
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: list[EvaluationRun]
    execution_provenance: EvaluationExecutionProvenance | None = None


@dataclass(frozen=True, slots=True)
class LoadedEvaluationSummary:
    """Read model that pairs artifact provenance with one evaluation runtime summary."""

    evaluation_id: str
    recorded_at: int
    manifest: TrainingArtifactManifest
    runtime: EvaluationRuntimeSummary


def iter_epoch_records(result: TrainingRunResult) -> Iterator[TrainingEpochRecord]:
    for index, (train_metrics, validation_metrics, objective_metrics) in enumerate(
        zip(
            result.training_result.train_history,
            result.training_result.validation_history,
            result.training_result.objective_history,
            strict=True,
        ),
        start=1,
    ):
        yield TrainingEpochRecord(
            epoch=index,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            objective_metrics=objective_metrics,
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
        best_objective_metric_id=result.training_result.objective_metric_id,
        best_objective_value=result.training_result.best_objective_value,
        best_validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
    )


def build_evaluation_runtime_summary(
    *,
    prepared: PreparedInferenceDataset,
    evaluation: EvaluationSummary,
    delay_seconds: int,
    evaluation_id: str,
    evaluation_config: EvaluatorConfig,
    metric_descriptors: tuple[MetricDescriptor, ...],
    execution_provenance: EvaluationExecutionProvenance | None = None,
) -> EvaluationRuntimeSummary:
    return EvaluationRuntimeSummary(
        delay_seconds=delay_seconds,
        evaluation_id=evaluation_id,
        evaluation_config=EvaluationConfigSnapshot.from_config(evaluation_config),
        execution_provenance=execution_provenance,
        metric_descriptors=metric_descriptors,
        n_history_rows=prepared.n_history_rows,
        n_evaluation_rows=prepared.n_evaluation_rows,
        sample_count=prepared.sample_count,
        metrics=evaluation.metrics,
        window_metrics=dict(evaluation.window_metrics),
        total_events=evaluation.total_events,
        runs=list(evaluation.runs),
    )


def _freeze_json_object(payload: Mapping[str, object]) -> _FrozenJsonObject:
    return _FrozenJsonObject(
        tuple((str(key), _freeze_json_value(value)) for key, value in sorted(payload.items()))
    )


def _freeze_json_value(value: object) -> FrozenJsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _freeze_json_object(cast(Mapping[str, object], value))
    if isinstance(value, (list, tuple)):
        items = cast(tuple[object, ...], value)
        return _FrozenJsonArray(tuple(_freeze_json_value(item) for item in items))
    raise TypeError(f"Evaluation config snapshot contains non-JSON value: {type(value).__name__}")


def _thaw_json_object(payload: _FrozenJsonObject) -> JsonObject:
    return {key: _thaw_json_value(value) for key, value in payload.items}


def _thaw_json_value(value: FrozenJsonValue) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _FrozenJsonObject):
        return _thaw_json_object(value)
    return [_thaw_json_value(item) for item in value.items]
