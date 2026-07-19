"""Typed training and historical read-only evaluation summary envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from ..config.models import (
    ArtifactVariant,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
    SequenceConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
)
from ..evaluation.contracts import EvaluationRun
from ..evaluation.registry import coerce_evaluator_config
from ..metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from ..modeling.dataset_builders import SequenceRuntimeMetadata
from ..modeling.families.base import ModelConfig
from ..semantics import ArtifactSemantics
from ..temporal.capability import TemporalCapability
from ..temporal.input_normalization import ScalerStats

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
class TrainingSourceProvenance:
    """Exact source range used to train an artifact or study."""

    corpus_id: str
    window_start_timestamp: int
    window_end_timestamp: int
    first_block: int
    last_block: int
    first_timestamp: int
    last_timestamp: int
    training_cutoff_timestamp: int | None
    source_requirements_fingerprint: str


@dataclass(frozen=True, slots=True)
class TrainingArtifactManifest:
    """Single-source persisted artifact provenance plus exact authored config payloads."""

    artifact_id: str
    sequence: SequenceConfig
    prediction: PredictionConfig
    chain_name: str
    corpus_id: str
    corpus_name: str
    training_source: TrainingSourceProvenance
    problem: ProblemSpec
    variant: ArtifactVariant
    study: StudyConfig | None
    study_id: str | None
    features: FeaturesConfig
    model: ModelConfig[str]
    split: SplitConfig
    training: TrainingConfig
    scaler: ScalerStats
    sequence_runtime_metadata: SequenceRuntimeMetadata
    temporal_capability: TemporalCapability
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
    def lookback_seconds(self) -> int:
        return self.semantics.problem.lookback_seconds

    @property
    def feature_prerequisites(self):
        return self.semantics.feature.feature_prerequisites

    @property
    def action_width(self) -> int:
        return self.temporal_capability.action_width

    @property
    def feature_graph_fingerprint(self) -> str:
        return self.semantics.feature.feature_graph_fingerprint

    @property
    def training_metric_descriptors(self):
        return self.semantics.prediction.training_metric_descriptors

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
    best_validation_total_loss: float
    test_total_loss: float


@dataclass(frozen=True, slots=True)
class LoadedTrainingSummary:
    """Read model that pairs artifact provenance with one training runtime summary."""

    manifest: TrainingArtifactManifest
    runtime: TrainingRuntimeSummary


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
    def from_payload(cls, payload: Mapping[str, object]) -> EvaluationConfigSnapshot:
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
    evaluator_id: str
    evaluation_config: EvaluationConfigSnapshot
    metric_descriptors: tuple[MetricDescriptor, ...]
    scenario_window_start_timestamp: int
    scenario_window_end_timestamp: int
    required_coverage_start_timestamp: int
    required_coverage_end_timestamp: int
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

    evaluation_storage_id: str
    recorded_at: int
    manifest: TrainingArtifactManifest
    runtime: EvaluationRuntimeSummary


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
