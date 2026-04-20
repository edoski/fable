"""Typed payload codecs for artifact manifests and summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from ..config import (
    ArtifactVariant,
    StudyConfig,
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from ..core.reporting import StageMetricDescriptor
from ..evaluation import EvaluationRun
from ..features import FeaturePrerequisites
from ..prediction import MetricDescriptor, MetricSet, WindowMetricSummary
from ..semantics import (
    ArtifactSemantics,
    CorpusSemantics,
    DatasetBuilderSemantics,
    FeatureSemantics,
    InputNormalizationSemantics,
    PredictionSemantics,
    ProblemSemantics,
    RealizationPolicySemantics,
    RepresentationSemantics,
    StudySemantics,
)
from ..temporal.scaling import ScalerStats
from .families.registry import coerce_model_config

if TYPE_CHECKING:
    from .results import (
        EvaluationRuntimeSummary,
        TrainingArtifactManifest,
        TrainingEpochRecord,
        TrainingRuntimeSummary,
    )


class CodecPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_METRIC_DESCRIPTOR_ADAPTER = TypeAdapter(MetricDescriptor)
_WINDOW_METRIC_ADAPTER = TypeAdapter(WindowMetricSummary)
_EVALUATION_RUN_ADAPTER = TypeAdapter(EvaluationRun)
_FEATURE_PREREQUISITES_ADAPTER = TypeAdapter(FeaturePrerequisites)
_PROBLEM_SEMANTICS_ADAPTER = TypeAdapter(ProblemSemantics)
_FEATURE_SEMANTICS_ADAPTER = TypeAdapter(FeatureSemantics)
_PREDICTION_SEMANTICS_ADAPTER = TypeAdapter(PredictionSemantics)
_INPUT_NORMALIZATION_SEMANTICS_ADAPTER = TypeAdapter(InputNormalizationSemantics)
_REALIZATION_POLICY_SEMANTICS_ADAPTER = TypeAdapter(RealizationPolicySemantics)
_REPRESENTATION_SEMANTICS_ADAPTER = TypeAdapter(RepresentationSemantics)
_DATASET_BUILDER_SEMANTICS_ADAPTER = TypeAdapter(DatasetBuilderSemantics)
_CORPUS_SEMANTICS_ADAPTER = TypeAdapter(CorpusSemantics)
_STUDY_SEMANTICS_ADAPTER = TypeAdapter(StudySemantics)
_ARTIFACT_SEMANTICS_ADAPTER = TypeAdapter(ArtifactSemantics)

_ADAPTER_NAMESPACE = {
    "FeaturePrerequisites": FeaturePrerequisites,
    "MetricDescriptor": MetricDescriptor,
    "StageMetricDescriptor": StageMetricDescriptor,
}
for _adapter in (
    _FEATURE_PREREQUISITES_ADAPTER,
    _PROBLEM_SEMANTICS_ADAPTER,
    _FEATURE_SEMANTICS_ADAPTER,
    _PREDICTION_SEMANTICS_ADAPTER,
    _INPUT_NORMALIZATION_SEMANTICS_ADAPTER,
    _REALIZATION_POLICY_SEMANTICS_ADAPTER,
    _REPRESENTATION_SEMANTICS_ADAPTER,
    _DATASET_BUILDER_SEMANTICS_ADAPTER,
    _CORPUS_SEMANTICS_ADAPTER,
    _STUDY_SEMANTICS_ADAPTER,
    _ARTIFACT_SEMANTICS_ADAPTER,
):
    _adapter.rebuild(_types_namespace=_ADAPTER_NAMESPACE)


def mapping_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Expected mapping payload")
    return dict(payload)


def _adapter_payload(adapter: object, value: object) -> dict[str, object]:
    payload = cast(TypeAdapter[Any], adapter).dump_python(value, mode="json")
    if not isinstance(payload, dict):
        raise TypeError("Expected adapter to serialize to a mapping payload")
    return cast(dict[str, object], payload)


AdapterValueT = TypeVar("AdapterValueT")


def _adapter_value(adapter: TypeAdapter[AdapterValueT], payload: object) -> AdapterValueT:
    return adapter.validate_python(payload)


def _metric_values_payload(metrics: MetricSet) -> dict[str, float]:
    return dict(metrics.values)


def _metric_set_from_payload(payload: object) -> MetricSet:
    mapping = mapping_payload(payload)
    return MetricSet(values={str(key): _float_value(value) for key, value in mapping.items()})


def _study_config_from_name(study_name: object) -> StudyConfig | None:
    if study_name is None:
        return None
    return StudyConfig(name=str(study_name))


def _metric_descriptor_payload(descriptor: MetricDescriptor) -> dict[str, str]:
    return {
        "id": descriptor.id,
        "label": descriptor.label,
        "role": descriptor.role,
    }


def _metric_descriptor_from_payload(payload: object) -> MetricDescriptor:
    mapping = mapping_payload(payload)
    return MetricDescriptor(
        id=str(mapping["id"]),
        label=str(mapping["label"]),
        role=cast(Any, str(mapping["role"])),
    )


def _prediction_semantics_payload(semantics: PredictionSemantics) -> dict[str, object]:
    payload = _adapter_payload(_PREDICTION_SEMANTICS_ADAPTER, semantics)
    payload["supported_workflows"] = sorted(str(value) for value in semantics.supported_workflows)
    return payload


class ArtifactManifestPayload(CodecPayloadModel):
    artifact_id: str
    dataset_builder: dict[str, object]
    prediction: dict[str, object]
    chain_name: str
    dataset_id: str
    dataset_name: str
    problem: dict[str, object]
    variant: str
    study_id: str | None
    study_name: str | None
    feature_set: dict[str, object]
    model: dict[str, object]
    scaler: dict[str, object]
    builder_runtime_metadata: dict[str, object]
    semantics: dict[str, object]

    @classmethod
    def from_manifest(cls, manifest: TrainingArtifactManifest) -> ArtifactManifestPayload:
        return cls(
            artifact_id=manifest.artifact_id,
            dataset_builder=manifest.dataset_builder.model_dump(mode="json", exclude_none=True),
            prediction=manifest.prediction.model_dump(mode="json"),
            chain_name=manifest.chain_name,
            dataset_id=manifest.dataset_id,
            dataset_name=manifest.dataset_name,
            problem=manifest.problem.model_dump(mode="json"),
            variant=manifest.variant.value,
            study_id=manifest.study_id,
            study_name=None if manifest.study is None else manifest.study.name,
            feature_set=manifest.feature_set.model_dump(mode="json", exclude_none=True),
            model=manifest.model.model_dump(mode="json", exclude_none=True),
            scaler=manifest.scaler.model_dump(mode="json", exclude_none=True),
            builder_runtime_metadata=dict(manifest.builder_runtime_metadata),
            semantics=artifact_semantics_payload(manifest.semantics),
        )

    def to_manifest(self) -> TrainingArtifactManifest:
        from .results import TrainingArtifactManifest

        return TrainingArtifactManifest(
            artifact_id=self.artifact_id,
            dataset_builder=coerce_dataset_builder_config(self.dataset_builder),
            prediction=coerce_prediction_config(self.prediction),
            chain_name=self.chain_name,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            problem=coerce_problem_spec(self.problem),
            variant=ArtifactVariant(self.variant),
            study=_study_config_from_name(self.study_name),
            study_id=self.study_id,
            feature_set=coerce_feature_set_config(self.feature_set),
            model=coerce_model_config(self.model),
            scaler=ScalerStats.model_validate(self.scaler),
            builder_runtime_metadata=mapping_payload(
                self.builder_runtime_metadata,
            ),
            semantics=artifact_semantics_from_payload(self.semantics),
        )


class TrainingSummaryPayload(CodecPayloadModel):
    n_rows_available: int
    n_rows_used: int
    train_samples: int
    validation_samples: int
    test_samples: int
    best_epoch: int
    resolved_device: str
    resolved_precision: str
    compiled: bool
    loader_strategy_id: str
    input_storage_mode_id: str
    target_storage_mode_id: str
    batch_planner_id: str
    best_validation_metrics: dict[str, float]
    test_metrics: dict[str, float]

    @classmethod
    def from_runtime(cls, summary: TrainingRuntimeSummary) -> TrainingSummaryPayload:
        return cls(
            n_rows_available=summary.n_rows_available,
            n_rows_used=summary.n_rows_used,
            train_samples=summary.split_sizes.train_samples,
            validation_samples=summary.split_sizes.validation_samples,
            test_samples=summary.split_sizes.test_samples,
            best_epoch=summary.best_epoch,
            resolved_device=summary.resolved_device,
            resolved_precision=summary.resolved_precision,
            compiled=summary.compiled,
            loader_strategy_id=summary.loader_strategy_id,
            input_storage_mode_id=summary.input_storage_mode_id,
            target_storage_mode_id=summary.target_storage_mode_id,
            batch_planner_id=summary.batch_planner_id,
            best_validation_metrics=_metric_values_payload(summary.best_validation_metrics),
            test_metrics=_metric_values_payload(summary.test_metrics),
        )

    def to_runtime(self) -> TrainingRuntimeSummary:
        from .results import SplitSizes, TrainingRuntimeSummary

        return TrainingRuntimeSummary(
            n_rows_available=self.n_rows_available,
            n_rows_used=self.n_rows_used,
            split_sizes=SplitSizes(
                train_samples=self.train_samples,
                validation_samples=self.validation_samples,
                test_samples=self.test_samples,
            ),
            best_epoch=self.best_epoch,
            resolved_device=self.resolved_device,
            resolved_precision=self.resolved_precision,
            compiled=self.compiled,
            loader_strategy_id=self.loader_strategy_id,
            input_storage_mode_id=self.input_storage_mode_id,
            target_storage_mode_id=self.target_storage_mode_id,
            batch_planner_id=self.batch_planner_id,
            best_validation_metrics=MetricSet(values=self.best_validation_metrics),
            test_metrics=MetricSet(values=self.test_metrics),
        )


class TrainingEpochPayload(CodecPayloadModel):
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]

    @classmethod
    def from_record(cls, record: TrainingEpochRecord) -> TrainingEpochPayload:
        return cls(
            train_metrics=_metric_values_payload(record.train_metrics),
            validation_metrics=_metric_values_payload(record.validation_metrics),
        )

    def to_record(self, *, epoch: int) -> TrainingEpochRecord:
        from .results import TrainingEpochRecord

        return TrainingEpochRecord(
            epoch=epoch,
            train_metrics=MetricSet(values=self.train_metrics),
            validation_metrics=MetricSet(values=self.validation_metrics),
        )


class EvaluationSummaryPayload(CodecPayloadModel):
    delay_seconds: int
    evaluator_id: str
    evaluator_config: dict[str, object]
    metric_descriptors: tuple[dict[str, str], ...]
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    metrics: dict[str, float]
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int

    @classmethod
    def from_runtime(cls, summary: EvaluationRuntimeSummary) -> EvaluationSummaryPayload:
        return cls(
            delay_seconds=summary.delay_seconds,
            evaluator_id=summary.evaluator_id,
            evaluator_config=dict(summary.evaluator_config),
            metric_descriptors=tuple(
                _metric_descriptor_payload(descriptor)
                for descriptor in summary.metric_descriptors
            ),
            n_history_rows=summary.n_history_rows,
            n_evaluation_rows=summary.n_evaluation_rows,
            sample_count=summary.sample_count,
            metrics=_metric_values_payload(summary.metrics),
            window_metrics=dict(summary.window_metrics),
            total_events=summary.total_events,
        )

    def to_runtime(self, *, runs: list[EvaluationRun]) -> EvaluationRuntimeSummary:
        from .results import EvaluationRuntimeSummary

        return EvaluationRuntimeSummary(
            delay_seconds=self.delay_seconds,
            evaluator_id=self.evaluator_id,
            evaluator_config=dict(self.evaluator_config),
            metric_descriptors=tuple(
                _metric_descriptor_from_payload(payload)
                for payload in self.metric_descriptors
            ),
            n_history_rows=self.n_history_rows,
            n_evaluation_rows=self.n_evaluation_rows,
            sample_count=self.sample_count,
            metrics=MetricSet(values=self.metrics),
            window_metrics=dict(self.window_metrics),
            total_events=self.total_events,
            runs=runs,
        )


def artifact_manifest_payload(manifest: TrainingArtifactManifest) -> dict[str, object]:
    return cast(
        dict[str, object],
        ArtifactManifestPayload.from_manifest(manifest).model_dump(mode="json"),
    )


def artifact_manifest_from_payload(payload: dict[str, object]):
    return ArtifactManifestPayload.model_validate(payload).to_manifest()


def training_summary_payload(summary: TrainingRuntimeSummary) -> dict[str, object]:
    return cast(
        dict[str, object],
        TrainingSummaryPayload.from_runtime(summary).model_dump(mode="json"),
    )


def training_summary_from_payload(payload: dict[str, object]):
    return TrainingSummaryPayload.model_validate(payload).to_runtime()


def training_epoch_payload(record: TrainingEpochRecord) -> dict[str, object]:
    return cast(
        dict[str, object],
        TrainingEpochPayload.from_record(record).model_dump(mode="json"),
    )


def training_epoch_from_payload(payload: dict[str, object], *, epoch: int):
    return TrainingEpochPayload.model_validate(payload).to_record(epoch=epoch)


def evaluation_summary_payload(summary: EvaluationRuntimeSummary) -> dict[str, object]:
    return cast(
        dict[str, object],
        EvaluationSummaryPayload.from_runtime(summary).model_dump(mode="json"),
    )


def evaluation_summary_from_payload(
    payload: dict[str, object],
    *,
    runs: list[EvaluationRun],
):
    return EvaluationSummaryPayload.model_validate(payload).to_runtime(runs=runs)


def evaluation_run_payload(run: EvaluationRun) -> dict[str, object]:
    normalized = EvaluationRun(
        n_events=run.n_events,
        metrics=dict(run.metrics),
        metadata={key: _metadata_value(value) for key, value in run.metadata.items()},
    )
    return _adapter_payload(_EVALUATION_RUN_ADAPTER, normalized)


def evaluation_run_from_payload(payload: dict[str, object]):
    return cast(EvaluationRun, _adapter_value(_EVALUATION_RUN_ADAPTER, payload))


def artifact_semantics_payload(semantics: ArtifactSemantics) -> dict[str, object]:
    return _adapter_payload(_ARTIFACT_SEMANTICS_ADAPTER, semantics)


def artifact_semantics_from_payload(payload: dict[str, object]) -> ArtifactSemantics:
    return cast(ArtifactSemantics, _adapter_value(_ARTIFACT_SEMANTICS_ADAPTER, payload))


def corpus_semantics_payload(semantics: CorpusSemantics) -> dict[str, object]:
    return _adapter_payload(_CORPUS_SEMANTICS_ADAPTER, semantics)


def corpus_semantics_from_payload(payload: dict[str, object]) -> CorpusSemantics:
    return cast(CorpusSemantics, _adapter_value(_CORPUS_SEMANTICS_ADAPTER, payload))


def study_semantics_payload(semantics: StudySemantics) -> dict[str, object]:
    return _adapter_payload(_STUDY_SEMANTICS_ADAPTER, semantics)


def study_semantics_from_payload(payload: dict[str, object]) -> StudySemantics:
    return cast(StudySemantics, _adapter_value(_STUDY_SEMANTICS_ADAPTER, payload))


def feature_semantics_payload(semantics: FeatureSemantics) -> dict[str, object]:
    return _adapter_payload(_FEATURE_SEMANTICS_ADAPTER, semantics)


def feature_semantics_from_payload(payload: dict[str, object]) -> FeatureSemantics:
    return cast(FeatureSemantics, _adapter_value(_FEATURE_SEMANTICS_ADAPTER, payload))


def problem_semantics_payload(semantics: ProblemSemantics) -> dict[str, object]:
    return _adapter_payload(_PROBLEM_SEMANTICS_ADAPTER, semantics)


def problem_semantics_from_payload(payload: dict[str, object]) -> ProblemSemantics:
    return cast(ProblemSemantics, _adapter_value(_PROBLEM_SEMANTICS_ADAPTER, payload))


def prediction_semantics_payload(semantics: PredictionSemantics) -> dict[str, object]:
    return _prediction_semantics_payload(semantics)


def prediction_semantics_from_payload(payload: dict[str, object]) -> PredictionSemantics:
    return cast(PredictionSemantics, _adapter_value(_PREDICTION_SEMANTICS_ADAPTER, payload))


def input_normalization_semantics_payload(
    semantics: InputNormalizationSemantics,
) -> dict[str, object]:
    return _adapter_payload(_INPUT_NORMALIZATION_SEMANTICS_ADAPTER, semantics)


def input_normalization_semantics_from_payload(
    payload: dict[str, object],
) -> InputNormalizationSemantics:
    return cast(
        InputNormalizationSemantics,
        _adapter_value(_INPUT_NORMALIZATION_SEMANTICS_ADAPTER, payload),
    )


def representation_semantics_payload(semantics: RepresentationSemantics) -> dict[str, object]:
    return _adapter_payload(_REPRESENTATION_SEMANTICS_ADAPTER, semantics)


def representation_semantics_from_payload(payload: dict[str, object]) -> RepresentationSemantics:
    return cast(
        RepresentationSemantics,
        _adapter_value(_REPRESENTATION_SEMANTICS_ADAPTER, payload),
    )


def dataset_builder_semantics_payload(semantics: DatasetBuilderSemantics) -> dict[str, object]:
    return _adapter_payload(_DATASET_BUILDER_SEMANTICS_ADAPTER, semantics)


def dataset_builder_semantics_from_payload(payload: dict[str, object]) -> DatasetBuilderSemantics:
    return cast(
        DatasetBuilderSemantics,
        _adapter_value(_DATASET_BUILDER_SEMANTICS_ADAPTER, payload),
    )


def _metadata_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _float_value(value: object) -> float:
    return float(cast(int | float | str | bytes, value))
