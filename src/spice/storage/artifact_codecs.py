"""Typed payload codecs for artifact-root manifests and summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from ..config.models import (
    ArtifactVariant,
    PredictionConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
    coerce_features_config,
    coerce_problem_spec,
)
from ..core.errors import StateLayoutError
from ..evaluation import EvaluationRun
from ..modeling.dataset_builders import (
    coerce_builder_runtime_metadata,
    coerce_dataset_builder_config,
)
from ..modeling.families.registry import coerce_model_config
from ..objectives import coerce_objective_config
from ..prediction import MetricDescriptor, MetricSet, WindowMetricSummary
from ..temporal.scaling import ScalerStats
from .payloads import mapping_payload
from .semantics_codecs import artifact_semantics_from_payload, artifact_semantics_payload

if TYPE_CHECKING:
    from ..modeling.results import (
        EvaluationRuntimeSummary,
        TrainingArtifactManifest,
        TrainingEpochRecord,
        TrainingRuntimeSummary,
    )


class CodecPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_EVALUATION_RUN_ADAPTER = TypeAdapter(EvaluationRun)
def _adapter_payload(adapter: object, value: object) -> dict[str, object]:
    payload = cast(TypeAdapter[Any], adapter).dump_python(value, mode="json")
    if not isinstance(payload, dict):
        raise StateLayoutError("Expected adapter to serialize to a mapping payload")
    return cast(dict[str, object], payload)


def _metric_values_payload(metrics: MetricSet) -> dict[str, float]:
    return dict(metrics.values)


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
    mapping = mapping_payload(payload, label="evaluation_summary.metric_descriptor")
    return MetricDescriptor(
        id=str(mapping["id"]),
        label=str(mapping["label"]),
        role=cast(Any, str(mapping["role"])),
    )


class ArtifactManifestPayload(CodecPayloadModel):
    artifact_id: str
    dataset_builder: dict[str, object]
    prediction: dict[str, object]
    objective: dict[str, object]
    chain_name: str
    dataset_id: str
    dataset_name: str
    problem: dict[str, object]
    variant: str
    study_id: str | None
    study_name: str | None
    features: dict[str, object]
    model: dict[str, object]
    split: dict[str, object]
    training: dict[str, object]
    scaler: dict[str, object]
    builder_runtime_metadata: dict[str, object]
    semantics: dict[str, object]

    @classmethod
    def from_manifest(cls, manifest: TrainingArtifactManifest) -> ArtifactManifestPayload:
        return cls(
            artifact_id=manifest.artifact_id,
            dataset_builder=manifest.dataset_builder.model_dump(mode="json", exclude_none=True),
            prediction=manifest.prediction.model_dump(mode="json"),
            objective=manifest.objective.model_dump(mode="json", exclude_none=True),
            chain_name=manifest.chain_name,
            dataset_id=manifest.dataset_id,
            dataset_name=manifest.dataset_name,
            problem=manifest.problem.model_dump(mode="json"),
            variant=manifest.variant.value,
            study_id=manifest.study_id,
            study_name=None if manifest.study is None else manifest.study.name,
            features=manifest.features.model_dump(mode="json", exclude_none=True),
            model=manifest.model.model_dump(mode="json", exclude_none=True),
            split=manifest.split.model_dump(mode="json"),
            training=manifest.training.model_dump(mode="json"),
            scaler=manifest.scaler.model_dump(mode="json", exclude_none=True),
            builder_runtime_metadata=manifest.builder_runtime_metadata.model_dump(mode="json"),
            semantics=artifact_semantics_payload(manifest.semantics),
        )

    def to_manifest(self) -> TrainingArtifactManifest:
        from ..modeling.results import TrainingArtifactManifest

        dataset_builder = coerce_dataset_builder_config(self.dataset_builder)
        return TrainingArtifactManifest(
            artifact_id=self.artifact_id,
            dataset_builder=dataset_builder,
            prediction=PredictionConfig.model_validate(self.prediction),
            objective=coerce_objective_config(self.objective),
            chain_name=self.chain_name,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            problem=coerce_problem_spec(self.problem),
            variant=ArtifactVariant(self.variant),
            study=_study_config_from_name(self.study_name),
            study_id=self.study_id,
            features=coerce_features_config(self.features),
            model=coerce_model_config(self.model),
            split=SplitConfig.model_validate(self.split),
            training=TrainingConfig.model_validate(self.training),
            scaler=ScalerStats.model_validate(self.scaler),
            builder_runtime_metadata=coerce_builder_runtime_metadata(
                dataset_builder.id,
                mapping_payload(
                    self.builder_runtime_metadata,
                    label="artifact_manifest.builder_runtime_metadata",
                ),
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
    best_objective_metric_id: str
    best_objective_value: float
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
            best_objective_metric_id=summary.best_objective_metric_id,
            best_objective_value=summary.best_objective_value,
            best_validation_metrics=_metric_values_payload(summary.best_validation_metrics),
            test_metrics=_metric_values_payload(summary.test_metrics),
        )

    def to_runtime(self) -> TrainingRuntimeSummary:
        from ..modeling.results import SplitSizes, TrainingRuntimeSummary

        return TrainingRuntimeSummary(
            n_rows_available=self.n_rows_available,
            n_rows_used=self.n_rows_used,
            split_sizes=SplitSizes(
                train_samples=self.train_samples,
                validation_samples=self.validation_samples,
                test_samples=self.test_samples,
            ),
            best_epoch=self.best_epoch,
            best_objective_metric_id=self.best_objective_metric_id,
            best_objective_value=self.best_objective_value,
            best_validation_metrics=MetricSet(values=self.best_validation_metrics),
            test_metrics=MetricSet(values=self.test_metrics),
        )


class TrainingEpochPayload(CodecPayloadModel):
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    objective_metrics: dict[str, float]

    @classmethod
    def from_record(cls, record: TrainingEpochRecord) -> TrainingEpochPayload:
        return cls(
            train_metrics=_metric_values_payload(record.train_metrics),
            validation_metrics=_metric_values_payload(record.validation_metrics),
            objective_metrics=_metric_values_payload(record.objective_metrics),
        )

    def to_record(self, *, epoch: int) -> TrainingEpochRecord:
        from ..modeling.results import TrainingEpochRecord

        return TrainingEpochRecord(
            epoch=epoch,
            train_metrics=MetricSet(values=self.train_metrics),
            validation_metrics=MetricSet(values=self.validation_metrics),
            objective_metrics=MetricSet(values=self.objective_metrics),
        )


class EvaluationExecutionProvenancePayload(CodecPayloadModel):
    execution_ref: str
    job_id: str | None = None
    log_path: str | None = None
    workflow_task: str | None = None
    target: str | None = None


class EvaluationSummaryPayload(CodecPayloadModel):
    delay_seconds: int
    evaluation_id: str
    evaluation_config: dict[str, object]
    execution_provenance: EvaluationExecutionProvenancePayload | None = None
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
            evaluation_id=summary.evaluation_id,
            evaluation_config=dict(summary.evaluation_config),
            execution_provenance=_execution_provenance_payload(
                summary.execution_provenance
            ),
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
        from ..modeling.results import EvaluationRuntimeSummary

        return EvaluationRuntimeSummary(
            delay_seconds=self.delay_seconds,
            evaluation_id=self.evaluation_id,
            evaluation_config=dict(self.evaluation_config),
            execution_provenance=_execution_provenance_from_payload(
                self.execution_provenance
            ),
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
    return cast(EvaluationRun, _EVALUATION_RUN_ADAPTER.validate_python(payload))


def _execution_provenance_payload(
    provenance: object,
) -> EvaluationExecutionProvenancePayload | None:
    if provenance is None:
        return None
    from ..modeling.results import EvaluationExecutionProvenance

    if not isinstance(provenance, EvaluationExecutionProvenance):
        raise TypeError("evaluation execution provenance has the wrong type")
    return EvaluationExecutionProvenancePayload(
        execution_ref=provenance.execution_ref,
        job_id=provenance.job_id,
        log_path=provenance.log_path,
        workflow_task=provenance.workflow_task,
        target=provenance.target,
    )


def _execution_provenance_from_payload(
    payload: EvaluationExecutionProvenancePayload | None,
):
    if payload is None:
        return None
    from ..modeling.results import EvaluationExecutionProvenance

    return EvaluationExecutionProvenance(
        execution_ref=payload.execution_ref,
        job_id=payload.job_id,
        log_path=payload.log_path,
        workflow_task=payload.workflow_task,
        target=payload.target,
    )


def _metadata_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)
