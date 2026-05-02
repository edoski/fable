"""Typed payload codecs for artifact-root manifests and summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..config.models import (
    ArtifactVariant,
    PredictionConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
    coerce_features_config,
    coerce_problem_spec,
)
from ..evaluation import EvaluationRun
from ..metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from ..modeling.dataset_builders import (
    coerce_builder_runtime_metadata,
    coerce_dataset_builder_config,
)
from ..modeling.families.registry import coerce_model_config
from ..objectives import coerce_objective_config
from ..temporal.contracts import (
    temporal_capability_from_payload,
    temporal_capability_payload,
)
from ..temporal.scaling import ScalerStats
from .payloads import (
    PayloadModel,
    decode_payload_model,
    mapping_payload,
    model_payload,
    string_payload,
)
from .semantics_codecs import artifact_semantics_from_payload, artifact_semantics_payload

if TYPE_CHECKING:
    from ..modeling.results import (
        EvaluationRuntimeSummary,
        TrainingArtifactManifest,
        TrainingEpochRecord,
        TrainingRuntimeSummary,
    )


def _metric_values_payload(metrics: MetricSet) -> dict[str, float]:
    return dict(metrics.values)


def _study_config_from_name(study_name: object) -> StudyConfig | None:
    if study_name is None:
        return None
    return StudyConfig(name=string_payload(study_name, label="artifact_manifest.study_name"))


def _metric_descriptor_payload(descriptor: MetricDescriptor) -> dict[str, str]:
    return {
        "id": descriptor.id,
        "label": descriptor.label,
        "role": descriptor.role,
    }


def _metric_descriptor_from_payload(payload: object) -> MetricDescriptor:
    mapping = mapping_payload(payload, label="evaluation_summary.metric_descriptor")
    return MetricDescriptor(
        id=string_payload(mapping["id"], label="evaluation_summary.metric_descriptor.id"),
        label=string_payload(
            mapping["label"],
            label="evaluation_summary.metric_descriptor.label",
        ),
        role=cast(
            Any,
            string_payload(
                mapping["role"],
                label="evaluation_summary.metric_descriptor.role",
            ),
        ),
    )


class ArtifactManifestPayload(PayloadModel):
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
    temporal_capability: dict[str, object]
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
            temporal_capability=temporal_capability_payload(manifest.temporal_capability),
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
            temporal_capability=temporal_capability_from_payload(
                mapping_payload(
                    self.temporal_capability,
                    label="artifact_manifest.temporal_capability",
                )
            ),
            semantics=artifact_semantics_from_payload(self.semantics),
        )


class TrainingSummaryPayload(PayloadModel):
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


class TrainingEpochPayload(PayloadModel):
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


class EvaluationRunPayload(PayloadModel):
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, str | int | float]

    @classmethod
    def from_run(cls, run: EvaluationRun) -> EvaluationRunPayload:
        return cls(
            n_events=run.n_events,
            metrics=dict(run.metrics),
            metadata={key: _metadata_value(value) for key, value in run.metadata.items()},
        )

    def to_run(self) -> EvaluationRun:
        return EvaluationRun(
            n_events=self.n_events,
            metrics=dict(self.metrics),
            metadata=dict(self.metadata),
        )


class WindowMetricSummaryPayload(PayloadModel):
    mean: float
    std: float

    @classmethod
    def from_summary(
        cls,
        summary: WindowMetricSummary,
    ) -> WindowMetricSummaryPayload:
        return cls(mean=summary.mean, std=summary.std)

    def to_summary(self) -> WindowMetricSummary:
        return WindowMetricSummary(mean=self.mean, std=self.std)


class EvaluationExecutionProvenancePayload(PayloadModel):
    execution_ref: str
    job_id: str | None = None
    log_path: str | None = None
    workflow_task: str | None = None
    target: str | None = None


class EvaluationSummaryPayload(PayloadModel):
    delay_seconds: int
    evaluation_id: str
    evaluation_config: dict[str, object]
    execution_provenance: EvaluationExecutionProvenancePayload | None = None
    metric_descriptors: list[dict[str, str]]
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    metrics: dict[str, float]
    window_metrics: dict[str, WindowMetricSummaryPayload]
    total_events: int

    @classmethod
    def from_runtime(cls, summary: EvaluationRuntimeSummary) -> EvaluationSummaryPayload:
        return cls(
            delay_seconds=summary.delay_seconds,
            evaluation_id=summary.evaluation_id,
            evaluation_config=summary.evaluation_config.payload(),
            execution_provenance=_execution_provenance_payload(
                summary.execution_provenance
            ),
            metric_descriptors=[
                _metric_descriptor_payload(descriptor)
                for descriptor in summary.metric_descriptors
            ],
            n_history_rows=summary.n_history_rows,
            n_evaluation_rows=summary.n_evaluation_rows,
            sample_count=summary.sample_count,
            metrics=_metric_values_payload(summary.metrics),
            window_metrics={
                metric_id: WindowMetricSummaryPayload.from_summary(window_metric)
                for metric_id, window_metric in summary.window_metrics.items()
            },
            total_events=summary.total_events,
        )

    def to_runtime(self, *, runs: list[EvaluationRun]) -> EvaluationRuntimeSummary:
        from ..modeling.results import EvaluationConfigSnapshot, EvaluationRuntimeSummary

        return EvaluationRuntimeSummary(
            delay_seconds=self.delay_seconds,
            evaluation_id=self.evaluation_id,
            evaluation_config=EvaluationConfigSnapshot.from_payload(self.evaluation_config),
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
            window_metrics={
                metric_id: window_metric.to_summary()
                for metric_id, window_metric in self.window_metrics.items()
            },
            total_events=self.total_events,
            runs=runs,
        )


def artifact_manifest_payload(manifest: TrainingArtifactManifest) -> dict[str, object]:
    return model_payload(ArtifactManifestPayload.from_manifest(manifest), label="artifact manifest")


def artifact_manifest_from_payload(payload: dict[str, object]):
    return decode_payload_model(
        "artifact manifest",
        ArtifactManifestPayload,
        payload,
        lambda model: model.to_manifest(),
    )


def training_summary_payload(summary: TrainingRuntimeSummary) -> dict[str, object]:
    return model_payload(TrainingSummaryPayload.from_runtime(summary), label="training summary")


def training_summary_from_payload(payload: dict[str, object]):
    return decode_payload_model(
        "training summary",
        TrainingSummaryPayload,
        payload,
        lambda model: model.to_runtime(),
    )


def training_epoch_payload(record: TrainingEpochRecord) -> dict[str, object]:
    return model_payload(TrainingEpochPayload.from_record(record), label="training epoch")


def training_epoch_from_payload(payload: dict[str, object], *, epoch: int):
    return decode_payload_model(
        "training epoch",
        TrainingEpochPayload,
        payload,
        lambda model: model.to_record(epoch=epoch),
    )


def evaluation_summary_payload(summary: EvaluationRuntimeSummary) -> dict[str, object]:
    return model_payload(EvaluationSummaryPayload.from_runtime(summary), label="evaluation summary")


def evaluation_summary_from_payload(
    payload: dict[str, object],
    *,
    runs: list[EvaluationRun],
):
    return decode_payload_model(
        "evaluation summary",
        EvaluationSummaryPayload,
        payload,
        lambda model: model.to_runtime(runs=runs),
    )


def evaluation_run_payload(run: EvaluationRun) -> dict[str, object]:
    return model_payload(EvaluationRunPayload.from_run(run), label="evaluation run")


def evaluation_run_from_payload(payload: dict[str, object]):
    return decode_payload_model(
        "evaluation run",
        EvaluationRunPayload,
        payload,
        lambda model: model.to_run(),
    )


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
