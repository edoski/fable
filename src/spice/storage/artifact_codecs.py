"""Typed payload codecs for artifact-root manifests and summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..config.models import (
    ArtifactVariant,
    PredictionConfig,
    SequenceConfig,
    SplitConfig,
    StudyConfig,
    TrainingConfig,
    coerce_features_config,
    coerce_problem_spec,
)
from ..evaluation.contracts import EvaluationRun
from ..metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from ..modeling.dataset_builders import SequenceRuntimeMetadata
from ..modeling.families.registry import coerce_model_config
from ..temporal.capability import TemporalCapability
from ..temporal.compilers import (
    problem_runtime_metadata_from_compiler_payload,
    problem_runtime_metadata_payload,
)
from ..temporal.input_normalization import ScalerStats
from .payloads import (
    PayloadCodec,
    PayloadRecord,
    decode_payload_record,
    mapping_payload,
    payload_record_codec,
)
from .semantics_codecs import ARTIFACT_SEMANTICS_CODEC

if TYPE_CHECKING:
    from ..modeling.results import (
        EvaluationRuntimeSummary,
        TrainingArtifactManifest,
        TrainingRuntimeSummary,
        TrainingSourceProvenance,
    )


def _study_config_from_name(study_name: object) -> StudyConfig | None:
    if study_name is None:
        return None
    if not isinstance(study_name, str):
        raise TypeError("artifact_manifest.study_name must be a string")
    return StudyConfig(name=study_name)


class MetricDescriptorPayload(PayloadRecord):
    id: str
    label: str
    role: str
    direction: str | None = None

    def to_descriptor(self) -> MetricDescriptor:
        return MetricDescriptor(
            id=self.id,
            label=self.label,
            role=cast(Any, self.role),
            direction=cast(Any, self.direction),
        )


def _metric_descriptor_from_payload(payload: MetricDescriptorPayload) -> MetricDescriptor:
    return payload.to_descriptor()


class TemporalCapabilityPayload(PayloadRecord):
    compiler_id: str
    max_delay_seconds: int
    action_width: int
    compiler_runtime_metadata: dict[str, object]

    @classmethod
    def from_capability(
        cls,
        capability: TemporalCapability,
    ) -> TemporalCapabilityPayload:
        return cls(
            compiler_id=capability.compiler_id,
            max_delay_seconds=capability.max_delay_seconds,
            action_width=capability.action_width,
            compiler_runtime_metadata=problem_runtime_metadata_payload(
                capability.compiler_id,
                capability.compiler_runtime_metadata,
            ),
        )

    def to_capability(self) -> TemporalCapability:
        return TemporalCapability(
            compiler_id=self.compiler_id,
            max_delay_seconds=self.max_delay_seconds,
            action_width=self.action_width,
            compiler_runtime_metadata=problem_runtime_metadata_from_compiler_payload(
                self.compiler_id,
                self.compiler_runtime_metadata,
            ),
        )


class TrainingSourcePayload(PayloadRecord):
    corpus_id: str
    window_start_timestamp: int
    window_end_timestamp: int
    first_block: int
    last_block: int
    first_timestamp: int
    last_timestamp: int
    training_cutoff_timestamp: int | None = None
    source_requirements_fingerprint: str

    @classmethod
    def from_provenance(
        cls,
        source: TrainingSourceProvenance,
    ) -> TrainingSourcePayload:
        return cls(
            corpus_id=source.corpus_id,
            window_start_timestamp=source.window_start_timestamp,
            window_end_timestamp=source.window_end_timestamp,
            first_block=source.first_block,
            last_block=source.last_block,
            first_timestamp=source.first_timestamp,
            last_timestamp=source.last_timestamp,
            training_cutoff_timestamp=source.training_cutoff_timestamp,
            source_requirements_fingerprint=source.source_requirements_fingerprint,
        )

    def to_provenance(self) -> TrainingSourceProvenance:
        from ..modeling.results import TrainingSourceProvenance

        return TrainingSourceProvenance(
            corpus_id=self.corpus_id,
            window_start_timestamp=self.window_start_timestamp,
            window_end_timestamp=self.window_end_timestamp,
            first_block=self.first_block,
            last_block=self.last_block,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            training_cutoff_timestamp=self.training_cutoff_timestamp,
            source_requirements_fingerprint=self.source_requirements_fingerprint,
        )


class ArtifactManifestPayload(PayloadRecord):
    artifact_id: str
    sequence: dict[str, object]
    prediction: dict[str, object]
    chain_name: str
    corpus_id: str
    corpus_name: str
    training_source: dict[str, object]
    problem: dict[str, object]
    variant: str
    study_id: str | None
    study_name: str | None
    features: dict[str, object]
    model: dict[str, object]
    split: dict[str, object]
    training: dict[str, object]
    scaler: dict[str, object]
    sequence_runtime_metadata: dict[str, object]
    temporal_capability: dict[str, object]
    semantics: dict[str, object]

    @classmethod
    def from_manifest(cls, manifest: TrainingArtifactManifest) -> ArtifactManifestPayload:
        return cls(
            artifact_id=manifest.artifact_id,
            sequence=manifest.sequence.model_dump(mode="json"),
            prediction=manifest.prediction.model_dump(mode="json"),
            chain_name=manifest.chain_name,
            corpus_id=manifest.corpus_id,
            corpus_name=manifest.corpus_name,
            training_source=TrainingSourcePayload.from_provenance(
                manifest.training_source
            ).model_dump(mode="json"),
            problem=manifest.problem.model_dump(mode="json"),
            variant=manifest.variant.value,
            study_id=manifest.study_id,
            study_name=None if manifest.study is None else manifest.study.name,
            features=manifest.features.model_dump(mode="json", exclude_none=True),
            model=manifest.model.model_dump(mode="json", exclude_none=True),
            split=manifest.split.model_dump(mode="json"),
            training=manifest.training.model_dump(mode="json"),
            scaler=manifest.scaler.model_dump(mode="json", exclude_none=True),
            sequence_runtime_metadata=manifest.sequence_runtime_metadata.model_dump(mode="json"),
            temporal_capability=TemporalCapabilityPayload.from_capability(
                manifest.temporal_capability
            ).model_dump(mode="json"),
            semantics=ARTIFACT_SEMANTICS_CODEC.encode(manifest.semantics),
        )

    def to_manifest(self) -> TrainingArtifactManifest:
        from ..modeling.results import TrainingArtifactManifest

        temporal_capability = TemporalCapabilityPayload.model_validate(
            mapping_payload(
                self.temporal_capability,
                label="artifact_manifest.temporal_capability",
            ),
            strict=True,
        ).to_capability()
        semantics = ARTIFACT_SEMANTICS_CODEC.decode(self.semantics)
        if semantics.temporal_capability != temporal_capability.semantics:
            raise ValueError(
                "artifact manifest temporal capability semantics do not match "
                "temporal_capability"
            )
        return TrainingArtifactManifest(
            artifact_id=self.artifact_id,
            sequence=SequenceConfig.model_validate(self.sequence),
            prediction=PredictionConfig.model_validate(self.prediction),
            chain_name=self.chain_name,
            corpus_id=self.corpus_id,
            corpus_name=self.corpus_name,
            training_source=TrainingSourcePayload.model_validate(
                mapping_payload(
                    self.training_source,
                    label="artifact_manifest.training_source",
                ),
                strict=True,
            ).to_provenance(),
            problem=coerce_problem_spec(self.problem),
            variant=ArtifactVariant(self.variant),
            study=_study_config_from_name(self.study_name),
            study_id=self.study_id,
            features=coerce_features_config(self.features),
            model=coerce_model_config(self.model),
            split=SplitConfig.model_validate(self.split),
            training=TrainingConfig.model_validate(self.training),
            scaler=ScalerStats.model_validate(self.scaler),
            sequence_runtime_metadata=SequenceRuntimeMetadata.model_validate(
                mapping_payload(
                    self.sequence_runtime_metadata,
                    label="artifact_manifest.sequence_runtime_metadata",
                ),
                strict=True,
            ),
            temporal_capability=temporal_capability,
            semantics=semantics,
        )


class TrainingSummaryPayload(PayloadRecord):
    n_rows_available: int
    n_rows_used: int
    train_samples: int
    validation_samples: int
    test_samples: int
    best_epoch: int
    best_validation_total_loss: float
    test_total_loss: float

    @classmethod
    def from_runtime(cls, summary: TrainingRuntimeSummary) -> TrainingSummaryPayload:
        return cls(
            n_rows_available=summary.n_rows_available,
            n_rows_used=summary.n_rows_used,
            train_samples=summary.split_sizes.train_samples,
            validation_samples=summary.split_sizes.validation_samples,
            test_samples=summary.split_sizes.test_samples,
            best_epoch=summary.best_epoch,
            best_validation_total_loss=summary.best_validation_total_loss,
            test_total_loss=summary.test_total_loss,
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
            best_validation_total_loss=self.best_validation_total_loss,
            test_total_loss=self.test_total_loss,
        )


class EvaluationRunPayload(PayloadRecord):
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, str | int | float]

    def to_run(self) -> EvaluationRun:
        return EvaluationRun(
            n_events=self.n_events,
            metrics=dict(self.metrics),
            metadata=dict(self.metadata),
        )


class WindowMetricSummaryPayload(PayloadRecord):
    mean: float
    std: float

    def to_summary(self) -> WindowMetricSummary:
        return WindowMetricSummary(mean=self.mean, std=self.std)


class EvaluationExecutionProvenancePayload(PayloadRecord):
    execution_ref: str
    job_id: str | None = None
    log_path: str | None = None
    workflow_task: str | None = None
    target: str | None = None


class EvaluationSummaryPayload(PayloadRecord):
    delay_seconds: int
    evaluator_id: str
    evaluation_config: dict[str, object]
    execution_provenance: EvaluationExecutionProvenancePayload | None = None
    metric_descriptors: list[MetricDescriptorPayload]
    scenario_window_start_timestamp: int
    scenario_window_end_timestamp: int
    required_coverage_start_timestamp: int
    required_coverage_end_timestamp: int
    n_history_rows: int
    n_evaluation_rows: int
    sample_count: int
    metrics: dict[str, float]
    window_metrics: dict[str, WindowMetricSummaryPayload]
    total_events: int
    runs: list[EvaluationRunPayload]

    def to_runtime(self) -> EvaluationRuntimeSummary:
        from ..modeling.results import EvaluationConfigSnapshot, EvaluationRuntimeSummary

        return EvaluationRuntimeSummary(
            delay_seconds=self.delay_seconds,
            evaluator_id=self.evaluator_id,
            evaluation_config=EvaluationConfigSnapshot.from_payload(self.evaluation_config),
            execution_provenance=_execution_provenance_from_payload(
                self.execution_provenance
            ),
            metric_descriptors=tuple(
                _metric_descriptor_from_payload(payload)
                for payload in self.metric_descriptors
            ),
            scenario_window_start_timestamp=self.scenario_window_start_timestamp,
            scenario_window_end_timestamp=self.scenario_window_end_timestamp,
            required_coverage_start_timestamp=self.required_coverage_start_timestamp,
            required_coverage_end_timestamp=self.required_coverage_end_timestamp,
            n_history_rows=self.n_history_rows,
            n_evaluation_rows=self.n_evaluation_rows,
            sample_count=self.sample_count,
            metrics=MetricSet(values=self.metrics),
            window_metrics={
                metric_id: window_metric.to_summary()
                for metric_id, window_metric in self.window_metrics.items()
            },
            total_events=self.total_events,
            runs=[run.to_run() for run in self.runs],
        )


ARTIFACT_MANIFEST_CODEC: PayloadCodec[TrainingArtifactManifest] = payload_record_codec(
    "artifact manifest",
    ArtifactManifestPayload,
    ArtifactManifestPayload.from_manifest,
    ArtifactManifestPayload.to_manifest,
)
TRAINING_SUMMARY_CODEC: PayloadCodec[TrainingRuntimeSummary] = payload_record_codec(
    "training summary",
    TrainingSummaryPayload,
    TrainingSummaryPayload.from_runtime,
    TrainingSummaryPayload.to_runtime,
)
def decode_evaluation_summary(payload: dict[str, object]) -> EvaluationRuntimeSummary:
    return decode_payload_record(
        "evaluation summary",
        EvaluationSummaryPayload,
        payload,
        EvaluationSummaryPayload.to_runtime,
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
