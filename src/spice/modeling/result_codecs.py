"""Typed payload codecs for artifact manifests and summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

from ..config import (
    ArtifactVariant,
    coerce_dataset_builder_config,
    StudyConfig,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
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


def mapping_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Expected mapping payload")
    return dict(payload)


def study_config_from_name(study_name: object) -> StudyConfig | None:
    if study_name is None:
        return None
    return StudyConfig(name=str(study_name))


def metric_descriptor_values(descriptor: MetricDescriptor) -> dict[str, str]:
    return {
        "id": descriptor.id,
        "label": descriptor.label,
        "role": descriptor.role,
    }


def metric_descriptor_from_payload(payload: object) -> MetricDescriptor:
    mapping = mapping_payload(payload)
    return MetricDescriptor(
        id=str(mapping["id"]),
        label=str(mapping["label"]),
        role=str(mapping["role"]),  # type: ignore[arg-type]
    )


def metric_set_values(metrics: MetricSet) -> dict[str, float]:
    return dict(metrics.values)


def metric_set_from_payload(payload: object) -> MetricSet:
    mapping = mapping_payload(payload)
    return MetricSet(values={str(key): _float_value(value) for key, value in mapping.items()})


def window_metric_values(summary: WindowMetricSummary) -> dict[str, float]:
    return {"mean": summary.mean, "std": summary.std}


def window_metric_from_payload(payload: object) -> WindowMetricSummary:
    mapping = mapping_payload(payload)
    return WindowMetricSummary(
        mean=_float_value(mapping["mean"]),
        std=_float_value(mapping["std"]),
    )


def metric_descriptors_values(
    descriptors: tuple[MetricDescriptor, ...] | list[MetricDescriptor],
) -> list[dict[str, str]]:
    return [metric_descriptor_values(descriptor) for descriptor in descriptors]


def metric_descriptors_from_payload(payload: object) -> tuple[MetricDescriptor, ...]:
    if not isinstance(payload, list):
        raise TypeError("Expected list payload")
    return tuple(metric_descriptor_from_payload(item) for item in payload)


def artifact_manifest_payload(manifest: TrainingArtifactManifest) -> dict[str, object]:
    return {
        "artifact_id": manifest.artifact_id,
        "dataset_builder": manifest.dataset_builder.model_dump(mode="json", exclude_none=True),
        "prediction": manifest.prediction.model_dump(mode="json"),
        "chain_name": manifest.chain.name,
        "dataset_id": manifest.dataset_id,
        "dataset_name": manifest.dataset_name,
        "problem": manifest.problem.model_dump(mode="json"),
        "variant": manifest.variant.value,
        "study_id": manifest.study_id,
        "study_name": None if manifest.study is None else manifest.study.name,
        "feature_set": manifest.feature_set.model_dump(mode="json", exclude_none=True),
        "model": manifest.model.model_dump(mode="json", exclude_none=True),
        "scaler": manifest.scaler.model_dump(mode="json", exclude_none=True),
        "builder_runtime_metadata": dict(manifest.builder_runtime_metadata),
        "semantics": artifact_semantics_payload(manifest.semantics),
    }


def artifact_manifest_from_payload(payload: dict[str, object]):
    from .results import ArtifactChainMetadata, TrainingArtifactManifest

    return TrainingArtifactManifest(
        artifact_id=str(payload["artifact_id"]),
        dataset_builder=coerce_dataset_builder_config(
            mapping_payload(payload["dataset_builder"])
        ),
        prediction=coerce_prediction_config(mapping_payload(payload["prediction"])),
        chain=ArtifactChainMetadata(name=str(payload["chain_name"])),
        dataset_id=str(payload["dataset_id"]),
        dataset_name=str(payload["dataset_name"]),
        problem=coerce_problem_spec(mapping_payload(payload["problem"])),
        variant=ArtifactVariant(str(payload["variant"])),
        study=study_config_from_name(payload.get("study_name")),
        study_id=_optional_str(payload.get("study_id")),
        feature_set=coerce_feature_set_config(mapping_payload(payload["feature_set"])),
        model=coerce_model_config(mapping_payload(payload["model"])),
        scaler=ScalerStats.model_validate(mapping_payload(payload["scaler"])),
        builder_runtime_metadata=mapping_payload(payload["builder_runtime_metadata"]),
        semantics=artifact_semantics_from_payload(mapping_payload(payload["semantics"])),
    )


def training_summary_payload(summary: TrainingRuntimeSummary) -> dict[str, object]:
    return {
        "n_rows_available": summary.n_rows_available,
        "n_rows_used": summary.n_rows_used,
        "train_samples": summary.split_sizes.train_samples,
        "validation_samples": summary.split_sizes.validation_samples,
        "test_samples": summary.split_sizes.test_samples,
        "best_epoch": summary.best_epoch,
        "resolved_device": summary.resolved_device,
        "resolved_precision": summary.resolved_precision,
        "compiled": summary.compiled,
        "loader_strategy_id": summary.loader_strategy_id,
        "input_storage_mode_id": summary.input_storage_mode_id,
        "target_storage_mode_id": summary.target_storage_mode_id,
        "batch_planner_id": summary.batch_planner_id,
        "best_validation_metrics": metric_set_values(summary.best_validation_metrics),
        "test_metrics": metric_set_values(summary.test_metrics),
    }


def training_summary_from_payload(payload: dict[str, object]):
    from .results import SplitSizes, TrainingRuntimeSummary

    return TrainingRuntimeSummary(
        n_rows_available=_int_value(payload["n_rows_available"]),
        n_rows_used=_int_value(payload["n_rows_used"]),
        split_sizes=SplitSizes(
            train_samples=_int_value(payload["train_samples"]),
            validation_samples=_int_value(payload["validation_samples"]),
            test_samples=_int_value(payload["test_samples"]),
        ),
        best_epoch=_int_value(payload["best_epoch"]),
        resolved_device=str(payload["resolved_device"]),
        resolved_precision=str(payload["resolved_precision"]),
        compiled=bool(payload["compiled"]),
        loader_strategy_id=str(payload["loader_strategy_id"]),
        input_storage_mode_id=str(payload["input_storage_mode_id"]),
        target_storage_mode_id=str(payload["target_storage_mode_id"]),
        batch_planner_id=str(payload["batch_planner_id"]),
        best_validation_metrics=metric_set_from_payload(payload["best_validation_metrics"]),
        test_metrics=metric_set_from_payload(payload["test_metrics"]),
    )


def training_epoch_payload(record: TrainingEpochRecord) -> dict[str, object]:
    return {
        "train_metrics": metric_set_values(record.train_metrics),
        "validation_metrics": metric_set_values(record.validation_metrics),
    }


def training_epoch_from_payload(payload: dict[str, object], *, epoch: int):
    from .results import TrainingEpochRecord

    return TrainingEpochRecord(
        epoch=epoch,
        train_metrics=metric_set_from_payload(payload["train_metrics"]),
        validation_metrics=metric_set_from_payload(payload["validation_metrics"]),
    )


def evaluation_summary_payload(summary: EvaluationRuntimeSummary) -> dict[str, object]:
    return {
        "delay_seconds": summary.delay_seconds,
        "evaluator_id": summary.evaluator_id,
        "evaluator_config": dict(summary.evaluator_config),
        "metric_descriptors": metric_descriptors_values(summary.metric_descriptors),
        "n_history_rows": summary.n_history_rows,
        "n_evaluation_rows": summary.n_evaluation_rows,
        "sample_count": summary.sample_count,
        "metrics": metric_set_values(summary.metrics),
        "window_metrics": {
            metric_id: window_metric_values(metric)
            for metric_id, metric in summary.window_metrics.items()
        },
        "total_events": summary.total_events,
    }


def evaluation_summary_from_payload(
    payload: dict[str, object],
    *,
    runs: list[EvaluationRun],
):
    from .results import EvaluationRuntimeSummary

    window_payload = mapping_payload(payload["window_metrics"])
    return EvaluationRuntimeSummary(
        delay_seconds=_int_value(payload["delay_seconds"]),
        evaluator_id=str(payload["evaluator_id"]),
        evaluator_config=mapping_payload(payload["evaluator_config"]),
        metric_descriptors=metric_descriptors_from_payload(payload["metric_descriptors"]),
        n_history_rows=_int_value(payload["n_history_rows"]),
        n_evaluation_rows=_int_value(payload["n_evaluation_rows"]),
        sample_count=_int_value(payload["sample_count"]),
        metrics=metric_set_from_payload(payload["metrics"]),
        window_metrics={
            str(metric_id): window_metric_from_payload(metric)
            for metric_id, metric in window_payload.items()
        },
        total_events=_int_value(payload["total_events"]),
        runs=runs,
    )


def evaluation_run_payload(run: EvaluationRun) -> dict[str, object]:
    return {
        "n_events": run.n_events,
        "metrics": dict(run.metrics),
        "metadata": dict(run.metadata),
    }


def evaluation_run_from_payload(payload: dict[str, object]):
    return EvaluationRun(
        n_events=_int_value(payload["n_events"]),
        metrics={
            str(key): _float_value(value)
            for key, value in mapping_payload(payload["metrics"]).items()
        },
        metadata={
            str(key): _metadata_value(value)
            for key, value in mapping_payload(payload["metadata"]).items()
        }
    )


def artifact_semantics_payload(semantics: ArtifactSemantics) -> dict[str, object]:
    return {
        "problem": problem_semantics_payload(semantics.problem),
        "feature": feature_semantics_payload(semantics.feature),
        "prediction": prediction_semantics_payload(semantics.prediction),
        "input_normalization": input_normalization_semantics_payload(
            semantics.input_normalization
        ),
        "representation": representation_semantics_payload(semantics.representation),
        "dataset_builder": dataset_builder_semantics_payload(semantics.dataset_builder),
        "max_candidate_slots": semantics.max_candidate_slots,
    }


def artifact_semantics_from_payload(payload: dict[str, object]) -> ArtifactSemantics:
    return ArtifactSemantics(
        problem=problem_semantics_from_payload(mapping_payload(payload["problem"])),
        feature=feature_semantics_from_payload(mapping_payload(payload["feature"])),
        prediction=prediction_semantics_from_payload(mapping_payload(payload["prediction"])),
        input_normalization=input_normalization_semantics_from_payload(
            mapping_payload(
                payload.get(
                    "input_normalization",
                    {"input_normalization_id": "window_weighted_standard"},
                )
            )
        ),
        representation=representation_semantics_from_payload(
            mapping_payload(payload["representation"])
        ),
        dataset_builder=dataset_builder_semantics_from_payload(
            mapping_payload(payload["dataset_builder"])
        ),
        max_candidate_slots=_int_value(payload["max_candidate_slots"]),
    )


def corpus_semantics_payload(semantics: CorpusSemantics) -> dict[str, object]:
    return {
        "problem": problem_semantics_payload(semantics.problem),
        "feature": feature_semantics_payload(semantics.feature),
    }


def corpus_semantics_from_payload(payload: dict[str, object]) -> CorpusSemantics:
    return CorpusSemantics(
        problem=problem_semantics_from_payload(mapping_payload(payload["problem"])),
        feature=feature_semantics_from_payload(mapping_payload(payload["feature"])),
    )


def study_semantics_payload(semantics: StudySemantics) -> dict[str, object]:
    return {
        "problem": problem_semantics_payload(semantics.problem),
        "feature": feature_semantics_payload(semantics.feature),
        "prediction": prediction_semantics_payload(semantics.prediction),
        "input_normalization": input_normalization_semantics_payload(
            semantics.input_normalization
        ),
        "representation": representation_semantics_payload(semantics.representation),
        "dataset_builder": dataset_builder_semantics_payload(semantics.dataset_builder),
    }


def study_semantics_from_payload(payload: dict[str, object]) -> StudySemantics:
    return StudySemantics(
        problem=problem_semantics_from_payload(mapping_payload(payload["problem"])),
        feature=feature_semantics_from_payload(mapping_payload(payload["feature"])),
        prediction=prediction_semantics_from_payload(mapping_payload(payload["prediction"])),
        input_normalization=input_normalization_semantics_from_payload(
            mapping_payload(
                payload.get(
                    "input_normalization",
                    {"input_normalization_id": "window_weighted_standard"},
                )
            )
        ),
        representation=representation_semantics_from_payload(
            mapping_payload(payload["representation"])
        ),
        dataset_builder=dataset_builder_semantics_from_payload(
            mapping_payload(payload["dataset_builder"])
        ),
    )


def feature_semantics_payload(semantics: FeatureSemantics) -> dict[str, object]:
    return {
        "feature_set_id": semantics.feature_set_id,
        "feature_family_id": semantics.feature_family_id,
        "feature_names": list(semantics.feature_names),
        "feature_graph_fingerprint": semantics.feature_graph_fingerprint,
        "feature_prerequisites": semantics.feature_prerequisites.model_dump(mode="json"),
    }


def feature_semantics_from_payload(payload: dict[str, object]) -> FeatureSemantics:
    names = payload["feature_names"]
    if not isinstance(names, list):
        raise TypeError("feature_names payload must be a list")
    return FeatureSemantics(
        feature_set_id=str(payload["feature_set_id"]),
        feature_family_id=str(payload["feature_family_id"]),
        feature_names=tuple(str(name) for name in names),
        feature_graph_fingerprint=str(payload["feature_graph_fingerprint"]),
        feature_prerequisites=FeaturePrerequisites.model_validate(
            mapping_payload(payload["feature_prerequisites"])
        ),
    )


def problem_semantics_payload(semantics: ProblemSemantics) -> dict[str, object]:
    return {
        "compiler_id": semantics.compiler_id,
        "problem_id": semantics.problem_id,
        "lookback_seconds": semantics.lookback_seconds,
        "sample_count": semantics.sample_count,
        "max_delay_seconds": semantics.max_delay_seconds,
    }


def problem_semantics_from_payload(payload: dict[str, object]) -> ProblemSemantics:
    return ProblemSemantics(
        compiler_id=str(payload["compiler_id"]),
        problem_id=str(payload["problem_id"]),
        lookback_seconds=_int_value(payload["lookback_seconds"]),
        sample_count=_int_value(payload["sample_count"]),
        max_delay_seconds=_int_value(payload["max_delay_seconds"]),
    )


def prediction_semantics_payload(semantics: PredictionSemantics) -> dict[str, object]:
    return {
        "prediction_id": semantics.prediction_id,
        "prediction_family_id": semantics.prediction_family_id,
        "training_metric_descriptors": metric_descriptors_values(
            semantics.training_metric_descriptors
        ),
        "progress_metric_descriptors": [
            {
                "id": descriptor.id,
                "label": descriptor.label,
                "width": descriptor.width,
            }
            for descriptor in semantics.progress_metric_descriptors
        ],
        "primary_metric_id": semantics.primary_metric_id,
        "direction": semantics.direction,
        "supported_workflows": sorted(semantics.supported_workflows),
    }


def prediction_semantics_from_payload(payload: dict[str, object]) -> PredictionSemantics:
    from ..core.reporting import StageMetricDescriptor

    progress_payload = payload["progress_metric_descriptors"]
    if not isinstance(progress_payload, list):
        raise TypeError("progress_metric_descriptors payload must be a list")
    workflows_payload = payload["supported_workflows"]
    if not isinstance(workflows_payload, list):
        raise TypeError("supported_workflows payload must be a list")
    return PredictionSemantics(
        prediction_id=str(payload["prediction_id"]),
        prediction_family_id=str(payload["prediction_family_id"]),
        training_metric_descriptors=metric_descriptors_from_payload(
            payload["training_metric_descriptors"]
        ),
        progress_metric_descriptors=tuple(
            StageMetricDescriptor(
                id=str(item["id"]),
                label=str(item["label"]),
                width=_int_value(item["width"]),
            )
            for item in progress_payload
            if isinstance(item, Mapping)
        ),
        primary_metric_id=str(payload["primary_metric_id"]),
        direction=_prediction_direction(payload["direction"]),
        supported_workflows=frozenset(str(value) for value in workflows_payload),
    )


def input_normalization_semantics_payload(
    semantics: InputNormalizationSemantics,
) -> dict[str, object]:
    return {"input_normalization_id": semantics.input_normalization_id}


def input_normalization_semantics_from_payload(
    payload: dict[str, object],
) -> InputNormalizationSemantics:
    return InputNormalizationSemantics(
        input_normalization_id=str(payload["input_normalization_id"]),
    )


def representation_semantics_payload(semantics: RepresentationSemantics) -> dict[str, object]:
    return {"representation_id": semantics.representation_id}


def representation_semantics_from_payload(payload: dict[str, object]) -> RepresentationSemantics:
    return RepresentationSemantics(representation_id=str(payload["representation_id"]))


def dataset_builder_semantics_payload(semantics: DatasetBuilderSemantics) -> dict[str, object]:
    return {"dataset_builder_id": semantics.dataset_builder_id}


def dataset_builder_semantics_from_payload(payload: dict[str, object]) -> DatasetBuilderSemantics:
    return DatasetBuilderSemantics(dataset_builder_id=str(payload["dataset_builder_id"]))


def _int_value(value: object) -> int:
    return int(cast(int | float | str | bytes, value))


def _float_value(value: object) -> float:
    return float(cast(int | float | str | bytes, value))


def _metadata_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _prediction_direction(value: object) -> Literal["maximize", "minimize"]:
    resolved = str(value)
    if resolved not in {"maximize", "minimize"}:
        raise ValueError(f"Unsupported prediction direction: {resolved}")
    return cast(Literal["maximize", "minimize"], resolved)
