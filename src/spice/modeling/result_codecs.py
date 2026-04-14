"""Typed state payload codecs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from sqlalchemy.engine import RowMapping

from ..config import (
    ArtifactVariant,
    StudyConfig,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from ..features import FeaturePrerequisites
from ..prediction import MetricDescriptor, MetricSet, WindowMetricSummary
from ..temporal.scaling import ScalerStats
from .families.registry import coerce_model_config

if TYPE_CHECKING:
    from ..prediction import PredictionSimulationRun
    from .artifacts import TrainingArtifactManifest
    from .results import (
        SimulationSummaryRecord,
        TrainingSummary,
    )


def mapping_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Expected mapping payload")
    return dict(payload)

def _row_value(row: RowMapping, key: str) -> object:
    return row[key]


def _row_optional_str(row: RowMapping, key: str) -> str | None:
    value = _row_value(row, key)
    if value is None:
        return None
    return str(value)


def _row_str(row: RowMapping, key: str) -> str:
    return str(_row_value(row, key))


def _row_int(row: RowMapping, key: str) -> int:
    return _int_value(_row_value(row, key))


def _row_float(row: RowMapping, key: str) -> float:
    return _float_value(_row_value(row, key))


def _int_value(value: object) -> int:
    return int(cast(int | float | str | bytes, value))


def _float_value(value: object) -> float:
    return float(cast(int | float | str | bytes, value))


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
    return {
        "mean": summary.mean,
        "std": summary.std,
    }


def window_metric_from_payload(payload: object) -> WindowMetricSummary:
    mapping = mapping_payload(payload)
    return WindowMetricSummary(
        mean=_float_value(mapping["mean"]),
        std=_float_value(mapping["std"]),
    )


def metric_descriptors_values(descriptors: list[MetricDescriptor]) -> list[dict[str, str]]:
    return [metric_descriptor_values(descriptor) for descriptor in descriptors]


def metric_descriptors_from_payload(payload: object) -> list[MetricDescriptor]:
    if not isinstance(payload, list):
        raise TypeError("Expected list payload")
    return [metric_descriptor_from_payload(item) for item in payload]


def artifact_manifest_values(manifest: TrainingArtifactManifest) -> dict[str, object]:
    return {
        "singleton": 1,
        "artifact_id": manifest.artifact_id,
        "prediction_id": manifest.prediction_id,
        "prediction_family_id": manifest.prediction_family_id,
        "prediction": manifest.prediction.model_dump(mode="json"),
        "training_metric_descriptors": metric_descriptors_values(
            manifest.training_metric_descriptors
        ),
        "chain_name": manifest.chain.name,
        "dataset_id": manifest.dataset_id,
        "dataset_name": manifest.dataset_name,
        "problem_id": manifest.problem_id,
        "problem": manifest.problem.model_dump(mode="json"),
        "variant": manifest.variant.value,
        "study_id": manifest.study_id,
        "study_name": None if manifest.study is None else manifest.study.name,
        "model_id": manifest.model.id,
        "max_supported_delay_seconds": manifest.max_supported_delay_seconds,
        "lookback_seconds": manifest.lookback_seconds,
        "feature_set": manifest.feature_set.model_dump(mode="json", exclude_none=True),
        "sample_count": manifest.sample_count,
        "feature_family_id": manifest.feature_family_id,
        "feature_prerequisites": manifest.feature_prerequisites.model_dump(mode="json"),
        "max_candidate_slots": manifest.max_candidate_slots,
        "feature_set_id": manifest.feature_set_id,
        "feature_names": list(manifest.feature_names),
        "feature_graph_fingerprint": manifest.feature_graph_fingerprint,
        "model": manifest.model.model_dump(mode="json", exclude_none=True),
        "scaler": manifest.scaler.model_dump(mode="json", exclude_none=True),
        "compiler_runtime_metadata": dict(manifest.compiler_runtime_metadata),
    }


def artifact_manifest_from_row(row: RowMapping):
    from .results import ArtifactChainMetadata, TrainingArtifactManifest

    return TrainingArtifactManifest(
        artifact_id=_row_str(row, "artifact_id"),
        prediction=coerce_prediction_config(mapping_payload(_row_value(row, "prediction"))),
        training_metric_descriptors=metric_descriptors_from_payload(
            _row_value(row, "training_metric_descriptors")
        ),
        chain=ArtifactChainMetadata(name=_row_str(row, "chain_name")),
        dataset_id=_row_str(row, "dataset_id"),
        dataset_name=_row_str(row, "dataset_name"),
        problem=coerce_problem_spec(mapping_payload(_row_value(row, "problem"))),
        variant=ArtifactVariant(_row_str(row, "variant")),
        study=study_config_from_name(_row_value(row, "study_name")),
        study_id=_row_optional_str(row, "study_id"),
        feature_set=coerce_feature_set_config(mapping_payload(_row_value(row, "feature_set"))),
        feature_prerequisites=FeaturePrerequisites.model_validate(
            mapping_payload(_row_value(row, "feature_prerequisites"))
        ),
        max_candidate_slots=_row_int(row, "max_candidate_slots"),
        feature_graph_fingerprint=_row_str(row, "feature_graph_fingerprint"),
        model=coerce_model_config(mapping_payload(_row_value(row, "model"))),
        scaler=ScalerStats.model_validate(mapping_payload(_row_value(row, "scaler"))),
        compiler_runtime_metadata=mapping_payload(_row_value(row, "compiler_runtime_metadata")),
    )


def training_summary_values(summary: TrainingSummary) -> dict[str, object]:
    return {
        "singleton": 1,
        "artifact_id": summary.artifact_id,
        "prediction_id": summary.prediction_id,
        "prediction_family_id": summary.prediction_family_id,
        "training_metric_descriptors": metric_descriptors_values(
            summary.training_metric_descriptors
        ),
        "chain_name": summary.chain,
        "dataset_id": summary.dataset_id,
        "dataset_name": summary.dataset_name,
        "variant": summary.variant.value,
        "study_id": summary.study_id,
        "study_name": None if summary.study is None else summary.study.name,
        "model_id": summary.model_id,
        "problem_id": summary.problem_id,
        "max_supported_delay_seconds": summary.max_supported_delay_seconds,
        "lookback_seconds": summary.lookback_seconds,
        "feature_family_id": summary.feature_family_id,
        "feature_prerequisites": summary.feature_prerequisites.model_dump(mode="json"),
        "sample_count": summary.sample_count,
        "max_candidate_slots": summary.max_candidate_slots,
        "n_rows_available": summary.n_rows_available,
        "n_rows_used": summary.n_rows_used,
        "train_samples": summary.split_sizes.train_samples,
        "validation_samples": summary.split_sizes.validation_samples,
        "test_samples": summary.split_sizes.test_samples,
        "best_epoch": summary.best_epoch,
        "resolved_device": summary.resolved_device,
        "resolved_precision": summary.resolved_precision,
        "compiled": summary.compiled,
        "representation_id": summary.representation_id,
        "storage_mode_id": summary.storage_mode_id,
        "batch_planner_id": summary.batch_planner_id,
        "best_validation_metrics": metric_set_values(summary.best_validation_metrics),
        "test_metrics": metric_set_values(summary.test_metrics),
    }


def training_summary_from_row(row: RowMapping):
    from .results import SplitSizes, TrainingSummary

    return TrainingSummary(
        artifact_id=_row_str(row, "artifact_id"),
        prediction_id=_row_str(row, "prediction_id"),
        prediction_family_id=_row_str(row, "prediction_family_id"),
        training_metric_descriptors=metric_descriptors_from_payload(
            _row_value(row, "training_metric_descriptors")
        ),
        chain=_row_str(row, "chain_name"),
        dataset_id=_row_str(row, "dataset_id"),
        dataset_name=_row_str(row, "dataset_name"),
        variant=ArtifactVariant(_row_str(row, "variant")),
        study=study_config_from_name(_row_value(row, "study_name")),
        study_id=_row_optional_str(row, "study_id"),
        model_id=_row_str(row, "model_id"),
        problem_id=_row_str(row, "problem_id"),
        max_supported_delay_seconds=_row_int(row, "max_supported_delay_seconds"),
        lookback_seconds=_row_int(row, "lookback_seconds"),
        feature_family_id=_row_str(row, "feature_family_id"),
        feature_prerequisites=FeaturePrerequisites.model_validate(
            mapping_payload(_row_value(row, "feature_prerequisites"))
        ),
        sample_count=_row_int(row, "sample_count"),
        max_candidate_slots=_row_int(row, "max_candidate_slots"),
        n_rows_available=_row_int(row, "n_rows_available"),
        n_rows_used=_row_int(row, "n_rows_used"),
        split_sizes=SplitSizes(
            train_samples=_row_int(row, "train_samples"),
            validation_samples=_row_int(row, "validation_samples"),
            test_samples=_row_int(row, "test_samples"),
        ),
        best_epoch=_row_int(row, "best_epoch"),
        resolved_device=_row_str(row, "resolved_device"),
        resolved_precision=_row_str(row, "resolved_precision"),
        compiled=bool(_row_value(row, "compiled")),
        representation_id=_row_str(row, "representation_id"),
        storage_mode_id=_row_str(row, "storage_mode_id"),
        batch_planner_id=_row_str(row, "batch_planner_id"),
        best_validation_metrics=metric_set_from_payload(_row_value(row, "best_validation_metrics")),
        test_metrics=metric_set_from_payload(_row_value(row, "test_metrics")),
    )


def training_epoch_values(record) -> dict[str, object]:
    return {
        "epoch": record.epoch,
        "train_metrics": metric_set_values(record.train_metrics),
        "validation_metrics": metric_set_values(record.validation_metrics),
    }


def training_epoch_from_row(row: RowMapping):
    from .results import TrainingEpochRecord

    return TrainingEpochRecord(
        epoch=_row_int(row, "epoch"),
        train_metrics=metric_set_from_payload(_row_value(row, "train_metrics")),
        validation_metrics=metric_set_from_payload(_row_value(row, "validation_metrics")),
    )


def simulation_summary_values(summary: SimulationSummaryRecord) -> dict[str, object]:
    return {
        "singleton": 1,
        "artifact_id": summary.artifact_id,
        "prediction_id": summary.prediction_id,
        "prediction_family_id": summary.prediction_family_id,
        "simulation_metric_descriptors": metric_descriptors_values(
            summary.simulation_metric_descriptors
        ),
        "chain_name": summary.chain,
        "dataset_id": summary.dataset_id,
        "dataset_name": summary.dataset_name,
        "variant": summary.variant.value,
        "study_id": summary.study_id,
        "study_name": None if summary.study is None else summary.study.name,
        "model_id": summary.model_id,
        "problem_id": summary.problem_id,
        "max_supported_delay_seconds": summary.max_supported_delay_seconds,
        "requested_delay_seconds": summary.requested_delay_seconds,
        "lookback_seconds": summary.lookback_seconds,
        "feature_family_id": summary.feature_family_id,
        "feature_prerequisites": summary.feature_prerequisites.model_dump(mode="json"),
        "simulation_window_seconds": summary.simulation_window_seconds,
        "arrival_rate_per_second": summary.arrival_rate_per_second,
        "repetitions": summary.repetitions,
        "n_history_rows": summary.n_history_rows,
        "n_evaluation_rows": summary.n_evaluation_rows,
        "sample_count": summary.sample_count,
        "max_candidate_slots": summary.max_candidate_slots,
        "metrics": metric_set_values(summary.metrics),
        "window_metrics": {
            metric_id: window_metric_values(metric)
            for metric_id, metric in summary.window_metrics.items()
        },
        "total_events": summary.total_events,
    }


def simulation_summary_from_row(
    row: RowMapping,
    *,
    runs: list[PredictionSimulationRun],
):
    from .results import SimulationSummaryRecord

    window_payload = mapping_payload(_row_value(row, "window_metrics"))
    return SimulationSummaryRecord(
        artifact_id=_row_str(row, "artifact_id"),
        prediction_id=_row_str(row, "prediction_id"),
        prediction_family_id=_row_str(row, "prediction_family_id"),
        simulation_metric_descriptors=metric_descriptors_from_payload(
            _row_value(row, "simulation_metric_descriptors")
        ),
        chain=_row_str(row, "chain_name"),
        dataset_id=_row_str(row, "dataset_id"),
        dataset_name=_row_str(row, "dataset_name"),
        variant=ArtifactVariant(_row_str(row, "variant")),
        study=study_config_from_name(_row_value(row, "study_name")),
        study_id=_row_optional_str(row, "study_id"),
        model_id=_row_str(row, "model_id"),
        problem_id=_row_str(row, "problem_id"),
        max_supported_delay_seconds=_row_int(row, "max_supported_delay_seconds"),
        requested_delay_seconds=_row_int(row, "requested_delay_seconds"),
        lookback_seconds=_row_int(row, "lookback_seconds"),
        feature_family_id=_row_str(row, "feature_family_id"),
        feature_prerequisites=FeaturePrerequisites.model_validate(
            mapping_payload(_row_value(row, "feature_prerequisites"))
        ),
        simulation_window_seconds=_row_int(row, "simulation_window_seconds"),
        arrival_rate_per_second=_row_float(row, "arrival_rate_per_second"),
        repetitions=_row_int(row, "repetitions"),
        n_history_rows=_row_int(row, "n_history_rows"),
        n_evaluation_rows=_row_int(row, "n_evaluation_rows"),
        sample_count=_row_int(row, "sample_count"),
        max_candidate_slots=_row_int(row, "max_candidate_slots"),
        metrics=metric_set_from_payload(_row_value(row, "metrics")),
        window_metrics={
            str(metric_id): window_metric_from_payload(metric)
            for metric_id, metric in window_payload.items()
        },
        total_events=_row_int(row, "total_events"),
        runs=runs,
    )


def simulation_run_values(run: PredictionSimulationRun, *, ordinal: int) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "window_start_timestamp": run.window_start_timestamp,
        "window_end_timestamp": run.window_end_timestamp,
        "n_arrivals": run.n_arrivals,
        "n_events": run.n_events,
        "metrics": dict(run.metrics),
    }


def simulation_run_from_row(row: RowMapping):
    from ..prediction import PredictionSimulationRun

    return PredictionSimulationRun(
        window_start_timestamp=_row_float(row, "window_start_timestamp"),
        window_end_timestamp=_row_float(row, "window_end_timestamp"),
        n_arrivals=_row_int(row, "n_arrivals"),
        n_events=_row_int(row, "n_events"),
        metrics={
            str(key): _float_value(value)
            for key, value in mapping_payload(_row_value(row, "metrics")).items()
        },
    )
