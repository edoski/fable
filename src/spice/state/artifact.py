"""Artifact-root SQLAlchemy persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..config import ArtifactVariant, StudyConfig
from ..data.normalization import ScalerStats
from ..modeling.registry import coerce_model_config
from .engine import create_state_engine, ensure_state_db, table_exists, touch_meta
from .schema import (
    ARTIFACT_TABLES,
    artifact_manifest,
    simulation_runs,
    simulation_summary,
    training_epochs,
    training_summary,
)

if TYPE_CHECKING:
    from ..modeling.artifacts import TrainingArtifactManifest
    from ..modeling.reporting import (
        MetricsSummary,
        SimulationRunRecord,
        SimulationSummaryRecord,
        TrainingSummary,
    )


def write_artifact_manifest(
    db_path: Path,
    *,
    manifest: TrainingArtifactManifest,
    root_kind: str,
) -> None:
    ensure_state_db(db_path, root_kind=root_kind, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            values = {
                "singleton": 1,
                "chain_name": manifest.chain.name,
                "chain_block_time_seconds": manifest.chain.block_time_seconds,
                "dataset_id": manifest.dataset_id,
                "task_id": manifest.task_id,
                "variant": manifest.variant.value,
                "study_id": None if manifest.study is None else manifest.study.id,
                "max_supported_delay_seconds": manifest.max_supported_delay_seconds,
                "lookback_seconds": manifest.lookback_seconds,
                "sample_count": manifest.sample_count,
                "feature_set_id": manifest.feature_set_id,
                "feature_names": list(manifest.feature_names),
                "feature_graph_fingerprint": manifest.feature_graph_fingerprint,
                "model": manifest.model.model_dump(mode="json", exclude_none=True),
                "scaler": manifest.scaler.model_dump(mode="json", exclude_none=True),
            }
            statement = sqlite_insert(artifact_manifest).values(**values)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[artifact_manifest.c.singleton],
                    set_={key: value for key, value in values.items() if key != "singleton"},
                )
            )
            touch_meta(conn, root_kind=root_kind)
    finally:
        engine.dispose()


def load_artifact_manifest(db_path: Path) -> TrainingArtifactManifest:
    from ..modeling.artifacts import ArtifactChainMetadata, TrainingArtifactManifest

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            row = conn.execute(select(artifact_manifest)).mappings().first()
        if row is None:
            raise ValueError(f"Missing artifact manifest: {db_path}")
        return TrainingArtifactManifest(
            chain=ArtifactChainMetadata(
                name=str(row["chain_name"]),
                block_time_seconds=float(row["chain_block_time_seconds"]),
            ),
            dataset_id=str(row["dataset_id"]),
            task_id=str(row["task_id"]),
            variant=ArtifactVariant(str(row["variant"])),
            study=_study_config(row["study_id"]),
            max_supported_delay_seconds=int(row["max_supported_delay_seconds"]),
            lookback_seconds=int(row["lookback_seconds"]),
            sample_count=int(row["sample_count"]),
            feature_set_id=str(row["feature_set_id"]),
            feature_names=_string_list(row["feature_names"]),
            feature_graph_fingerprint=str(row["feature_graph_fingerprint"]),
            model=coerce_model_config(_mapping(row["model"])),
            scaler=ScalerStats.model_validate(_mapping(row["scaler"])),
        )
    finally:
        engine.dispose()


def write_training_state(
    db_path: Path,
    *,
    root_kind: str,
    summary: TrainingSummary,
    epoch_rows: list[tuple[int, MetricsSummary, MetricsSummary]],
) -> None:
    ensure_state_db(db_path, root_kind=root_kind, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            values = {
                "singleton": 1,
                "chain_name": summary.chain,
                "dataset_id": summary.dataset_id,
                "variant": summary.variant.value,
                "study_id": None if summary.study is None else summary.study.id,
                "model_id": summary.model_id,
                "task_id": summary.task_id,
                "max_supported_delay_seconds": summary.max_supported_delay_seconds,
                "lookback_seconds": summary.lookback_seconds,
                "sample_count": summary.sample_count,
                "n_blocks_available": summary.n_blocks_available,
                "n_blocks_used": summary.n_blocks_used,
                "train_samples": summary.split_sizes.train_samples,
                "validation_samples": summary.split_sizes.validation_samples,
                "test_samples": summary.split_sizes.test_samples,
                "best_epoch": summary.best_epoch,
                "resolved_device": summary.resolved_device,
                "resolved_precision": summary.resolved_precision,
                "compiled": summary.compiled,
                "best_validation_metrics": asdict(summary.best_validation_metrics),
                "test_metrics": asdict(summary.test_metrics),
            }
            statement = sqlite_insert(training_summary).values(**values)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[training_summary.c.singleton],
                    set_={key: value for key, value in values.items() if key != "singleton"},
                )
            )
            conn.execute(delete(training_epochs))
            if epoch_rows:
                conn.execute(
                    training_epochs.insert(),
                    [
                        {
                            "epoch": epoch,
                            "train_metrics": asdict(train_metrics),
                            "validation_metrics": asdict(validation_metrics),
                        }
                        for epoch, train_metrics, validation_metrics in epoch_rows
                    ],
                )
            touch_meta(conn, root_kind=root_kind)
    finally:
        engine.dispose()


def load_training_summary(db_path: Path) -> TrainingSummary | None:
    from ..modeling.reporting import SplitSizes, TrainingSummary

    if not table_exists(db_path, training_summary.name):
        return None
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            row = conn.execute(select(training_summary)).mappings().first()
        if row is None:
            return None
        return TrainingSummary(
            chain=str(row["chain_name"]),
            dataset_id=str(row["dataset_id"]),
            variant=ArtifactVariant(str(row["variant"])),
            study=_study_config(row["study_id"]),
            model_id=str(row["model_id"]),
            task_id=str(row["task_id"]),
            max_supported_delay_seconds=int(row["max_supported_delay_seconds"]),
            lookback_seconds=int(row["lookback_seconds"]),
            sample_count=int(row["sample_count"]),
            n_blocks_available=int(row["n_blocks_available"]),
            n_blocks_used=int(row["n_blocks_used"]),
            split_sizes=SplitSizes(
                train_samples=int(row["train_samples"]),
                validation_samples=int(row["validation_samples"]),
                test_samples=int(row["test_samples"]),
            ),
            best_epoch=int(row["best_epoch"]),
            resolved_device=str(row["resolved_device"]),
            resolved_precision=str(row["resolved_precision"]),
            compiled=bool(row["compiled"]),
            best_validation_metrics=_metrics_from_payload(row["best_validation_metrics"]),
            test_metrics=_metrics_from_payload(row["test_metrics"]),
        )
    finally:
        engine.dispose()


def list_training_epochs(db_path: Path) -> list[dict[str, object]]:
    if not table_exists(db_path, training_epochs.name):
        return []
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(training_epochs).order_by(training_epochs.c.epoch)
            ).mappings().all()
        return [dict(row) for row in rows]
    finally:
        engine.dispose()


def write_simulation_state(
    db_path: Path,
    *,
    root_kind: str,
    summary: SimulationSummaryRecord,
) -> None:
    ensure_state_db(db_path, root_kind=root_kind, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            values = {
                "singleton": 1,
                "chain_name": summary.chain,
                "dataset_id": summary.dataset_id,
                "variant": summary.variant.value,
                "study_id": None if summary.study is None else summary.study.id,
                "model_id": summary.model_id,
                "task_id": summary.task_id,
                "max_supported_delay_seconds": summary.max_supported_delay_seconds,
                "requested_delay_seconds": summary.requested_delay_seconds,
                "lookback_seconds": summary.lookback_seconds,
                "simulation_window_seconds": summary.simulation_window_seconds,
                "arrival_rate_per_second": summary.arrival_rate_per_second,
                "repetitions": summary.repetitions,
                "n_history_context_blocks": summary.n_history_context_blocks,
                "n_evaluation_blocks": summary.n_evaluation_blocks,
                "sample_count": summary.sample_count,
                "profit_over_baseline": asdict(summary.profit_over_baseline),
                "cost_over_optimum": asdict(summary.cost_over_optimum),
                "baseline_cost_over_optimum": asdict(summary.baseline_cost_over_optimum),
                "total_events": summary.total_events,
            }
            statement = sqlite_insert(simulation_summary).values(**values)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[simulation_summary.c.singleton],
                    set_={key: value for key, value in values.items() if key != "singleton"},
                )
            )
            conn.execute(delete(simulation_runs))
            if summary.runs:
                conn.execute(
                    simulation_runs.insert(),
                    [
                        {
                            "ordinal": ordinal,
                            "window_start_timestamp": run.window_start_timestamp,
                            "window_end_timestamp": run.window_end_timestamp,
                            "n_arrivals": run.n_arrivals,
                            "n_events": run.n_events,
                            "profit_over_baseline": run.profit_over_baseline,
                            "cost_over_optimum": run.cost_over_optimum,
                            "baseline_cost_over_optimum": run.baseline_cost_over_optimum,
                        }
                        for ordinal, run in enumerate(summary.runs, start=1)
                    ],
                )
            touch_meta(conn, root_kind=root_kind)
    finally:
        engine.dispose()


def load_simulation_summary(db_path: Path) -> SimulationSummaryRecord | None:
    from ..modeling.reporting import SimulationSummaryRecord

    if not table_exists(db_path, simulation_summary.name):
        return None
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            row = conn.execute(select(simulation_summary)).mappings().first()
        if row is None:
            return None
        runs = list_simulation_runs(db_path)
        return SimulationSummaryRecord(
            chain=str(row["chain_name"]),
            dataset_id=str(row["dataset_id"]),
            variant=ArtifactVariant(str(row["variant"])),
            study=_study_config(row["study_id"]),
            model_id=str(row["model_id"]),
            task_id=str(row["task_id"]),
            max_supported_delay_seconds=int(row["max_supported_delay_seconds"]),
            requested_delay_seconds=int(row["requested_delay_seconds"]),
            lookback_seconds=int(row["lookback_seconds"]),
            simulation_window_seconds=int(row["simulation_window_seconds"]),
            arrival_rate_per_second=float(row["arrival_rate_per_second"]),
            repetitions=int(row["repetitions"]),
            n_history_context_blocks=int(row["n_history_context_blocks"]),
            n_evaluation_blocks=int(row["n_evaluation_blocks"]),
            sample_count=int(row["sample_count"]),
            profit_over_baseline=_aggregate_from_payload(row["profit_over_baseline"]),
            cost_over_optimum=_aggregate_from_payload(row["cost_over_optimum"]),
            baseline_cost_over_optimum=_aggregate_from_payload(row["baseline_cost_over_optimum"]),
            total_events=int(row["total_events"]),
            runs=runs,
        )
    finally:
        engine.dispose()


def list_simulation_runs(db_path: Path) -> list[SimulationRunRecord]:
    from ..modeling.reporting import SimulationRunRecord

    if not table_exists(db_path, simulation_runs.name):
        return []
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(simulation_runs).order_by(simulation_runs.c.ordinal)
            ).mappings().all()
        return [
            SimulationRunRecord(
                window_start_timestamp=float(row["window_start_timestamp"]),
                window_end_timestamp=float(row["window_end_timestamp"]),
                n_arrivals=int(row["n_arrivals"]),
                n_events=int(row["n_events"]),
                profit_over_baseline=float(row["profit_over_baseline"]),
                cost_over_optimum=float(row["cost_over_optimum"]),
                baseline_cost_over_optimum=float(row["baseline_cost_over_optimum"]),
            )
            for row in rows
        ]
    finally:
        engine.dispose()


def _study_config(study_id: object) -> StudyConfig | None:
    if study_id is None:
        return None
    return StudyConfig(id=str(study_id))


def _metrics_from_payload(payload: object) -> MetricsSummary:
    from ..modeling.reporting import MetricsSummary

    mapping = _mapping(payload)
    return MetricsSummary(
        total_loss=float(mapping["total_loss"]),
        accuracy=float(mapping["accuracy"]),
        mean_cost_over_optimum=float(mapping["mean_cost_over_optimum"]),
        mean_profit_over_baseline=float(mapping["mean_profit_over_baseline"]),
    )


def _aggregate_from_payload(payload: object):
    from ..modeling.reporting import SimulationAggregateSummary

    mapping = _mapping(payload)
    return SimulationAggregateSummary(
        mean=float(mapping["mean"]),
        std=float(mapping["std"]),
    )


def _mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("Expected mapping payload")
    return dict(payload)


def _string_list(payload: object) -> list[str]:
    if not isinstance(payload, list):
        raise TypeError("Expected list payload")
    return [str(value) for value in payload]
