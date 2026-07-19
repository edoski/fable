"""Artifact-root SQLite read models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.engine import Connection, RowMapping

from ..core.errors import MissingStateError, StateLayoutError
from .artifact_codecs import (
    ARTIFACT_MANIFEST_CODEC,
    TRAINING_SUMMARY_CODEC,
    decode_evaluation_summary,
)
from .engine import (
    ARTIFACT_ROOT_KIND,
    create_state_engine,
    require_root_kind,
    table_exists,
)
from .payloads import SingletonPayloadStore, mapping_payload
from .schema import artifact_manifest, evaluation_summary, training_summary

if TYPE_CHECKING:
    from ..evaluation.contracts import EvaluationRun
    from ..modeling.results import (
        LoadedEvaluationSummary,
        LoadedTrainingSummary,
        TrainingArtifactManifest,
    )

_ARTIFACT_MANIFEST_STORE = SingletonPayloadStore(
    table=artifact_manifest,
    codec=ARTIFACT_MANIFEST_CODEC,
)
_TRAINING_SUMMARY_STORE = SingletonPayloadStore(
    table=training_summary,
    codec=TRAINING_SUMMARY_CODEC,
)


def load_artifact_manifest(db_path: Path) -> TrainingArtifactManifest:
    """Load the canonical artifact manifest that owns persisted artifact provenance."""

    if not db_path.is_file():
        raise MissingStateError(f"Missing artifact manifest: {db_path}")
    require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _ARTIFACT_MANIFEST_STORE.load(conn)
        if manifest is None:
            raise MissingStateError(f"Missing artifact manifest: {db_path}")
        return manifest
    finally:
        engine.dispose()


def load_training_summary(db_path: Path) -> LoadedTrainingSummary | None:
    """Load the training read model as manifest plus runtime summary."""

    if not table_exists(db_path, training_summary.name):
        return None
    require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    from ..modeling.results import LoadedTrainingSummary

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _ARTIFACT_MANIFEST_STORE.load(conn)
            summary = _TRAINING_SUMMARY_STORE.load(conn)
        if manifest is None or summary is None:
            return None
        return LoadedTrainingSummary(
            manifest=manifest,
            runtime=summary,
        )
    finally:
        engine.dispose()


def load_evaluation_summary(
    db_path: Path,
    *,
    evaluation_storage_id: str | None = None,
) -> LoadedEvaluationSummary | None:
    """Load the evaluation read model as manifest plus runtime summary."""

    if not table_exists(db_path, evaluation_summary.name):
        return None
    require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    summaries = list_evaluation_summaries(db_path) if evaluation_storage_id is None else []
    if evaluation_storage_id is None:
        if not summaries:
            return None
        if len(summaries) > 1:
            raise StateLayoutError(
                "Multiple evaluation summaries stored; use list_evaluation_summaries() "
                "or specify evaluation_storage_id"
            )
        return summaries[0]
    summary_by_id = {
        summary.evaluation_storage_id: summary
        for summary in list_evaluation_summaries(db_path)
    }
    return summary_by_id.get(evaluation_storage_id)


def list_evaluation_summaries(db_path: Path) -> list[LoadedEvaluationSummary]:
    if not table_exists(db_path, evaluation_summary.name):
        return []
    require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    from ..modeling.results import LoadedEvaluationSummary

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _ARTIFACT_MANIFEST_STORE.load(conn)
            rows = _evaluation_summary_rows(conn)
        if manifest is None:
            return []
        return [
            LoadedEvaluationSummary(
                evaluation_storage_id=str(row["evaluation_id"]),
                recorded_at=int(row["recorded_at"]),
                manifest=manifest,
                runtime=decode_evaluation_summary(
                    mapping_payload(row["payload"], label="evaluation_summary")
                ),
            )
            for row in rows
        ]
    finally:
        engine.dispose()


def list_evaluation_runs(
    db_path: Path,
    *,
    evaluation_storage_id: str | None = None,
) -> list[EvaluationRun]:
    if db_path.is_file():
        require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    if not table_exists(db_path, evaluation_summary.name):
        return []
    if evaluation_storage_id is None:
        summaries = list_evaluation_summaries(db_path)
        if not summaries:
            return []
        if len(summaries) > 1:
            raise StateLayoutError(
                "Multiple evaluation summaries stored; specify evaluation_storage_id "
                "when listing evaluation runs"
            )
        return list(summaries[0].runtime.runs)
    summary = load_evaluation_summary(db_path, evaluation_storage_id=evaluation_storage_id)
    return [] if summary is None else list(summary.runtime.runs)


def _evaluation_summary_rows(conn: Connection) -> list[RowMapping]:
    return list(
        conn.execute(
            select(
                evaluation_summary.c.evaluation_id,
                evaluation_summary.c.recorded_at,
                evaluation_summary.c.payload,
            ).order_by(
                evaluation_summary.c.recorded_at.desc(),
                evaluation_summary.c.evaluation_id,
            )
        )
        .mappings()
        .all()
    )
