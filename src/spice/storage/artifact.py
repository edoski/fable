# pyright: strict

"""Artifact-root SQLite persistence."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, RowMapping

from ..core.errors import MissingStateError, StateLayoutError
from .artifact_codecs import (
    ARTIFACT_MANIFEST_CODEC,
    EVALUATION_SUMMARY_CODEC,
    TRAINING_EPOCH_CODEC,
    TRAINING_SUMMARY_CODEC,
)
from .engine import (
    ARTIFACT_ROOT_KIND,
    create_state_engine,
    ensure_state_db,
    require_root_kind,
    table_exists,
    touch_meta,
)
from .payloads import SingletonPayloadStore, mapping_payload
from .schema import (
    ARTIFACT_TABLES,
    artifact_manifest,
    evaluation_summary,
    training_epochs,
    training_summary,
)

if TYPE_CHECKING:
    from ..evaluation import EvaluationRun
    from ..modeling.results import (
        EvaluationRuntimeSummary,
        LoadedEvaluationSummary,
        LoadedTrainingSummary,
        TrainingArtifactManifest,
        TrainingEpochRecord,
        TrainingRuntimeSummary,
    )

_ARTIFACT_MANIFEST_STORE = SingletonPayloadStore(
    table=artifact_manifest,
    codec=ARTIFACT_MANIFEST_CODEC,
)
_TRAINING_SUMMARY_STORE = SingletonPayloadStore(
    table=training_summary,
    codec=TRAINING_SUMMARY_CODEC,
)


def write_artifact_manifest(
    db_path: Path,
    *,
    manifest: TrainingArtifactManifest,
) -> None:
    """Persist the canonical artifact manifest before any runtime summaries are written."""

    ensure_state_db(db_path, root_kind=ARTIFACT_ROOT_KIND, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            _ARTIFACT_MANIFEST_STORE.upsert(conn, manifest)
            touch_meta(conn, root_kind=ARTIFACT_ROOT_KIND)
    finally:
        engine.dispose()


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


def write_training_state(
    db_path: Path,
    *,
    summary: TrainingRuntimeSummary,
    epoch_rows: list[TrainingEpochRecord],
) -> None:
    """Persist training runtime summary plus ordered epoch history for one artifact root."""

    ensure_state_db(db_path, root_kind=ARTIFACT_ROOT_KIND, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            _TRAINING_SUMMARY_STORE.upsert(conn, summary)
            conn.execute(delete(training_epochs))
            if epoch_rows:
                conn.execute(
                    training_epochs.insert(),
                    [
                        {"epoch": record.epoch, "payload": TRAINING_EPOCH_CODEC.encode(record)}
                        for record in epoch_rows
                    ],
                )
            touch_meta(conn, root_kind=ARTIFACT_ROOT_KIND)
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


def list_training_epochs(db_path: Path) -> list[TrainingEpochRecord]:
    if not table_exists(db_path, training_epochs.name):
        return []
    require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    select(training_epochs.c.epoch, training_epochs.c.payload).order_by(
                        training_epochs.c.epoch
                    )
                )
                .mappings()
                .all()
            )
        records = [
            TRAINING_EPOCH_CODEC.decode(mapping_payload(row["payload"], label="training_epochs"))
            for row in rows
        ]
        for row, record in zip(rows, records, strict=True):
            if record.epoch != int(row["epoch"]):
                raise StateLayoutError("training epoch payload does not match row epoch")
        return records
    finally:
        engine.dispose()


def upsert_evaluation_state(
    db_path: Path,
    *,
    summary: EvaluationRuntimeSummary,
) -> tuple[str, int]:
    """Persist evaluation runtime summary plus ordered run rows for one artifact root."""

    ensure_state_db(db_path, root_kind=ARTIFACT_ROOT_KIND, tables=ARTIFACT_TABLES)
    evaluation_storage_id = _evaluation_storage_id(summary)
    recorded_at = int(time.time())
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            payload = EVALUATION_SUMMARY_CODEC.encode(summary)
            statement = sqlite_insert(evaluation_summary).values(
                evaluation_id=evaluation_storage_id,
                recorded_at=recorded_at,
                payload=payload,
            )
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[evaluation_summary.c.evaluation_id],
                    set_={
                        "recorded_at": recorded_at,
                        "payload": payload,
                    },
                )
            )
            touch_meta(conn, root_kind=ARTIFACT_ROOT_KIND)
    finally:
        engine.dispose()
    return evaluation_storage_id, recorded_at


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
                runtime=EVALUATION_SUMMARY_CODEC.decode(
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


def _evaluation_storage_id(summary: EvaluationRuntimeSummary) -> str:
    identity_payload: dict[str, object] = {
        "delay_seconds": summary.delay_seconds,
        "evaluator_id": summary.evaluator_id,
        "evaluation_config": summary.evaluation_config.payload(),
    }
    if summary.execution_provenance is not None:
        identity_payload["execution_provenance"] = {
            "execution_ref": summary.execution_provenance.execution_ref,
        }
    canonical_payload = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_payload).hexdigest()[:16]
    return f"{summary.evaluator_id}-{summary.delay_seconds}s-{digest}"


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
