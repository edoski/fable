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
    artifact_manifest_from_payload,
    artifact_manifest_payload,
    evaluation_run_from_payload,
    evaluation_run_payload,
    evaluation_summary_from_payload,
    evaluation_summary_payload,
    training_epoch_from_payload,
    training_epoch_payload,
    training_summary_from_payload,
    training_summary_payload,
)
from .engine import (
    ARTIFACT_ROOT_KIND,
    create_state_engine,
    ensure_state_db,
    require_root_kind,
    table_exists,
    touch_meta,
)
from .payloads import PayloadCodec, SingletonPayloadStore, mapping_payload
from .schema import (
    ARTIFACT_TABLES,
    artifact_manifest,
    evaluation_runs,
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
    codec=PayloadCodec(
        encode=artifact_manifest_payload,
        decode=artifact_manifest_from_payload,
    ),
)
_TRAINING_SUMMARY_STORE = SingletonPayloadStore(
    table=training_summary,
    codec=PayloadCodec(
        encode=training_summary_payload,
        decode=training_summary_from_payload,
    ),
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
                        {"epoch": record.epoch, "payload": training_epoch_payload(record)}
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
        return [
            training_epoch_from_payload(
                mapping_payload(row["payload"], label="training_epochs"),
                epoch=int(row["epoch"]),
            )
            for row in rows
        ]
    finally:
        engine.dispose()


def upsert_evaluation_state(
    db_path: Path,
    *,
    summary: EvaluationRuntimeSummary,
) -> tuple[str, int]:
    """Persist evaluation runtime summary plus ordered run rows for one artifact root."""

    ensure_state_db(db_path, root_kind=ARTIFACT_ROOT_KIND, tables=ARTIFACT_TABLES)
    evaluation_id = _evaluation_storage_id(summary)
    recorded_at = int(time.time())
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            payload = evaluation_summary_payload(summary)
            statement = sqlite_insert(evaluation_summary).values(
                evaluation_id=evaluation_id,
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
            conn.execute(
                delete(evaluation_runs).where(evaluation_runs.c.evaluation_id == evaluation_id)
            )
            if summary.runs:
                conn.execute(
                    evaluation_runs.insert(),
                    [
                        {
                            "evaluation_id": evaluation_id,
                            "ordinal": ordinal,
                            "payload": evaluation_run_payload(run),
                        }
                        for ordinal, run in enumerate(summary.runs, start=1)
                    ],
                )
            touch_meta(conn, root_kind=ARTIFACT_ROOT_KIND)
    finally:
        engine.dispose()
    return evaluation_id, recorded_at


def load_evaluation_summary(
    db_path: Path,
    *,
    evaluation_id: str | None = None,
) -> LoadedEvaluationSummary | None:
    """Load the evaluation read model as manifest plus runtime summary."""

    if not table_exists(db_path, evaluation_summary.name):
        return None
    require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    summaries = list_evaluation_summaries(db_path) if evaluation_id is None else []
    if evaluation_id is None:
        if not summaries:
            return None
        if len(summaries) > 1:
            raise StateLayoutError(
                "Multiple evaluation summaries stored; use list_evaluation_summaries() "
                "or specify evaluation_id"
            )
        return summaries[0]
    summary_by_id = {
        summary.evaluation_id: summary for summary in list_evaluation_summaries(db_path)
    }
    return summary_by_id.get(evaluation_id)


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
            runs_by_id = _evaluation_runs_by_id(conn)
        if manifest is None:
            return []
        return [
            LoadedEvaluationSummary(
                evaluation_id=str(row["evaluation_id"]),
                recorded_at=int(row["recorded_at"]),
                manifest=manifest,
                runtime=evaluation_summary_from_payload(
                    mapping_payload(row["payload"], label="evaluation_summary"),
                    runs=runs_by_id.get(str(row["evaluation_id"]), []),
                ),
            )
            for row in rows
        ]
    finally:
        engine.dispose()


def list_evaluation_runs(
    db_path: Path,
    *,
    evaluation_id: str | None = None,
) -> list[EvaluationRun]:
    if db_path.is_file():
        require_root_kind(db_path, ARTIFACT_ROOT_KIND)
    if not table_exists(db_path, evaluation_runs.name):
        return []
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            resolved_evaluation_id = evaluation_id
            if resolved_evaluation_id is None:
                evaluation_ids = _evaluation_ids(conn)
                if not evaluation_ids:
                    return []
                if len(evaluation_ids) > 1:
                    raise StateLayoutError(
                        "Multiple evaluation summaries stored; specify evaluation_id "
                        "when listing evaluation runs"
                    )
                resolved_evaluation_id = evaluation_ids[0]
            return _evaluation_runs_by_id(conn).get(resolved_evaluation_id, [])
    finally:
        engine.dispose()


def _evaluation_storage_id(summary: EvaluationRuntimeSummary) -> str:
    identity_payload: dict[str, object] = {
        "delay_seconds": summary.delay_seconds,
        "evaluation_id": summary.evaluation_id,
        "evaluation_config": summary.evaluation_config,
    }
    if summary.execution_provenance is not None:
        identity_payload["execution_provenance"] = {
            "execution_ref": summary.execution_provenance.execution_ref,
            "job_id": summary.execution_provenance.job_id,
            "log_path": summary.execution_provenance.log_path,
            "workflow_task": summary.execution_provenance.workflow_task,
            "target": summary.execution_provenance.target,
        }
    canonical_payload = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_payload).hexdigest()[:16]
    return f"{summary.evaluation_id}-{summary.delay_seconds}s-{digest}"


def _evaluation_ids(conn: Connection) -> list[str]:
    return [
        str(row["evaluation_id"])
        for row in conn.execute(
            select(evaluation_summary.c.evaluation_id).order_by(evaluation_summary.c.evaluation_id)
        )
        .mappings()
        .all()
    ]


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


def _evaluation_runs_by_id(conn: Connection) -> dict[str, list[EvaluationRun]]:
    grouped: dict[str, list[EvaluationRun]] = {}
    for row in (
        conn.execute(
            select(evaluation_runs.c.evaluation_id, evaluation_runs.c.payload).order_by(
                evaluation_runs.c.evaluation_id,
                evaluation_runs.c.ordinal,
            )
        )
        .mappings()
        .all()
    ):
        evaluation_id = str(row["evaluation_id"])
        grouped.setdefault(evaluation_id, []).append(
            evaluation_run_from_payload(mapping_payload(row["payload"], label="evaluation_runs"))
        )
    return grouped
