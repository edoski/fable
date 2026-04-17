# pyright: strict

"""Artifact-root SQLite persistence."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from ..core.errors import MissingStateError, StateLayoutError
from ..modeling.result_codecs import (
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
from .engine import RootKind, create_state_engine, ensure_state_db, table_exists, touch_meta
from .payloads import PayloadCodec, SequencePayloadStore, SingletonPayloadStore
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

_RAW_PAYLOAD_CODEC = PayloadCodec[dict[str, object]](
    encode=lambda payload: payload,
    decode=lambda payload: payload,
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
    codec=_RAW_PAYLOAD_CODEC,
)
_TRAINING_EPOCH_STORE = SequencePayloadStore(
    table=training_epochs,
    codec=_RAW_PAYLOAD_CODEC,
)


def write_artifact_manifest(
    db_path: Path,
    *,
    manifest: TrainingArtifactManifest,
    root_kind: RootKind,
) -> None:
    """Persist the canonical artifact manifest before any runtime summaries are written."""

    ensure_state_db(db_path, root_kind=root_kind, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            _ARTIFACT_MANIFEST_STORE.upsert(conn, manifest)
            touch_meta(conn, root_kind=root_kind)
    finally:
        engine.dispose()


def load_artifact_manifest(db_path: Path) -> TrainingArtifactManifest:
    """Load the canonical artifact manifest that owns persisted artifact provenance."""

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
    root_kind: RootKind,
    summary: TrainingRuntimeSummary,
    epoch_rows: list[TrainingEpochRecord],
) -> None:
    """Persist training runtime summary plus ordered epoch history for one artifact root."""

    ensure_state_db(db_path, root_kind=root_kind, tables=ARTIFACT_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            _TRAINING_SUMMARY_STORE.upsert(conn, training_summary_payload(summary))
            _TRAINING_EPOCH_STORE.replace(
                conn,
                [
                    {"epoch": record.epoch, "payload": training_epoch_payload(record)}
                    for record in epoch_rows
                ],
            )
            touch_meta(conn, root_kind=root_kind)
    finally:
        engine.dispose()


def load_training_summary(db_path: Path) -> LoadedTrainingSummary | None:
    """Load the training read model as manifest plus runtime summary."""

    if not table_exists(db_path, training_summary.name):
        return None
    from ..modeling.results import LoadedTrainingSummary

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _ARTIFACT_MANIFEST_STORE.load(conn)
            payload = _TRAINING_SUMMARY_STORE.load(conn)
        if manifest is None or payload is None:
            return None
        return LoadedTrainingSummary(
            manifest=manifest,
            runtime=training_summary_from_payload(payload),
        )
    finally:
        engine.dispose()


def list_training_epochs(db_path: Path) -> list[TrainingEpochRecord]:
    if not table_exists(db_path, training_epochs.name):
        return []
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
            training_epoch_from_payload(_payload_mapping(row["payload"]), epoch=int(row["epoch"]))
            for row in rows
        ]
    finally:
        engine.dispose()


def write_evaluation_state(
    db_path: Path,
    *,
    root_kind: RootKind,
    summary: EvaluationRuntimeSummary,
) -> tuple[str, int]:
    """Persist evaluation runtime summary plus ordered run rows for one artifact root."""

    ensure_state_db(db_path, root_kind=root_kind, tables=ARTIFACT_TABLES)
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
            touch_meta(conn, root_kind=root_kind)
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
    from ..modeling.results import LoadedEvaluationSummary

    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _ARTIFACT_MANIFEST_STORE.load(conn)
            rows = (
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
        if manifest is None:
            return []
        summaries: list[LoadedEvaluationSummary] = []
        for row in rows:
            evaluation_id = str(row["evaluation_id"])
            runs = list_evaluation_runs(db_path, evaluation_id=evaluation_id, engine=engine)
            summaries.append(
                LoadedEvaluationSummary(
                    evaluation_id=evaluation_id,
                    recorded_at=int(row["recorded_at"]),
                    manifest=manifest,
                    runtime=evaluation_summary_from_payload(
                        _payload_mapping(row["payload"]),
                        runs=runs,
                    ),
                )
            )
        return summaries
    finally:
        engine.dispose()


def list_evaluation_runs(
    db_path: Path,
    *,
    evaluation_id: str | None = None,
    engine: Engine | None = None,
) -> list[EvaluationRun]:
    if not table_exists(db_path, evaluation_runs.name):
        return []
    owns_engine = engine is None
    active_engine = create_state_engine(db_path) if engine is None else engine
    try:
        with active_engine.connect() as conn:
            resolved_evaluation_id = evaluation_id
            if resolved_evaluation_id is None:
                evaluation_ids = [
                    str(row["evaluation_id"])
                    for row in conn.execute(
                        select(evaluation_summary.c.evaluation_id).order_by(
                            evaluation_summary.c.evaluation_id
                        )
                    )
                    .mappings()
                    .all()
                ]
                if not evaluation_ids:
                    return []
                if len(evaluation_ids) > 1:
                    raise StateLayoutError(
                        "Multiple evaluation summaries stored; specify evaluation_id "
                        "when listing evaluation runs"
                    )
                resolved_evaluation_id = evaluation_ids[0]
            rows = (
                conn.execute(
                    select(evaluation_runs.c.payload)
                    .where(evaluation_runs.c.evaluation_id == resolved_evaluation_id)
                    .order_by(evaluation_runs.c.ordinal)
                )
                .mappings()
                .all()
            )
        return [evaluation_run_from_payload(_payload_mapping(row["payload"])) for row in rows]
    finally:
        if owns_engine:
            active_engine.dispose()


def _evaluation_storage_id(summary: EvaluationRuntimeSummary) -> str:
    canonical_payload = json.dumps(
        {
            "delay_seconds": summary.delay_seconds,
            "evaluator_id": summary.evaluator_id,
            "evaluator_config": summary.evaluator_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_payload).hexdigest()[:16]
    return f"{summary.evaluator_id}-{summary.delay_seconds}s-{digest}"


def _payload_mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Artifact payload must be a mapping")
    mapping = cast(Mapping[object, object], payload)
    return {str(key): value for key, value in mapping.items()}
