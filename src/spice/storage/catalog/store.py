"""Catalog persistence operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from sqlalchemy import Table, and_, create_engine, inspect, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.sql.elements import ColumnElement

from .records import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .schema import artifact_index, dataset_index, metadata, study_index

RecordT = TypeVar("RecordT", CatalogDatasetRecord, CatalogStudyRecord, CatalogArtifactRecord)


@dataclass(frozen=True, slots=True)
class _CatalogRecordSpec(Generic[RecordT]):
    table: Table
    key_name: str
    field_names: tuple[str, ...]
    build: Callable[..., RecordT]
    nullable_fields: frozenset[str] = frozenset()


_DATASET_SPEC = _CatalogRecordSpec(
    table=dataset_index,
    key_name="dataset_id",
    field_names=("dataset_id", "dataset_name", "chain_name", "root_path", "state_db_path"),
    build=CatalogDatasetRecord,
)
_STUDY_SPEC = _CatalogRecordSpec(
    table=study_index,
    key_name="study_id",
    field_names=(
        "study_id",
        "study_name",
        "dataset_id",
        "dataset_name",
        "chain_name",
        "feature_set_id",
        "prediction_id",
        "model_id",
        "problem_id",
        "root_path",
        "state_db_path",
    ),
    build=CatalogStudyRecord,
)
_ARTIFACT_SPEC = _CatalogRecordSpec(
    table=artifact_index,
    key_name="artifact_id",
    field_names=(
        "artifact_id",
        "dataset_id",
        "dataset_name",
        "chain_name",
        "feature_set_id",
        "prediction_id",
        "model_id",
        "problem_id",
        "variant",
        "study_id",
        "study_name",
        "root_path",
        "state_db_path",
    ),
    build=CatalogArtifactRecord,
    nullable_fields=frozenset({"study_id", "study_name"}),
)


def ensure_catalog_db(path: Path) -> None:
    engine = _create_engine(path, create_dirs=True)
    try:
        metadata.create_all(engine)
    finally:
        engine.dispose()


def upsert_dataset_record(
    path: Path,
    *,
    dataset_id: str,
    dataset_name: str,
    chain_name: str,
    root_path: Path,
    state_db_path: Path,
) -> None:
    _upsert_record(
        path,
        spec=_DATASET_SPEC,
        values={
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "chain_name": chain_name,
            "root_path": str(root_path),
            "state_db_path": str(state_db_path),
        },
    )


def upsert_study_record(
    path: Path,
    *,
    study_id: str,
    study_name: str,
    dataset_id: str,
    dataset_name: str,
    chain_name: str,
    feature_set_id: str,
    prediction_id: str,
    model_id: str,
    problem_id: str,
    root_path: Path,
    state_db_path: Path,
) -> None:
    _upsert_record(
        path,
        spec=_STUDY_SPEC,
        values={
            "study_id": study_id,
            "study_name": study_name,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "chain_name": chain_name,
            "feature_set_id": feature_set_id,
            "prediction_id": prediction_id,
            "model_id": model_id,
            "problem_id": problem_id,
            "root_path": str(root_path),
            "state_db_path": str(state_db_path),
        },
    )


def upsert_artifact_record(
    path: Path,
    *,
    artifact_id: str,
    dataset_id: str,
    dataset_name: str,
    chain_name: str,
    feature_set_id: str,
    prediction_id: str,
    model_id: str,
    problem_id: str,
    variant: str,
    study_id: str | None,
    study_name: str | None,
    root_path: Path,
    state_db_path: Path,
) -> None:
    _upsert_record(
        path,
        spec=_ARTIFACT_SPEC,
        values={
            "artifact_id": artifact_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "chain_name": chain_name,
            "feature_set_id": feature_set_id,
            "prediction_id": prediction_id,
            "model_id": model_id,
            "problem_id": problem_id,
            "variant": variant,
            "study_id": study_id,
            "study_name": study_name,
            "root_path": str(root_path),
            "state_db_path": str(state_db_path),
        },
    )


def list_dataset_records(
    path: Path,
    *,
    chain_name: str | None = None,
    dataset_name: str | None = None,
) -> list[CatalogDatasetRecord]:
    return _list_records(
        path,
        spec=_DATASET_SPEC,
        filters={
            "chain_name": chain_name,
            "dataset_name": dataset_name,
        },
        order_by=("chain_name", "dataset_name"),
    )


def list_study_records(
    path: Path,
    *,
    chain_name: str | None = None,
    dataset_name: str | None = None,
    feature_set_id: str | None = None,
    prediction_id: str | None = None,
    model_id: str | None = None,
    problem_id: str | None = None,
    study_name: str | None = None,
) -> list[CatalogStudyRecord]:
    return _list_records(
        path,
        spec=_STUDY_SPEC,
        filters={
            "chain_name": chain_name,
            "dataset_name": dataset_name,
            "feature_set_id": feature_set_id,
            "prediction_id": prediction_id,
            "model_id": model_id,
            "problem_id": problem_id,
            "study_name": study_name,
        },
        order_by=(
            "chain_name",
            "dataset_name",
            "feature_set_id",
            "prediction_id",
            "model_id",
            "problem_id",
            "study_name",
        ),
    )


def list_artifact_records(
    path: Path,
    *,
    chain_name: str | None = None,
    dataset_name: str | None = None,
    feature_set_id: str | None = None,
    prediction_id: str | None = None,
    model_id: str | None = None,
    problem_id: str | None = None,
    variant: str | None = None,
    study_name: str | None = None,
) -> list[CatalogArtifactRecord]:
    return _list_records(
        path,
        spec=_ARTIFACT_SPEC,
        filters={
            "chain_name": chain_name,
            "dataset_name": dataset_name,
            "feature_set_id": feature_set_id,
            "prediction_id": prediction_id,
            "model_id": model_id,
            "problem_id": problem_id,
            "variant": variant,
            "study_name": study_name,
        },
        order_by=(
            "chain_name",
            "dataset_name",
            "feature_set_id",
            "prediction_id",
            "model_id",
            "problem_id",
            "variant",
        ),
    )


def delete_dataset_record(path: Path, *, dataset_id: str) -> None:
    _delete_record(path, spec=_DATASET_SPEC, key=dataset_id)


def delete_study_record(path: Path, *, study_id: str) -> None:
    _delete_record(path, spec=_STUDY_SPEC, key=study_id)


def delete_artifact_record(path: Path, *, artifact_id: str) -> None:
    _delete_record(path, spec=_ARTIFACT_SPEC, key=artifact_id)


def list_studies_for_dataset(path: Path, *, dataset_id: str) -> list[CatalogStudyRecord]:
    return _list_records(
        path,
        spec=_STUDY_SPEC,
        filters={"dataset_id": dataset_id},
        order_by=("study_name",),
    )


def list_artifacts_for_dataset(path: Path, *, dataset_id: str) -> list[CatalogArtifactRecord]:
    return _list_records(
        path,
        spec=_ARTIFACT_SPEC,
        filters={"dataset_id": dataset_id},
        order_by=("variant", "model_id"),
    )


def list_artifacts_for_study(path: Path, *, study_id: str) -> list[CatalogArtifactRecord]:
    return _list_records(
        path,
        spec=_ARTIFACT_SPEC,
        filters={"study_id": study_id},
        order_by=("variant", "model_id"),
    )


def _create_engine(path: Path, *, create_dirs: bool = False):
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path.resolve().as_posix()}", future=True)


def _upsert_record(
    path: Path,
    *,
    spec: _CatalogRecordSpec[Any],
    values: dict[str, object],
) -> None:
    ensure_catalog_db(path)
    engine = _create_engine(path)
    now = _now_timestamp()
    payload = {
        **values,
        "created_at": now,
        "updated_at": now,
    }
    try:
        key_column = spec.table.c[spec.key_name]
        statement = sqlite_insert(spec.table).values(**payload)
        with engine.begin() as conn:
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[key_column],
                    set_={key: value for key, value in payload.items() if key != key_column.name},
                )
            )
    finally:
        engine.dispose()


def _delete_record(path: Path, *, spec: _CatalogRecordSpec[Any], key: str) -> None:
    if not path.is_file():
        return
    engine = _create_engine(path)
    try:
        if not inspect(engine).has_table(spec.table.name):
            return
        with engine.begin() as conn:
            conn.execute(spec.table.delete().where(spec.table.c[spec.key_name] == key))
    finally:
        engine.dispose()


def _list_records(
    path: Path,
    *,
    spec: _CatalogRecordSpec[RecordT],
    filters: dict[str, str | None],
    order_by: tuple[str, ...],
) -> list[RecordT]:
    return [
        _record_from_row(spec, row)
        for row in _select_rows(
            path,
            table=spec.table,
            filters=[
                spec.table.c[name] == value
                for name, value in filters.items()
                if value is not None
            ],
            order_by=[spec.table.c[name] for name in order_by],
        )
    ]


def _record_from_row(spec: _CatalogRecordSpec[RecordT], row: RowMapping) -> RecordT:
    payload: dict[str, object] = {}
    for field_name in spec.field_names:
        value = row[field_name]
        if field_name in {"root_path", "state_db_path"}:
            payload[field_name] = Path(str(value))
        elif value is None and field_name in spec.nullable_fields:
            payload[field_name] = None
        else:
            payload[field_name] = str(value)
    return spec.build(**payload)


def _select_rows(
    path: Path,
    *,
    table: Table,
    filters: list[ColumnElement[bool]],
    order_by: list[ColumnElement[Any]],
) -> list[RowMapping]:
    if not path.is_file():
        return []
    engine = _create_engine(path)
    try:
        if not inspect(engine).has_table(table.name):
            return []
        statement = select(table)
        if filters:
            statement = statement.where(and_(*filters))
        if order_by:
            statement = statement.order_by(*order_by)
        with engine.connect() as conn:
            return list(conn.execute(statement).mappings().all())
    finally:
        engine.dispose()


def _now_timestamp() -> int:
    return int(time.time())
