"""Global catalog for datasets, studies, and artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Column, Integer, MetaData, String, Table, and_, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.sql.elements import ColumnElement

metadata = MetaData()

dataset_index = Table(
    "dataset_index",
    metadata,
    Column("dataset_id", String, primary_key=True),
    Column("dataset_name", String, nullable=False),
    Column("chain_name", String, nullable=False),
    Column("provider_name", String, nullable=False),
    Column("root_path", String, nullable=False),
    Column("state_db_path", String, nullable=False),
    Column("created_at", Integer, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

study_index = Table(
    "study_index",
    metadata,
    Column("study_id", String, primary_key=True),
    Column("study_name", String, nullable=False),
    Column("dataset_id", String, nullable=False),
    Column("dataset_name", String, nullable=False),
    Column("chain_name", String, nullable=False),
    Column("feature_set_id", String, nullable=False),
    Column("prediction_id", String, nullable=False),
    Column("model_id", String, nullable=False),
    Column("problem_id", String, nullable=False),
    Column("root_path", String, nullable=False),
    Column("state_db_path", String, nullable=False),
    Column("created_at", Integer, nullable=False),
    Column("updated_at", Integer, nullable=False),
)

artifact_index = Table(
    "artifact_index",
    metadata,
    Column("artifact_id", String, primary_key=True),
    Column("dataset_id", String, nullable=False),
    Column("dataset_name", String, nullable=False),
    Column("chain_name", String, nullable=False),
    Column("feature_set_id", String, nullable=False),
    Column("prediction_id", String, nullable=False),
    Column("model_id", String, nullable=False),
    Column("problem_id", String, nullable=False),
    Column("variant", String, nullable=False),
    Column("study_id", String),
    Column("study_name", String),
    Column("root_path", String, nullable=False),
    Column("state_db_path", String, nullable=False),
    Column("created_at", Integer, nullable=False),
    Column("updated_at", Integer, nullable=False),
)


@dataclass(frozen=True, slots=True)
class CatalogDatasetRecord:
    dataset_id: str
    dataset_name: str
    chain_name: str
    provider_name: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class CatalogStudyRecord:
    study_id: str
    study_name: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    feature_set_id: str
    prediction_id: str
    model_id: str
    problem_id: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class CatalogArtifactRecord:
    artifact_id: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    feature_set_id: str
    prediction_id: str
    model_id: str
    problem_id: str
    variant: str
    study_id: str | None
    study_name: str | None
    root_path: Path
    state_db_path: Path


def ensure_catalog_db(path: Path) -> None:
    engine = _create_engine(path)
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
    provider_name: str,
    root_path: Path,
    state_db_path: Path,
) -> None:
    now = _now_timestamp()
    values = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "chain_name": chain_name,
        "provider_name": provider_name,
        "root_path": str(root_path),
        "state_db_path": str(state_db_path),
        "created_at": now,
        "updated_at": now,
    }
    _upsert(path, dataset_index, values, key_column=dataset_index.c.dataset_id)


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
    now = _now_timestamp()
    values = {
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
        "created_at": now,
        "updated_at": now,
    }
    _upsert(path, study_index, values, key_column=study_index.c.study_id)


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
    now = _now_timestamp()
    values = {
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
        "created_at": now,
        "updated_at": now,
    }
    _upsert(path, artifact_index, values, key_column=artifact_index.c.artifact_id)


def list_dataset_records(
    path: Path,
    *,
    chain_name: str | None = None,
    dataset_name: str | None = None,
) -> list[CatalogDatasetRecord]:
    return [
        CatalogDatasetRecord(
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            chain_name=str(row["chain_name"]),
            provider_name=str(row["provider_name"]),
            root_path=Path(str(row["root_path"])),
            state_db_path=Path(str(row["state_db_path"])),
        )
        for row in _select_rows(
            path,
            table=dataset_index,
            filters=[
                _eq(dataset_index.c.chain_name, chain_name),
                _eq(dataset_index.c.dataset_name, dataset_name),
            ],
            order_by=[dataset_index.c.chain_name, dataset_index.c.dataset_name],
        )
    ]


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
    return [
        CatalogStudyRecord(
            study_id=str(row["study_id"]),
            study_name=str(row["study_name"]),
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            chain_name=str(row["chain_name"]),
            feature_set_id=str(row["feature_set_id"]),
            prediction_id=str(row["prediction_id"]),
            model_id=str(row["model_id"]),
            problem_id=str(row["problem_id"]),
            root_path=Path(str(row["root_path"])),
            state_db_path=Path(str(row["state_db_path"])),
        )
        for row in _select_rows(
            path,
            table=study_index,
            filters=[
                _eq(study_index.c.chain_name, chain_name),
                _eq(study_index.c.dataset_name, dataset_name),
                _eq(study_index.c.feature_set_id, feature_set_id),
                _eq(study_index.c.prediction_id, prediction_id),
                _eq(study_index.c.model_id, model_id),
                _eq(study_index.c.problem_id, problem_id),
                _eq(study_index.c.study_name, study_name),
            ],
            order_by=[
                study_index.c.chain_name,
                study_index.c.dataset_name,
                study_index.c.feature_set_id,
                study_index.c.prediction_id,
                study_index.c.model_id,
                study_index.c.problem_id,
                study_index.c.study_name,
            ],
        )
    ]


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
    return [
        CatalogArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            chain_name=str(row["chain_name"]),
            feature_set_id=str(row["feature_set_id"]),
            prediction_id=str(row["prediction_id"]),
            model_id=str(row["model_id"]),
            problem_id=str(row["problem_id"]),
            variant=str(row["variant"]),
            study_id=None if row["study_id"] is None else str(row["study_id"]),
            study_name=None if row["study_name"] is None else str(row["study_name"]),
            root_path=Path(str(row["root_path"])),
            state_db_path=Path(str(row["state_db_path"])),
        )
        for row in _select_rows(
            path,
            table=artifact_index,
            filters=[
                _eq(artifact_index.c.chain_name, chain_name),
                _eq(artifact_index.c.dataset_name, dataset_name),
                _eq(artifact_index.c.feature_set_id, feature_set_id),
                _eq(artifact_index.c.prediction_id, prediction_id),
                _eq(artifact_index.c.model_id, model_id),
                _eq(artifact_index.c.problem_id, problem_id),
                _eq(artifact_index.c.variant, variant),
                _eq(artifact_index.c.study_name, study_name),
            ],
            order_by=[
                artifact_index.c.chain_name,
                artifact_index.c.dataset_name,
                artifact_index.c.feature_set_id,
                artifact_index.c.prediction_id,
                artifact_index.c.model_id,
                artifact_index.c.problem_id,
                artifact_index.c.variant,
            ],
        )
    ]


def delete_dataset_record(path: Path, *, dataset_id: str) -> None:
    _delete_row(path, table=dataset_index, key_column=dataset_index.c.dataset_id, key=dataset_id)


def delete_study_record(path: Path, *, study_id: str) -> None:
    _delete_row(path, table=study_index, key_column=study_index.c.study_id, key=study_id)


def delete_artifact_record(path: Path, *, artifact_id: str) -> None:
    _delete_row(
        path,
        table=artifact_index,
        key_column=artifact_index.c.artifact_id,
        key=artifact_id,
    )


def list_studies_for_dataset(path: Path, *, dataset_id: str) -> list[CatalogStudyRecord]:
    return [
        CatalogStudyRecord(
            study_id=str(row["study_id"]),
            study_name=str(row["study_name"]),
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            chain_name=str(row["chain_name"]),
            feature_set_id=str(row["feature_set_id"]),
            prediction_id=str(row["prediction_id"]),
            model_id=str(row["model_id"]),
            problem_id=str(row["problem_id"]),
            root_path=Path(str(row["root_path"])),
            state_db_path=Path(str(row["state_db_path"])),
        )
        for row in _select_rows(
            path,
            table=study_index,
            filters=[study_index.c.dataset_id == dataset_id],
            order_by=[study_index.c.study_name],
        )
    ]


def list_artifacts_for_dataset(path: Path, *, dataset_id: str) -> list[CatalogArtifactRecord]:
    return [
        CatalogArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            chain_name=str(row["chain_name"]),
            feature_set_id=str(row["feature_set_id"]),
            prediction_id=str(row["prediction_id"]),
            model_id=str(row["model_id"]),
            problem_id=str(row["problem_id"]),
            variant=str(row["variant"]),
            study_id=None if row["study_id"] is None else str(row["study_id"]),
            study_name=None if row["study_name"] is None else str(row["study_name"]),
            root_path=Path(str(row["root_path"])),
            state_db_path=Path(str(row["state_db_path"])),
        )
        for row in _select_rows(
            path,
            table=artifact_index,
            filters=[artifact_index.c.dataset_id == dataset_id],
            order_by=[artifact_index.c.variant, artifact_index.c.model_id],
        )
    ]


def list_artifacts_for_study(path: Path, *, study_id: str) -> list[CatalogArtifactRecord]:
    return [
        CatalogArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            chain_name=str(row["chain_name"]),
            feature_set_id=str(row["feature_set_id"]),
            prediction_id=str(row["prediction_id"]),
            model_id=str(row["model_id"]),
            problem_id=str(row["problem_id"]),
            variant=str(row["variant"]),
            study_id=None if row["study_id"] is None else str(row["study_id"]),
            study_name=None if row["study_name"] is None else str(row["study_name"]),
            root_path=Path(str(row["root_path"])),
            state_db_path=Path(str(row["state_db_path"])),
        )
        for row in _select_rows(
            path,
            table=artifact_index,
            filters=[artifact_index.c.study_id == study_id],
            order_by=[artifact_index.c.variant, artifact_index.c.model_id],
        )
    ]


def _create_engine(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path.resolve().as_posix()}", future=True)


def _upsert(
    path: Path,
    table: Table,
    values: dict[str, object],
    *,
    key_column: ColumnElement[Any],
) -> None:
    ensure_catalog_db(path)
    engine = _create_engine(path)
    try:
        statement = sqlite_insert(table).values(**values)
        with engine.begin() as conn:
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[key_column],
                    set_={key: value for key, value in values.items() if key != key_column.name},
                )
            )
    finally:
        engine.dispose()


def _delete_row(path: Path, *, table: Table, key_column: ColumnElement[Any], key: str) -> None:
    ensure_catalog_db(path)
    engine = _create_engine(path)
    try:
        with engine.begin() as conn:
            conn.execute(table.delete().where(key_column == key))
    finally:
        engine.dispose()


def _select_rows(
    path: Path,
    *,
    table: Table,
    filters: list[ColumnElement[bool] | None],
    order_by: list[ColumnElement[Any]],
) -> list[RowMapping]:
    ensure_catalog_db(path)
    engine = _create_engine(path)
    try:
        statement = select(table)
        active_filters: list[ColumnElement[bool]] = [
            condition for condition in filters if condition is not None
        ]
        if active_filters:
            statement = statement.where(and_(*active_filters))
        if order_by:
            statement = statement.order_by(*order_by)
        with engine.connect() as conn:
            return list(conn.execute(statement).mappings().all())
    finally:
        engine.dispose()


def _eq(column: ColumnElement[Any], value: str | None) -> ColumnElement[bool] | None:
    if value is None:
        return None
    return column == value


def _now_timestamp() -> int:
    return int(time.time())
