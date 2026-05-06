"""Catalog SQLite persistence operations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from sqlalchemy import and_, inspect, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.sql.elements import ColumnElement

from ..engine import RootKind, create_sqlite_engine, ensure_table_shapes
from .records import CatalogRecord
from .registry import (
    CatalogRootKindSpec,
    all_root_kind_specs,
    spec_for_record,
    spec_for_root_kind,
)
from .schema import metadata


def ensure_catalog_db(path: Path) -> None:
    engine = _create_engine(path, create_dirs=True)
    try:
        metadata.create_all(engine)
        with engine.begin() as conn:
            ensure_table_shapes(conn, tables=tuple(spec.table for spec in all_root_kind_specs()))
    finally:
        engine.dispose()


def upsert_catalog_record(catalog_path: Path, record: CatalogRecord) -> None:
    spec = spec_for_record(record)
    _upsert_record(catalog_path, spec=spec, values=_record_to_row(spec, record))


def delete_catalog_record(catalog_path: Path, record: CatalogRecord) -> None:
    spec = spec_for_record(record)
    _delete_record(catalog_path, spec=spec, key=_record_key(spec, record))


def list_catalog_records(
    catalog_path: Path,
    root_kind: RootKind,
    *,
    filters: dict[str, str | None],
    order_by: tuple[str, ...] | None = None,
) -> list[CatalogRecord]:
    spec = spec_for_root_kind(root_kind)
    return [
        _record_from_row(spec, row)
        for row in _select_rows(
            catalog_path,
            table=spec.table,
            filters=[
                spec.table.c[name] == value
                for name, value in filters.items()
                if value is not None
            ],
            order_by=[spec.table.c[name] for name in (order_by or spec.default_order)],
        )
    ]


def _record_key(spec: CatalogRootKindSpec[Any], record: CatalogRecord) -> str:
    return str(getattr(spec.require_record(record), spec.key_field))


def _record_to_row(
    spec: CatalogRootKindSpec[Any],
    record: CatalogRecord,
) -> dict[str, object]:
    concrete = spec.require_record(record)
    return {field_name: getattr(concrete, field_name) for field_name in spec.field_names}


def _record_from_row(
    spec: CatalogRootKindSpec[Any],
    row: RowMapping,
) -> CatalogRecord:
    payload: dict[str, object | None] = {}
    for field_name in spec.field_names:
        value = row[field_name]
        if value is None and field_name in spec.nullable_fields:
            payload[field_name] = None
        else:
            payload[field_name] = str(value)
    return spec.record_type(**cast(Any, payload))


def _create_engine(path: Path, *, create_dirs: bool = False):
    return create_sqlite_engine(path, create_dirs=create_dirs)


def _upsert_record(
    path: Path,
    *,
    spec: CatalogRootKindSpec[Any],
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
        key_column = spec.table.c[spec.key_field]
        statement = sqlite_insert(spec.table).values(**payload)
        with engine.begin() as conn:
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[key_column],
                    set_={
                        key: value
                        for key, value in payload.items()
                        if key not in {key_column.name, "created_at"}
                    },
                )
            )
    finally:
        engine.dispose()


def _delete_record(path: Path, *, spec: CatalogRootKindSpec[Any], key: str) -> None:
    if not path.is_file():
        return
    engine = _create_engine(path)
    try:
        if not inspect(engine).has_table(spec.table.name):
            return
        with engine.begin() as conn:
            conn.execute(spec.table.delete().where(spec.table.c[spec.key_field] == key))
    finally:
        engine.dispose()


def _select_rows(
    path: Path,
    *,
    table,
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
