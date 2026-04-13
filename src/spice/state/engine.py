"""SQLAlchemy engine helpers for per-root state databases."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import Engine, event, inspect, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, create_engine

from .schema import metadata, spice_meta

DATASET_ROOT_KIND = "corpus"
ARTIFACT_ROOT_KIND = "artifact"
STUDY_ROOT_KIND = "study"
STATE_DB_FILENAME = "state.sqlite"


def state_db_path(root: Path) -> Path:
    return root / ".spice" / STATE_DB_FILENAME


def db_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def create_state_engine(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        db_url(path),
        future=True,
        connect_args={"timeout": 5},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA busy_timeout = 5000")
        cursor.close()

    return engine


def ensure_state_db(path: Path, *, root_kind: str, tables: Iterable) -> None:
    managed_tables = (spice_meta, *tuple(tables))
    engine = create_state_engine(path)
    try:
        metadata.create_all(engine, tables=managed_tables)
        with engine.begin() as conn:
            _ensure_table_shapes(conn, tables=managed_tables)
            _ensure_root_kind(conn, root_kind=root_kind)
            touch_meta(conn, root_kind=root_kind)
    finally:
        engine.dispose()


def touch_meta(conn: Connection, *, root_kind: str) -> None:
    now = int(time.time())
    statement = sqlite_insert(spice_meta).values(
        singleton=1,
        root_kind=root_kind,
        created_at=now,
        updated_at=now,
    )
    conn.execute(
        statement.on_conflict_do_update(
            index_elements=[spice_meta.c.singleton],
            set_={
                "root_kind": root_kind,
                "updated_at": now,
            },
        )
    )


def detect_root_kind(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing state database: {path}")
    engine = create_state_engine(path)
    try:
        inspector = inspect(engine)
        if not inspector.has_table(spice_meta.name):
            raise FileNotFoundError(f"Missing SPICE state metadata: {path}")
        with engine.connect() as conn:
            row = conn.execute(
                select(spice_meta.c.root_kind).where(spice_meta.c.singleton == 1)
            ).mappings().first()
        if row is None:
            raise ValueError(f"Missing SPICE state metadata: {path}")
        return str(row["root_kind"])
    finally:
        engine.dispose()


def table_exists(path: Path, table_name: str) -> bool:
    engine = create_state_engine(path)
    try:
        return inspect(engine).has_table(table_name)
    finally:
        engine.dispose()


def _ensure_root_kind(conn: Connection, *, root_kind: str) -> None:
    row = conn.execute(
        select(spice_meta.c.root_kind).where(spice_meta.c.singleton == 1)
    ).mappings().first()
    if row is None:
        return
    if str(row["root_kind"]) != root_kind:
        raise ValueError(
            f"SPICE state root kind mismatch: expected {root_kind}, got {row['root_kind']}"
        )


def _ensure_table_shapes(conn: Connection, *, tables: Iterable) -> None:
    inspector = inspect(conn)
    for table in tables:
        expected_columns = tuple(column.name for column in table.columns)
        actual_columns = tuple(column["name"] for column in inspector.get_columns(table.name))
        if actual_columns != expected_columns:
            raise ValueError(
                "Unsupported SPICE state layout for table "
                f"{table.name}: expected columns {expected_columns}, found {actual_columns}. "
                "Delete and regenerate this state root."
            )
