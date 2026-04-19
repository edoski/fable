# pyright: strict

"""Shared payload codecs and SQLite payload stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection
from sqlalchemy.schema import Table

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PayloadCodec(Generic[T]):
    encode: Callable[[T], dict[str, object]]
    decode: Callable[[dict[str, object]], T]


@dataclass(frozen=True, slots=True)
class SingletonPayloadStore(Generic[T]):
    table: Table
    codec: PayloadCodec[T]

    def upsert(self, conn: Connection, value: T) -> None:
        payload = self.codec.encode(value)
        statement = sqlite_insert(self.table).values(singleton=1, payload=payload)
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=[self.table.c.singleton],
                set_={"payload": payload},
            )
        )

    def load(self, conn: Connection) -> T | None:
        row = conn.execute(select(self.table.c.payload)).mappings().first()
        if row is None:
            return None
        return self.codec.decode(mapping_payload(row["payload"], label=self.table.name))


@dataclass(frozen=True, slots=True)
class SequencePayloadStore(Generic[T]):
    table: Table
    codec: PayloadCodec[T]

    def append(self, conn: Connection, value: T, **extra_columns: object) -> None:
        conn.execute(
            self.table.insert().values(
                payload=self.codec.encode(value),
                **extra_columns,
            )
        )

    def replace(self, conn: Connection, rows: Sequence[dict[str, object]]) -> None:
        conn.execute(delete(self.table))
        if rows:
            conn.execute(self.table.insert(), list(rows))

    def list(self, conn: Connection, *, order_by: Any) -> list[T]:
        rows = conn.execute(select(self.table.c.payload).order_by(order_by)).mappings().all()
        values: list[T] = []
        for row in rows:
            values.append(self.codec.decode(mapping_payload(row["payload"], label=self.table.name)))
        return values


def mapping_payload(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label}.payload must be a mapping")
    return {str(key): value for key, value in cast(Mapping[object, object], payload).items()}
