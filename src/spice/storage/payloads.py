# pyright: strict

"""Shared payload codecs and SQLite payload stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection
from sqlalchemy.schema import Table

from ..core.errors import ConfigResolutionError, StateLayoutError

T = TypeVar("T")
PayloadModelT = TypeVar("PayloadModelT", bound="PayloadModel")


class PayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
        raise StateLayoutError(f"{label}.payload must be a mapping")
    return {str(key): value for key, value in cast(Mapping[object, object], payload).items()}


def decode_payload(label: str, decode: Callable[[], T]) -> T:
    try:
        return decode()
    except StateLayoutError:
        raise
    except (ConfigResolutionError, KeyError, ValueError, TypeError) as exc:
        raise StateLayoutError(f"Invalid {label} payload: {exc}") from exc


def model_payload(model: BaseModel, *, label: str) -> dict[str, object]:
    return _model_dump_mapping(model.model_dump(mode="json"), label=label)


def _model_dump_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise StateLayoutError(f"{label} must serialize to a mapping payload")
    return cast(dict[str, object], payload)


def decode_payload_model(
    label: str,
    model_type: type[PayloadModelT],
    payload: dict[str, object],
    decode: Callable[[PayloadModelT], T],
) -> T:
    return decode_payload(
        label,
        lambda: decode(model_type.model_validate(payload, strict=True)),
    )


def payload_model_codec(
    label: str,
    model_type: type[PayloadModelT],
    from_value: Callable[[T], PayloadModelT],
    to_value: Callable[[PayloadModelT], T],
) -> PayloadCodec[T]:
    return PayloadCodec(
        encode=lambda value: model_payload(from_value(value), label=label),
        decode=lambda payload: decode_payload_model(label, model_type, payload, to_value),
    )


def type_adapter_payload(
    adapter: TypeAdapter[Any],
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    payload = adapter.dump_python(value, mode="json")
    if not isinstance(payload, dict):
        raise StateLayoutError(f"{label} must serialize to a mapping payload")
    return cast(dict[str, object], payload)


def type_adapter_value(
    adapter: TypeAdapter[T],
    payload: object,
    *,
    label: str,
    strict: bool = False,
) -> T:
    return decode_payload(label, lambda: adapter.validate_python(payload, strict=strict))


def string_payload(value: object, *, label: str) -> str:
    if isinstance(value, str):
        return value
    raise StateLayoutError(f"{label} must be a string")


def int_payload(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise StateLayoutError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    raise StateLayoutError(f"{label} must be an integer")


def optional_int_payload(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return int_payload(value, label=label)


def sequence_payload(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StateLayoutError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def int_list_payload(value: object, *, label: str) -> list[int]:
    return [int_payload(item, label=label) for item in sequence_payload(value, label=label)]
