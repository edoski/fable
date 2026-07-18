# pyright: strict

"""Shared payload codecs and SQLite payload stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection
from sqlalchemy.schema import Table

from ..core.errors import ConfigResolutionError, StateLayoutError

T = TypeVar("T")
PayloadRecordT = TypeVar("PayloadRecordT", bound="PayloadRecord")
PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)


class PayloadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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


def mapping_model_payload(
    model: BaseModel,
    *,
    label: str,
    exclude_none: bool = False,
) -> dict[str, object]:
    payload = cast(object, model.model_dump(mode="json", exclude_none=exclude_none))
    if not isinstance(payload, dict):
        raise StateLayoutError(f"{label} must serialize to a mapping payload")
    return cast(dict[str, object], payload)


def decode_payload_record(
    label: str,
    model_type: type[PayloadRecordT],
    payload: dict[str, object],
    decode: Callable[[PayloadRecordT], T],
) -> T:
    return decode_payload(
        label,
        lambda: decode(model_type.model_validate(payload, strict=True)),
    )


def payload_record_codec(
    label: str,
    model_type: type[PayloadRecordT],
    from_value: Callable[[T], PayloadRecordT],
    to_value: Callable[[PayloadRecordT], T],
    *,
    exclude_none: bool = False,
) -> PayloadCodec[T]:
    return PayloadCodec(
        encode=lambda value: mapping_model_payload(
            from_value(value),
            label=label,
            exclude_none=exclude_none,
        ),
        decode=lambda payload: decode_payload_record(label, model_type, payload, to_value),
    )


def pydantic_model_codec(
    label: str,
    model_type: type[PydanticModelT],
    *,
    exclude_none: bool = False,
) -> PayloadCodec[PydanticModelT]:
    return PayloadCodec(
        encode=lambda value: mapping_model_payload(
            value,
            label=label,
            exclude_none=exclude_none,
        ),
        decode=lambda payload: decode_payload(
            label,
            lambda: model_type.model_validate(payload, strict=True),
        ),
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
