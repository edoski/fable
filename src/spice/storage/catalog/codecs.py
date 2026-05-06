"""Strict catalog record codecs for remote storage transfer."""

from __future__ import annotations

import json
from typing import Any, cast

from ...core.errors import SpiceOperatorError
from ..engine import RootKind
from .records import CatalogRecord
from .registry import spec_for_record, spec_for_root_kind


def encode_remote_catalog_record(record: CatalogRecord) -> str:
    spec = spec_for_record(record)
    return json.dumps(
        {
            "root_kind": spec.root_kind.value,
            "record": _record_to_payload(record),
        }
    )


def decode_remote_catalog_record(
    payload: str,
    *,
    expected_root_kind: RootKind | None = None,
) -> CatalogRecord:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SpiceOperatorError("remote catalog record payload is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise SpiceOperatorError("remote catalog record payload must be a mapping")
    if set(raw) != {"root_kind", "record"}:
        raise SpiceOperatorError("remote catalog record payload must contain root_kind and record")
    root_kind_value = raw["root_kind"]
    if not isinstance(root_kind_value, str):
        raise SpiceOperatorError("remote catalog record root_kind must be a string")
    try:
        root_kind = RootKind(root_kind_value)
    except ValueError as exc:
        message = f"unsupported remote catalog root kind: {root_kind_value}"
        raise SpiceOperatorError(message) from exc
    if expected_root_kind is not None and root_kind is not expected_root_kind:
        raise SpiceOperatorError(
            f"remote catalog root kind mismatch: expected {expected_root_kind}, got {root_kind}"
        )
    record_payload = raw["record"]
    if not isinstance(record_payload, dict):
        raise SpiceOperatorError("remote catalog record must be a mapping")
    return _record_from_payload(root_kind, record_payload)


def _record_to_payload(record: CatalogRecord) -> dict[str, object | None]:
    spec = spec_for_record(record)
    concrete = spec.require_record(record)
    return {field_name: getattr(concrete, field_name) for field_name in spec.field_names}


def _record_from_payload(
    root_kind: RootKind,
    payload: dict[str, object | None],
) -> CatalogRecord:
    spec = spec_for_root_kind(root_kind)
    fields_set = set(spec.field_names)
    payload_keys = set(payload)
    missing = sorted(fields_set - payload_keys)
    extra = sorted(payload_keys - fields_set)
    if missing:
        raise SpiceOperatorError(
            f"remote catalog record is missing fields: {', '.join(missing)}"
        )
    if extra:
        raise SpiceOperatorError(f"remote catalog record has extra fields: {', '.join(extra)}")
    values: dict[str, object | None] = {}
    for name in spec.field_names:
        value = payload[name]
        if value is None:
            if name not in spec.nullable_fields:
                raise SpiceOperatorError(f"remote catalog record field {name} cannot be null")
            values[name] = None
            continue
        values[name] = _require_string(name, value)
    return spec.record_type(**cast(Any, values))


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise SpiceOperatorError(f"remote catalog record field {name} must be a string")
    return value
