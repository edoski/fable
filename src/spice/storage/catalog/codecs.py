"""Strict catalog record codecs for remote storage transfer."""

from __future__ import annotations

import json

from ...core.errors import SpiceOperatorError
from ..engine import RootKind
from .records import CatalogRecord
from .root_kind_specs import spec_for_record, spec_for_root_kind


def encode_remote_catalog_record(record: CatalogRecord) -> str:
    spec = spec_for_record(record)
    return json.dumps(
        {
            "root_kind": spec.root_kind.value,
            "record": spec.to_record_payload(record),
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
    return spec_for_root_kind(root_kind).from_record_payload(record_payload)
