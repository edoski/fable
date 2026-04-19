"""Tiny helpers for repo-owned closed dispatch seams."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, TypeVar

from .errors import ConfigResolutionError


class ConfigModelLike(Protocol):
    id: str

    def model_dump(self, *, mode: str) -> dict[str, object]: ...


ConfigT = TypeVar("ConfigT", bound=ConfigModelLike)


def validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


def mapping_id(payload: Mapping[str, object], *, field_name: str) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError(f"{field_name} is required")
    return value


def unknown_id_error(
    *,
    field_name: str,
    component_id: str,
    known_ids: Iterable[str],
) -> ConfigResolutionError:
    known = ", ".join(sorted(known_ids))
    return ConfigResolutionError(
        f"Unknown {field_name}: {component_id}. Known values: {known}"
    )


def config_payload_and_id(
    raw_config: Mapping[str, object] | ConfigT,
    *,
    config_type: type[ConfigT],
    field_name: str,
    mapping_label: str,
) -> tuple[dict[str, object], str]:
    if isinstance(raw_config, config_type):
        return raw_config.model_dump(mode="json"), raw_config.id
    if isinstance(raw_config, Mapping):
        payload = dict(raw_config)
        return payload, mapping_id(payload, field_name=field_name)
    raise ConfigResolutionError(f"{mapping_label} must be a mapping")
