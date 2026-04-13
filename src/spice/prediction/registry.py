"""Open registry for prediction-family specs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from .base import PredictionFamilyConfig
from .contracts import CompiledPredictionContract

PredictionFamilyConfigT = TypeVar("PredictionFamilyConfigT", bound=PredictionFamilyConfig)


@dataclass(frozen=True, slots=True)
class PredictionFamilySpec(Generic[PredictionFamilyConfigT]):
    id: str
    config_type: type[PredictionFamilyConfigT]
    compile: Callable[[str, PredictionFamilyConfigT], CompiledPredictionContract]


_PREDICTION_FAMILY_SPECS: dict[str, PredictionFamilySpec[Any]] = {}
_BUILTINS_LOADED = False


def register_prediction_family_spec(spec: PredictionFamilySpec[Any]) -> None:
    existing = _PREDICTION_FAMILY_SPECS.get(spec.id)
    if existing is not None:
        raise ValueError(f"Duplicate prediction family spec id: {spec.id}")
    _PREDICTION_FAMILY_SPECS[spec.id] = spec


def _ensure_builtin_prediction_families_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .families import candidate_slate_current  # noqa: F401

    _BUILTINS_LOADED = True


def prediction_family_spec(family_id: str) -> PredictionFamilySpec[Any]:
    _ensure_builtin_prediction_families_loaded()
    try:
        return _PREDICTION_FAMILY_SPECS[family_id]
    except KeyError as exc:
        known = ", ".join(sorted(_PREDICTION_FAMILY_SPECS)) or "<none>"
        raise ValueError(
            f"Unknown prediction.family.id: {family_id}. Known families: {known}"
        ) from exc


def coerce_prediction_family_config(
    raw_config: Mapping[str, object] | PredictionFamilyConfig,
) -> PredictionFamilyConfig:
    _ensure_builtin_prediction_families_loaded()
    if isinstance(raw_config, PredictionFamilyConfig):
        family_id = raw_config.id
        payload = raw_config.model_dump(mode="json")
    elif isinstance(raw_config, Mapping):
        if "id" not in raw_config:
            raise ValueError("prediction.family.id is required")
        family_id = str(raw_config["id"])
        payload = dict(raw_config)
    else:
        raise TypeError("prediction.family must be a mapping")
    return prediction_family_spec(family_id).config_type.model_validate(payload)


def compile_prediction_contract(
    *,
    prediction_id: str,
    family_config: PredictionFamilyConfig,
) -> CompiledPredictionContract:
    spec = prediction_family_spec(family_config.id)
    return spec.compile(prediction_id, cast(Any, family_config))
