"""Closed dispatch for supported prediction families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from ..core.errors import ConfigResolutionError
from .base import PredictionFamilyConfig
from .contracts import CompiledPredictionContract

PredictionFamilyConfigT = TypeVar("PredictionFamilyConfigT", bound=PredictionFamilyConfig)


@dataclass(frozen=True, slots=True)
class PredictionFamilySpec(Generic[PredictionFamilyConfigT]):
    id: str
    config_type: type[PredictionFamilyConfigT]
    compile: Callable[[str, PredictionFamilyConfigT], CompiledPredictionContract]


_KNOWN_PREDICTION_FAMILIES = ("candidate_offset_selection", "min_block_fee_multitask")


def prediction_family_spec(family_id: str) -> PredictionFamilySpec[PredictionFamilyConfig]:
    if family_id == "candidate_offset_selection":
        from .families.candidate_offset_selection import (
            CandidateOffsetSelectionFamilyConfig,
            compile_prediction_family as compile_candidate_offset_selection,
        )

        return PredictionFamilySpec(
            id="candidate_offset_selection",
            config_type=CandidateOffsetSelectionFamilyConfig,
            compile=compile_candidate_offset_selection,
        )
    if family_id == "min_block_fee_multitask":
        from .families.min_block_fee_multitask import (
            MinBlockFeeMultitaskFamilyConfig,
            compile_prediction_family as compile_min_block_fee_multitask,
        )

        return PredictionFamilySpec(
            id="min_block_fee_multitask",
            config_type=MinBlockFeeMultitaskFamilyConfig,
            compile=compile_min_block_fee_multitask,
        )
    known = ", ".join(_KNOWN_PREDICTION_FAMILIES)
    raise ConfigResolutionError(
        f"Unknown prediction.family.id: {family_id}. Known values: {known}"
    )


def coerce_prediction_family_config(
    raw_config: Mapping[str, object] | PredictionFamilyConfig,
) -> PredictionFamilyConfig:
    if isinstance(raw_config, PredictionFamilyConfig):
        family_id = raw_config.id
        payload = raw_config.model_dump(mode="json")
    elif isinstance(raw_config, Mapping):
        if "id" not in raw_config:
            raise ConfigResolutionError("prediction.family.id is required")
        family_id = str(raw_config["id"])
        payload = dict(raw_config)
    else:
        raise ConfigResolutionError("prediction.family must be a mapping")
    return prediction_family_spec(family_id).config_type.model_validate(payload)


def compile_prediction_contract(
    *,
    prediction_id: str,
    family_config: PredictionFamilyConfig,
) -> CompiledPredictionContract:
    spec = prediction_family_spec(family_config.id)
    return spec.compile(prediction_id, cast(Any, family_config))
