"""Prediction-family specs for the fixed in-repo families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.specs import lookup_local_spec
from .contracts import CompiledPredictionContract


@dataclass(frozen=True, slots=True)
class PredictionFamilySpec:
    compile_contract: Callable[[str], CompiledPredictionContract]


def _compile_min_block_fee_multitask(prediction_id: str) -> CompiledPredictionContract:
    from .families.min_block_fee_multitask import compile_prediction_family

    return compile_prediction_family(prediction_id)


_PREDICTION_FAMILY_SPECS: dict[str, PredictionFamilySpec] = {
    "min_block_fee_multitask": PredictionFamilySpec(
        compile_contract=_compile_min_block_fee_multitask,
    ),
}


def prediction_family_spec(family_id: str) -> PredictionFamilySpec:
    return lookup_local_spec(
        _PREDICTION_FAMILY_SPECS,
        family_id,
        "prediction.family_id",
    )


def validate_prediction_family_id(family_id: str) -> str:
    prediction_family_spec(family_id)
    return family_id


def compile_prediction_contract(
    *,
    prediction_id: str,
    family_id: str,
) -> CompiledPredictionContract:
    return prediction_family_spec(family_id).compile_contract(prediction_id)
