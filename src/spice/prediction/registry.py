"""Prediction-family specs for the fixed in-repo families."""

from __future__ import annotations

from ..core.specs import lookup_local_spec
from .contracts import CompiledPredictionContract

_SUPPORTED_PREDICTION_FAMILY_IDS = frozenset({"min_block_fee_multitask"})


def _compile_min_block_fee_multitask(prediction_id: str) -> CompiledPredictionContract:
    from .families.min_block_fee_multitask import compile_prediction_family

    return compile_prediction_family(prediction_id)


def _require_prediction_family_id(family_id: str) -> str:
    return lookup_local_spec(
        {family_id: family_id for family_id in _SUPPORTED_PREDICTION_FAMILY_IDS},
        family_id,
        "prediction.family_id",
    )


def validate_prediction_family_id(family_id: str) -> str:
    return _require_prediction_family_id(family_id)


def compile_prediction_contract(
    *,
    prediction_id: str,
    family_id: str,
) -> CompiledPredictionContract:
    family_id = _require_prediction_family_id(family_id)
    if family_id == "min_block_fee_multitask":
        return _compile_min_block_fee_multitask(prediction_id)
    raise AssertionError(f"unhandled prediction.family_id: {family_id}")
