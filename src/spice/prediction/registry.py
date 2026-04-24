"""Direct prediction-family dispatch for the fixed in-repo families."""

from __future__ import annotations

from ..core.errors import ConfigResolutionError
from .contracts import CompiledPredictionContract

_KNOWN_PREDICTION_FAMILY_IDS = frozenset(
    {"candidate_offset_selection", "min_block_fee_multitask"}
)


def validate_prediction_family_id(family_id: str) -> str:
    if family_id not in _KNOWN_PREDICTION_FAMILY_IDS:
        known = ", ".join(sorted(_KNOWN_PREDICTION_FAMILY_IDS))
        raise ConfigResolutionError(
            f"Unknown prediction.family_id: {family_id}. Known values: {known}"
        )
    return family_id


def compile_prediction_contract(
    *,
    prediction_id: str,
    family_id: str,
) -> CompiledPredictionContract:
    validate_prediction_family_id(family_id)
    if family_id == "candidate_offset_selection":
        from .families.candidate_offset_selection import compile_prediction_family

        return compile_prediction_family(prediction_id)
    assert family_id == "min_block_fee_multitask"
    from .families.min_block_fee_multitask import compile_prediction_family

    return compile_prediction_family(prediction_id)
