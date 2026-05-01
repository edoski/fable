"""Feature catalog lookup."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.specs import lookup_local_spec
from .core import FeatureCatalog, validate_feature_names


@dataclass(frozen=True, slots=True)
class FeaturesEntry:
    catalog: FeatureCatalog
    allowed_outputs: tuple[str, ...]


def _features_entries() -> dict[str, FeaturesEntry]:
    from .sets.core_fee_dynamics import (
        CORE_FEE_DYNAMICS,
        CORE_FEE_DYNAMICS_ELAPSED_POSITION_OUTPUTS,
        CORE_FEE_DYNAMICS_OUTPUTS,
        CORE_FEE_DYNAMICS_PRIORITY_FEE_OUTPUTS,
        CORE_FEE_DYNAMICS_UNSAFE_OUTPUTS,
    )

    return {
        "core_fee_dynamics": FeaturesEntry(
            catalog=CORE_FEE_DYNAMICS,
            allowed_outputs=CORE_FEE_DYNAMICS_OUTPUTS,
        ),
        "core_fee_dynamics_elapsed_position": FeaturesEntry(
            catalog=CORE_FEE_DYNAMICS,
            allowed_outputs=CORE_FEE_DYNAMICS_ELAPSED_POSITION_OUTPUTS,
        ),
        "core_fee_dynamics_unsafe": FeaturesEntry(
            catalog=CORE_FEE_DYNAMICS,
            allowed_outputs=CORE_FEE_DYNAMICS_UNSAFE_OUTPUTS,
        ),
        "core_fee_dynamics_with_priority_fee": FeaturesEntry(
            catalog=CORE_FEE_DYNAMICS,
            allowed_outputs=CORE_FEE_DYNAMICS_PRIORITY_FEE_OUTPUTS,
        ),
    }


def feature_entry(features_id: str) -> FeaturesEntry:
    return lookup_local_spec(_features_entries(), features_id, "features.id")


def validate_feature_selection(
    features_id: str,
    feature_names: tuple[str, ...],
) -> None:
    entry = feature_entry(features_id)
    validate_feature_names(
        features_id,
        feature_names,
        known_feature_names=entry.allowed_outputs,
    )
