"""Feature catalog lookup."""

from __future__ import annotations

from ..core.specs import lookup_local_spec
from .core import FeatureCatalog, validate_feature_names


def _feature_catalogs() -> dict[str, FeatureCatalog]:
    from .sets.core_fee_dynamics.elapsed_position import (
        CORE_FEE_DYNAMICS_ELAPSED_POSITION,
    )
    from .sets.core_fee_dynamics.safe import (
        CORE_FEE_DYNAMICS,
    )
    from .sets.core_fee_dynamics.with_priority_fee import (
        CORE_FEE_DYNAMICS_PRIORITY_FEE,
    )

    return {
        "core_fee_dynamics": CORE_FEE_DYNAMICS,
        "core_fee_dynamics_elapsed_position": CORE_FEE_DYNAMICS_ELAPSED_POSITION,
        "core_fee_dynamics_with_priority_fee": CORE_FEE_DYNAMICS_PRIORITY_FEE,
    }


def feature_entry(features_id: str) -> FeatureCatalog:
    return lookup_local_spec(_feature_catalogs(), features_id, "features.id")


def validate_feature_selection(
    features_id: str,
    feature_names: tuple[str, ...],
) -> None:
    validate_feature_names(
        features_id,
        feature_names,
        known_feature_names=feature_entry(features_id).allowed_outputs,
    )
