from __future__ import annotations

from spice.objectives import ObjectiveConfig, ObjectiveDirection, coerce_objective_config


def test_objective_coercer_preserves_config_identity() -> None:
    config = ObjectiveConfig(
        id="validation",
        metric_id="total_loss",
        direction=ObjectiveDirection.MINIMIZE,
    )

    assert coerce_objective_config(config) is config
