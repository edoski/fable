from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.modeling.forward_runtime import (
    run_planned_model_input_forward,
    run_planned_prediction_forward,
)
from spice.modeling.representations import RepresentationRuntimeContext
from spice.modeling.runtime_planning import ModelingRuntimePlan


def _runtime_plan() -> ModelingRuntimePlan:
    return ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        representation_runtime_context=RepresentationRuntimeContext(
            batch_size=2,
            available_host_memory_bytes=1024,
        ),
        deterministic=None,
        seed=3,
    )


def test_planned_forward_rejects_empty_sample_indices() -> None:
    with pytest.raises(ValueError, match="sample_indices must be non-empty"):
        run_planned_model_input_forward(
            object(),
            store=object(),
            sample_indices=np.array([], dtype=np.int64),
            representation_contract=object(),
            execution_policy=object(),
            runtime_plan=_runtime_plan(),
            on_outputs=lambda _batch, _outputs: None,
        )

    with pytest.raises(ValueError, match="sample_indices must be non-empty"):
        run_planned_prediction_forward(
            object(),
            store=object(),
            sample_indices=np.array([], dtype=np.int64),
            representation_contract=object(),
            prediction_contract=object(),
            execution_policy=object(),
            runtime_plan=_runtime_plan(),
            on_outputs=lambda _batch, _outputs: None,
        )
