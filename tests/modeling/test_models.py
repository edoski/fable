from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.transformer import TransformerModelConfig
from spice.modeling.families.transformer_lstm import TransformerLstmModelConfig
from spice.modeling.models import (
    LSTMBaseline,
    TransformerBaseline,
    TransformerLSTMBaseline,
)
from spice.modeling.representations import build_sequence_input_batch
from spice.prediction.families.min_block_fee_multitask.outputs import (
    OFFSET_LOGITS_HEAD_ID,
    build_output_spec,
)
from spice.temporal import (
    coerce_execution_policy_config,
    compile_execution_policy_contract,
)
from spice.temporal.problem_store import CompiledProblemStore


def _test_store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.array(
            [
                [-1.0, 0.0, 0.1],
                [-2.0, 0.1, 0.2],
                [0.5, 0.2, 0.3],
                [1.5, 0.3, 0.4],
                [-0.2, 0.4, 0.5],
                [2.0, 0.5, 0.6],
                [-1.1, 0.6, 0.7],
                [0.3, 0.7, 0.8],
                [1.2, 0.8, 0.9],
                [-0.7, 0.9, 1.0],
            ],
            dtype=np.float32,
        ),
        log_base_fees=np.array(
            [0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.4, 0.12, 0.22, 0.18],
            dtype=np.float32,
        ),
        timestamps=np.array([0, 5, 11, 19, 28, 40, 55, 71, 88, 106], dtype=np.int64),
        anchor_rows=np.array([2, 4, 5, 7], dtype=np.int64),
        context_start_rows=np.array([0, 1, 0, 4], dtype=np.int64),
        candidate_start_rows=np.array([3, 5, 6, 8], dtype=np.int64),
        candidate_end_rows=np.array([5, 8, 7, 10], dtype=np.int64),
        max_candidate_slots=3,
    )


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def _assert_model_ignores_padded_timesteps(model, inputs, input_mask, max_candidate_slots) -> None:
    model.eval()
    with torch.no_grad():
        baseline = model(inputs, input_mask).head(OFFSET_LOGITS_HEAD_ID)
        perturbed = inputs.clone()
        perturbed[~input_mask.bool()] = 1000.0
        changed = model(perturbed, input_mask).head(OFFSET_LOGITS_HEAD_ID)

    assert tuple(baseline.shape) == (inputs.shape[0], max_candidate_slots)
    assert torch.allclose(baseline, changed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        (
            LSTMBaseline,
            LstmModelConfig(
                input_projection_dim=8,
                hidden_size=8,
                num_layers=2,
                dropout=0.0,
                head_hidden_dim=4,
            ),
        ),
        (
            TransformerBaseline,
            TransformerModelConfig(
                dropout=0.0,
                d_model=4,
                nhead=2,
                transformer_layers=1,
                feedforward_dim=16,
                head_hidden_dim=4,
            ),
        ),
        (
            TransformerLSTMBaseline,
            TransformerLstmModelConfig(
                hidden_size=8,
                num_layers=2,
                dropout=0.0,
                d_model=4,
                nhead=2,
                transformer_layers=1,
                feedforward_dim=16,
                head_hidden_dim=4,
            ),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:"
    "The PyTorch API of nested tensors is in prototype stage and will change "
    "in the near future.*:UserWarning"
)
def test_sequence_models_emit_offset_logits_and_ignore_padding(model_type, config) -> None:
    torch.manual_seed(5)
    store = _test_store()
    sample_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    execution_policy = _execution_policy()
    batch = build_sequence_input_batch(
        store,
        sample_indices=sample_indices,
        action_mask=execution_policy.prepare_action_space(store, sample_indices).action_mask,
    )
    model = model_type(
        n_features=batch.inputs.shape[-1],
        output_spec=build_output_spec(store.max_candidate_slots),
        config=config,
    )

    _assert_model_ignores_padded_timesteps(
        model,
        batch.inputs,
        batch.input_mask,
        store.max_candidate_slots,
    )
