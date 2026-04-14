from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.transformer_lstm import TransformerLstmModelConfig
from spice.modeling.models import (
    LSTMBaseline,
    TransformerLSTMBaseline,
    packed_lstm_last_state_reference,
    take_last_valid,
)
from spice.modeling.representations import build_sequence_input_batch
from spice.prediction.families.candidate_offset_selection.outputs import (
    CANDIDATE_LOGITS_HEAD_ID,
    build_output_spec,
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
        candidate_end_rows=np.array([5, 8, 7, 10], dtype=np.int64),
        max_candidate_slots=3,
    )


def test_lstm_baseline_matches_packed_reference() -> None:
    torch.manual_seed(7)
    store = _test_store()
    batch = build_sequence_input_batch(store, sample_indices=np.array([0, 1, 2, 3]))
    model = LSTMBaseline(
        n_features=batch.inputs.shape[-1],
        output_spec=build_output_spec(store.max_candidate_slots),
        config=LstmModelConfig(
            input_projection_dim=8,
            hidden_size=8,
            num_layers=2,
            dropout=0.0,
            head_hidden_dim=4,
        ),
    )
    model.eval()

    with torch.no_grad():
        projected = model.input_projection(batch.inputs)
        reference_last = packed_lstm_last_state_reference(
            model.backbone,
            projected,
            batch.input_mask,
        )
        recurrent, _ = model.backbone(projected)
        dense_last = take_last_valid(recurrent, batch.input_mask)

        assert torch.allclose(dense_last, reference_last, atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            model.output_head(dense_last).head(CANDIDATE_LOGITS_HEAD_ID),
            model(batch.inputs, batch.input_mask).head(CANDIDATE_LOGITS_HEAD_ID),
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.filterwarnings(
    "ignore:"
    "The PyTorch API of nested tensors is in prototype stage and will change "
    "in the near future.*:UserWarning"
)
def test_transformer_lstm_baseline_matches_packed_reference() -> None:
    torch.manual_seed(11)
    store = _test_store()
    batch = build_sequence_input_batch(store, sample_indices=np.array([0, 1, 2, 3]))
    model = TransformerLSTMBaseline(
        n_features=batch.inputs.shape[-1],
        output_spec=build_output_spec(store.max_candidate_slots),
        config=TransformerLstmModelConfig(
            hidden_size=8,
            num_layers=2,
            dropout=0.0,
            d_model=4,
            nhead=2,
            transformer_layers=1,
            feedforward_dim=16,
            head_hidden_dim=4,
        ),
    )
    model.eval()

    with torch.no_grad():
        projected = model.input_projection(batch.inputs)
        encoded = model.encoder(
            model.position_encoding(projected),
            src_key_padding_mask=~batch.input_mask.bool(),
        )
        reference_last = packed_lstm_last_state_reference(
            model.lstm,
            encoded,
            batch.input_mask,
        )
        recurrent, _ = model.lstm(encoded)
        dense_last = take_last_valid(recurrent, batch.input_mask)

        assert torch.allclose(dense_last, reference_last, atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            model.output_head(dense_last).head(CANDIDATE_LOGITS_HEAD_ID),
            model(batch.inputs, batch.input_mask).head(CANDIDATE_LOGITS_HEAD_ID),
            atol=1e-6,
            rtol=1e-6,
        )
