from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.prediction.decoded_offsets import DecodedOffsets
from spice.prediction.decoding import ActionSpaceDecodeContext
from spice.prediction.families.min_block_fee_multitask.outputs import masked_offset_argmax


def test_decoded_offsets_allocation_write_select_and_equality() -> None:
    offsets = DecodedOffsets.allocate(4)
    offsets.write(
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([5, 3], dtype=torch.int64),
    )

    assert offsets.decoded_result_id == "candidate_offsets"
    assert len(offsets) == 4
    assert offsets == [3, 0, 5, 0]
    assert not offsets == [3.2, 0, 5, 0]
    assert offsets == torch.tensor([3, 0, 5, 0], dtype=torch.int64)
    np.testing.assert_array_equal(
        offsets.select(np.array([2, 0], dtype=np.int64)),
        np.array([5, 3], dtype=np.int64),
    )


def test_decoded_offsets_preserve_coercion_and_shape_validation() -> None:
    with pytest.raises(ValueError, match="sample_count must be non-negative"):
        DecodedOffsets.allocate(-1)
    with pytest.raises(ValueError, match="decoded_offsets must be one-dimensional"):
        DecodedOffsets(torch.zeros((1, 1), dtype=torch.int64))

    offsets = DecodedOffsets.allocate(2)
    with pytest.raises(ValueError, match="matching shape"):
        offsets.write(torch.tensor([0, 1]), torch.tensor([1]))


def test_masked_offset_argmax_preserves_mask_and_tie_semantics() -> None:
    logits = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [9.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(
        masked_offset_argmax(logits, mask),
        torch.tensor([0, 2], dtype=torch.int64),
    )


def test_decode_context_validation_preserves_action_mask_rules() -> None:
    context = ActionSpaceDecodeContext(
        sample_positions=np.array([1, 0], dtype=np.int64),
        action_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
    )

    assert torch.equal(context.sample_positions, torch.tensor([1, 0], dtype=torch.int64))
    assert context.action_mask.dtype == torch.bool

    with pytest.raises(ValueError, match="matching rows"):
        ActionSpaceDecodeContext(
            sample_positions=torch.tensor([0, 1]),
            action_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="allow at least one action"):
        ActionSpaceDecodeContext(
            sample_positions=torch.tensor([0]),
            action_mask=torch.zeros((1, 2), dtype=torch.bool),
        )
