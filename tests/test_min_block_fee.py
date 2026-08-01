from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fable.min_block_fee import (
    MinBlockFeeOutput,
    decode_action,
    fit_target_state,
    min_block_fee_loss,
    standardize_target,
)


def test_target_and_native_loss_match_hand_derived_fixture() -> None:
    raw_minima = np.array([1, 4, 16, 64], dtype=np.int64)
    target_state = fit_target_state(raw_minima)
    log_four = math.log(4.0)

    assert target_state.mean == pytest.approx(1.5 * log_four)
    assert target_state.standard_deviation == pytest.approx(math.sqrt(5.0) * log_four / 2.0)

    target_z = standardize_target(raw_minima, target_state)
    expected_z = np.array(
        [-3.0 / math.sqrt(5.0), -1.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 3.0 / math.sqrt(5.0)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(target_z, expected_z)
    assert target_z.dtype == np.float32
    assert target_z.flags.c_contiguous

    labels = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    targets = torch.from_numpy(target_z)
    logits = torch.zeros((4, 3), dtype=torch.float32, requires_grad=True)
    predictions = (targets + torch.tensor([0.0, 0.5, 1.0, 2.0])).detach().requires_grad_()
    output = MinBlockFeeOutput(action_logits=logits, minimum_fee_z=predictions)

    loss_by_origin = min_block_fee_loss(output, label=labels, target=targets)
    log_three = math.log(3.0)
    expected_total = torch.tensor([log_three, log_three + 0.125, log_three + 0.5, log_three + 1.5])

    torch.testing.assert_close(loss_by_origin, expected_total)
    assert loss_by_origin.requires_grad

    loss_by_origin.mean().backward()
    assert logits.grad is not None
    assert predictions.grad is not None


def test_target_state_requires_nonconstant_minima() -> None:
    with pytest.raises(ValueError):
        fit_target_state(np.array([2, 2], dtype=np.int64))


def test_decode_uses_native_first_index_argmax_and_ignores_auxiliary_values() -> None:
    output = MinBlockFeeOutput(
        torch.tensor([[3.0, 3.0, 2.0], [4.0, 4.0, -1.0]]), torch.tensor([math.nan])
    )

    assert torch.equal(decode_action(output), torch.tensor([0, 0], dtype=torch.int64))


def test_decode_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError):
        decode_action(MinBlockFeeOutput(torch.tensor([[math.nan, 0.0]]), torch.zeros(1)))
