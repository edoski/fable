from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
import torch

from fable.config import LossDefinition
from fable.min_block_fee import (
    ClassificationLossState,
    MinBlockFeeOutput,
    TargetState,
    decode_action,
    fit_classification_loss_state,
    fit_target_state,
    min_block_fee_loss,
    standardize_target,
    target_natural_log,
)


def _loss_definition(
    weighting: Literal["unweighted", "corrected_inverse_frequency"],
    *,
    regression_threshold: float,
    classification_scale: float,
    regression_scale: float,
) -> LossDefinition:
    return LossDefinition(
        classification_algorithm="cross_entropy",
        classification_weighting=weighting,
        regression_algorithm="smooth_l1",
        regression_threshold=regression_threshold,
        classification_scale=classification_scale,
        regression_scale=regression_scale,
    )


def _valid_loss_definition() -> LossDefinition:
    return _loss_definition(
        "unweighted",
        regression_threshold=1.0,
        classification_scale=1.0,
        regression_scale=1.0,
    )


def test_target_and_loss_match_hand_derived_fixture() -> None:
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
    torch.testing.assert_close(
        target_natural_log(torch.from_numpy(target_z), target_state),
        torch.log(torch.tensor(raw_minima, dtype=torch.float64)),
    )

    labels_array = np.array([0, 0, 1, 2], dtype=np.int64)
    unweighted_definition = _loss_definition(
        "unweighted",
        regression_threshold=0.5,
        classification_scale=2.0,
        regression_scale=3.0,
    )
    corrected_definition = _loss_definition(
        "corrected_inverse_frequency",
        regression_threshold=2.0,
        classification_scale=0.5,
        regression_scale=2.0,
    )
    assert (
        fit_classification_loss_state(
            labels_array,
            horizon_blocks=3,
            loss_definition=unweighted_definition,
        )
        is None
    )
    classification_state = fit_classification_loss_state(
        labels_array,
        horizon_blocks=3,
        loss_definition=corrected_definition,
    )
    assert classification_state == ClassificationLossState(class_support=(2, 1, 1))

    labels = torch.from_numpy(labels_array)
    targets = torch.from_numpy(target_z)
    logits = torch.zeros((4, 3), dtype=torch.float32, requires_grad=True)
    predictions = (targets + torch.tensor([0.0, 0.5, 1.0, 2.0])).detach().requires_grad_()
    output = MinBlockFeeOutput(action_logits=logits, minimum_fee_z=predictions)

    unweighted = min_block_fee_loss(
        output,
        label=labels,
        target=targets,
        classification_state=classification_state,
        loss_definition=unweighted_definition,
    )
    corrected = min_block_fee_loss(
        output,
        label=labels,
        target=targets,
        classification_state=classification_state,
        loss_definition=corrected_definition,
    )
    log_three = math.log(3.0)
    expected_unweighted_classification = torch.full((4,), 2.0 * log_three)
    expected_unweighted_regression = torch.tensor([0.0, 0.75, 2.25, 5.25])
    expected_unweighted = expected_unweighted_classification + expected_unweighted_regression
    expected_corrected_classification = torch.tensor([1.0, 1.0, 2.0, 2.0]) * (log_three / 3.0)
    expected_corrected_regression = torch.tensor([0.0, 0.125, 0.5, 2.0])

    torch.testing.assert_close(
        unweighted.classification_by_origin,
        expected_unweighted_classification,
    )
    torch.testing.assert_close(unweighted.regression_by_origin, expected_unweighted_regression)
    torch.testing.assert_close(unweighted.total_by_origin, expected_unweighted)
    torch.testing.assert_close(unweighted.mean_total, expected_unweighted.sum() / 4.0)
    torch.testing.assert_close(
        corrected.classification_by_origin,
        expected_corrected_classification,
    )
    torch.testing.assert_close(
        corrected.regression_by_origin,
        expected_corrected_regression,
    )
    torch.testing.assert_close(
        corrected.total_by_origin,
        expected_corrected_classification + expected_corrected_regression,
    )
    assert corrected.mean_total.requires_grad
    assert not corrected.total_by_origin.requires_grad
    assert not corrected.classification_by_origin.requires_grad
    assert not corrected.regression_by_origin.requires_grad

    first = min_block_fee_loss(
        MinBlockFeeOutput(logits[:3], predictions[:3]),
        label=labels[:3],
        target=targets[:3],
        classification_state=classification_state,
        loss_definition=corrected_definition,
    )
    tail = min_block_fee_loss(
        MinBlockFeeOutput(logits[3:], predictions[3:]),
        label=labels[3:],
        target=targets[3:],
        classification_state=classification_state,
        loss_definition=corrected_definition,
    )
    complete_total = torch.cat((first.total_by_origin, tail.total_by_origin))
    torch.testing.assert_close(complete_total, corrected.total_by_origin)
    torch.testing.assert_close(complete_total.sum() / 4.0, corrected.mean_total.detach())

    corrected.mean_total.backward()
    assert logits.grad is not None
    assert predictions.grad is not None


def _valid_output() -> MinBlockFeeOutput:
    return MinBlockFeeOutput(
        action_logits=torch.zeros((2, 2), dtype=torch.float32),
        minimum_fee_z=torch.zeros(2, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda: TargetState(mean=math.nan, standard_deviation=1.0),
            id="target-state-finite",
        ),
        pytest.param(
            lambda: TargetState(mean=0.0, standard_deviation=0.0),
            id="target-state-positive-standard-deviation",
        ),
        pytest.param(
            lambda: ClassificationLossState(class_support=()),
            id="classification-state-nonempty",
        ),
        pytest.param(
            lambda: ClassificationLossState(class_support=(1, 0)),
            id="classification-state-positive",
        ),
        pytest.param(
            lambda: fit_target_state(np.array([1, 2], dtype="int32")),
            id="target-int64",
        ),
        pytest.param(
            lambda: fit_target_state(np.array([1, 0], dtype=np.int64)),
            id="target-positive",
        ),
        pytest.param(
            lambda: fit_target_state(np.array([2, 2], dtype=np.int64)),
            id="target-nonconstant",
        ),
        pytest.param(
            lambda: standardize_target(
                np.array([1, -1], dtype=np.int64),
                TargetState(mean=0.0, standard_deviation=1.0),
            ),
            id="standardize-positive",
        ),
        pytest.param(
            lambda: fit_classification_loss_state(
                np.array([0, 0], dtype=np.int64),
                horizon_blocks=2,
                loss_definition=_loss_definition(
                    "corrected_inverse_frequency",
                    regression_threshold=1.0,
                    classification_scale=1.0,
                    regression_scale=1.0,
                ),
            ),
            id="corrected-inverse-frequency-full-support",
        ),
        pytest.param(
            lambda: min_block_fee_loss(
                MinBlockFeeOutput(
                    action_logits=torch.tensor([[math.inf, 0.0], [0.0, 0.0]]),
                    minimum_fee_z=torch.zeros(2),
                ),
                label=torch.tensor([0, 1]),
                target=torch.zeros(2),
                classification_state=None,
                loss_definition=_valid_loss_definition(),
            ),
            id="finite-output",
        ),
        pytest.param(
            lambda: min_block_fee_loss(
                _valid_output(),
                label=torch.tensor([0.0, 1.0]),
                target=torch.zeros(2),
                classification_state=None,
                loss_definition=_valid_loss_definition(),
            ),
            id="label-int64",
        ),
        pytest.param(
            lambda: min_block_fee_loss(
                _valid_output(),
                label=torch.tensor([0, 2]),
                target=torch.zeros(2),
                classification_state=None,
                loss_definition=_valid_loss_definition(),
            ),
            id="label-range",
        ),
        pytest.param(
            lambda: min_block_fee_loss(
                _valid_output(),
                label=torch.tensor([0, 1]),
                target=torch.tensor([0.0, math.nan]),
                classification_state=None,
                loss_definition=_valid_loss_definition(),
            ),
            id="finite-target",
        ),
        pytest.param(
            lambda: decode_action(
                MinBlockFeeOutput(
                    action_logits=torch.zeros((2, 2)),
                    minimum_fee_z=torch.zeros(1),
                )
            ),
            id="aligned-output-heads",
        ),
    ],
)
def test_owned_invalid_inputs_are_rejected(operation: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        operation()


def test_decode_uses_native_first_index_argmax_and_ignores_auxiliary_values() -> None:
    logits = torch.tensor([[-1.0, 3.0, 2.0], [4.0, 0.0, -1.0]])
    low_auxiliary = MinBlockFeeOutput(logits, torch.tensor([-100.0, -100.0]))
    high_auxiliary = MinBlockFeeOutput(logits, torch.tensor([100.0, 100.0]))
    expected = torch.tensor([1, 0], dtype=torch.int64)

    assert torch.equal(decode_action(low_auxiliary), expected)
    assert torch.equal(decode_action(high_auxiliary), expected)
