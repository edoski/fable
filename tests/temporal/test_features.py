from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from fable.config import CorpusDefinition, FeatureName
from fable.corpus import BlockFrame
from fable.temporal import (
    FeatureState,
    fit_feature_state,
    transform_feature_rows,
)


def _blocks(
    *,
    base_fees: list[int],
    gas_used: list[int],
    gas_limits: list[int],
    tx_counts: list[int],
    timestamps: list[int],
    priority_fees: list[int],
    chain_id: int = 1,
) -> BlockFrame:
    count = len(base_fees)
    frame = pl.DataFrame(
        {
            "block_number": range(count),
            "timestamp": timestamps,
            "chain_id": [chain_id] * count,
            "base_fee_per_gas": base_fees,
            "gas_used": gas_used,
            "gas_limit": gas_limits,
            "tx_count": tx_counts,
            "effective_priority_fee_per_gas_p50": priority_fees,
        }
    )
    return BlockFrame(
        frame,
        CorpusDefinition(chain_id=chain_id, first_block=0, last_block=count - 1),
    )


def test_requested_feature_formulas_fit_in_order_and_transform_held_out_rows() -> None:
    forming_support = _blocks(
        base_fees=[1_000, 2_000, 3_000, 8_000_000_000_000_000_000],
        gas_used=[500, 1_200, 1_200, 4],
        gas_limits=[1_000, 2_000, 3_000, 4],
        tx_counts=[0, 1, 2, 3],
        timestamps=[0, 6 * 3_600, 12 * 3_600, 18 * 3_600],
        priority_fees=[0, 1, 2, 3],
    )
    forming_order = (
        "hour_cos",
        "log_exact_forming_base_fee_per_gas",
        "gas_utilization",
        "log1p_tx_count",
        "log_base_fee_per_gas",
        "hour_sin",
        "log_gas_limit",
    )
    raw = np.column_stack(
        (
            np.cos([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]),
            np.log([1_000, 2_050, 2_925, 9_000_000_000_000_000_000]),
            [0.5, 0.6, 0.4, 1.0],
            np.log1p([0, 1, 2, 3]),
            np.log([1_000, 2_000, 3_000, 8_000_000_000_000_000_000]),
            np.sin([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]),
            np.log([1_000, 2_000, 3_000, 4]),
        )
    ).astype(np.float64)
    forming_state = fit_feature_state(forming_support, ordered_features=forming_order)
    np.testing.assert_allclose(forming_state.means, raw.mean(axis=0))
    np.testing.assert_allclose(
        forming_state.standard_deviations,
        raw.std(axis=0, ddof=0),
    )

    transformed = transform_feature_rows(
        forming_support,
        ordered_features=forming_order,
        state=forming_state,
    )
    expected = ((raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)).astype(np.float32)
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous

    held_out = _blocks(
        base_fees=[1],
        gas_used=[101],
        gas_limits=[200],
        tx_counts=[4],
        timestamps=[3_600],
        priority_fees=[4],
    )
    held_out_raw = np.array(
        [
            [
                np.cos(np.pi / 12),
                np.log(2),
                101 / 200,
                np.log(5),
                0.0,
                np.sin(np.pi / 12),
                np.log(200),
            ]
        ],
        dtype=np.float64,
    )

    held_out_result = transform_feature_rows(
        held_out,
        ordered_features=forming_order,
        state=forming_state,
    )

    np.testing.assert_allclose(
        held_out_result,
        (
            (held_out_raw - np.asarray(forming_state.means))
            / np.asarray(forming_state.standard_deviations)
        ).astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fee_and_nonnegative_interval_features_use_their_exact_block_facts() -> None:
    blocks = _blocks(
        base_fees=[10, 20, 30, 40],
        gas_used=[1, 2, 3, 4],
        gas_limits=[10, 10, 10, 10],
        tx_counts=[1, 2, 3, 4],
        timestamps=[100, 100, 112, 130],
        priority_fees=[0, 9, 99, 999],
    )
    ordered_features = (
        "log1p_effective_priority_fee_per_gas_p50",
        "block_interval_seconds",
    )
    raw = np.column_stack((np.log1p([9, 99, 999]), [0, 12, 18])).astype(np.float64)

    state = fit_feature_state(blocks, ordered_features=ordered_features)
    transformed = transform_feature_rows(
        blocks,
        ordered_features=ordered_features,
        state=state,
    )

    np.testing.assert_allclose(state.means, raw.mean(axis=0))
    np.testing.assert_allclose(state.standard_deviations, raw.std(axis=0, ddof=0))
    np.testing.assert_allclose(
        transformed,
        ((raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)).astype(np.float32),
    )


def _valid_blocks() -> BlockFrame:
    return _blocks(
        base_fees=[10, 20],
        gas_used=[1, 3],
        gas_limits=[2, 4],
        tx_counts=[0, 1],
        timestamps=[0, 3_600],
        priority_fees=[0, 1],
    )


def _fit(
    blocks: BlockFrame,
    *,
    ordered_features: tuple[FeatureName, ...] = (
        "log_base_fee_per_gas",
        "gas_utilization",
    ),
) -> FeatureState:
    return fit_feature_state(blocks, ordered_features=ordered_features)


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        pytest.param(
            lambda: _fit(
                _blocks(
                    base_fees=[10, 20],
                    gas_used=[1, 2],
                    gas_limits=[2, 4],
                    tx_counts=[1, 2],
                    timestamps=[0, 1],
                    priority_fees=[1, 2],
                    chain_id=137,
                ),
                ordered_features=("log_exact_forming_base_fee_per_gas",),
            ),
            "Ethereum-only",
            id="forming-fee-chain",
        ),
        pytest.param(
            lambda: FeatureState(means=(0.0,), standard_deviations=(1.0, 2.0)),
            "equal widths",
            id="state-widths",
        ),
        pytest.param(
            lambda: transform_feature_rows(
                _valid_blocks(),
                ordered_features=("log_base_fee_per_gas", "gas_utilization"),
                state=FeatureState(
                    means=(0.0, 0.0),
                    standard_deviations=(1e-300, 1e-300),
                ),
            ),
            "finite float32",
            id="float32-overflow",
        ),
    ],
)
def test_feature_contract_rejections(
    operation: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        operation()


def test_fit_feature_state_rejects_constant_feature() -> None:
    blocks = _blocks(
        base_fees=[10, 10],
        gas_used=[1, 1],
        gas_limits=[2, 2],
        tx_counts=[0, 0],
        timestamps=[0, 0],
        priority_fees=[0, 0],
    )

    with pytest.raises(ValueError):
        _fit(blocks, ordered_features=("log_base_fee_per_gas",))
