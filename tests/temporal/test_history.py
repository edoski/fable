from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import UUID4, TypeAdapter
from torch.utils.data import DataLoader

from fable.config import (
    BlockWindow,
    CorpusDefinition,
    CorpusRequest,
    ExperimentSemantics,
)
from fable.corpus import BlockFrame, Corpus, FinalizedAnchor
from fable.temporal import prepare_fit_history, prepare_historical_window

_CORPUS_ID = TypeAdapter(UUID4).validate_python("11111111-1111-4111-8111-111111111111")
_BASE_FEES = np.array(
    [11, 12, 10, 4, 9, 4, 8, 3, 5, 6, 10, 6, 2, 2, 7, 6, 5, 4, 4, 9],
    dtype=np.int64,
)


def _corpus(first_block: int = 10, last_block: int = 29) -> Corpus:
    blocks = np.arange(10, 30, dtype=np.int64)
    frame = pl.DataFrame(
        {
            "block_number": blocks,
            "timestamp": blocks * 12,
            "chain_id": np.ones(blocks.size, dtype=np.int64),
            "base_fee_per_gas": _BASE_FEES,
            "gas_used": 35 + np.arange(blocks.size, dtype=np.int64),
            "gas_limit": np.full(blocks.size, 100, dtype=np.int64),
            "tx_count": 20 + np.arange(blocks.size, dtype=np.int64),
            "effective_priority_fee_per_gas_p50": 2 + np.arange(blocks.size, dtype=np.int64),
            "effective_priority_fee_per_gas_p90": 4 + np.arange(blocks.size, dtype=np.int64),
        },
    ).filter(pl.col("block_number").is_between(first_block, last_block))
    request = CorpusRequest(
        corpus_id=_CORPUS_ID,
        definition=CorpusDefinition(
            chain_id=1,
            first_block=first_block,
            last_block=last_block,
        ),
    )
    return Corpus(
        request=request,
        finalized_anchor=FinalizedAnchor(
            block_number=last_block,
            block_hash="a" * 64,
        ),
        blocks=BlockFrame(frame, request.definition),
    )


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(
            first_parent_block=12,
            last_parent_block=15,
        ),
        validation_window=BlockWindow(
            first_parent_block=20,
            last_parent_block=21,
        ),
        context_blocks=3,
        horizon_blocks=3,
        ordered_features=("log_base_fee_per_gas", "gas_utilization"),
    )


def test_fit_history_preserves_geometry_statistics_and_collation() -> None:
    preparation = prepare_fit_history(_corpus(), _experiment())

    support_fees = _BASE_FEES[:6].astype(np.float64)
    support_raw = np.column_stack(
        (np.log(support_fees), (35 + np.arange(6, dtype=np.float64)) / 100.0)
    )
    np.testing.assert_allclose(
        preparation.feature_state.means,
        support_raw.mean(axis=0),
    )
    np.testing.assert_allclose(
        preparation.feature_state.standard_deviations,
        support_raw.std(axis=0, ddof=0),
    )

    training_minima = np.array([4, 4, 3, 3], dtype=np.int64)
    logged_minima = np.log(training_minima.astype(np.float64))
    logged_mean = logged_minima.mean()
    logged_standard_deviation = logged_minima.std(ddof=0)
    assert preparation.target_state.mean == pytest.approx(logged_mean)
    assert preparation.target_state.standard_deviation == pytest.approx(logged_standard_deviation)

    first = preparation.training[0]
    assert set(first) == {"inputs", "label", "target", "base_fees", "origin_block"}
    assert first["inputs"].shape == (3, 2)
    assert first["inputs"].dtype == torch.float32
    assert first["label"].shape == ()
    assert first["label"].dtype == torch.int64
    assert first["target"].shape == ()
    assert first["target"].dtype == torch.float32
    assert first["base_fees"].shape == (3,)
    assert first["base_fees"].dtype == torch.int64
    assert first["origin_block"].shape == ()
    assert first["origin_block"].dtype == torch.int64
    assert all(value.device.type == "cpu" for value in first.values())
    expected_inputs = np.ascontiguousarray(
        (support_raw[:3] - support_raw.mean(axis=0)) / support_raw.std(axis=0, ddof=0),
        dtype=np.float32,
    )
    torch.testing.assert_close(first["inputs"], torch.from_numpy(expected_inputs))
    assert int(first["label"]) == 0
    assert float(first["target"]) == pytest.approx(
        (np.log(4.0) - logged_mean) / logged_standard_deviation
    )
    assert first["base_fees"].tolist() == [4, 9, 4]

    validation = preparation.validation[0]
    assert [
        int(preparation.validation[index]["origin_block"])
        for index in range(len(preparation.validation))
    ] == [20, 21]
    assert validation["base_fees"].tolist() == [6, 2, 2]
    assert int(validation["label"]) == 1
    assert float(validation["target"]) == pytest.approx(
        (np.log(2.0) - logged_mean) / logged_standard_deviation
    )

    testing = prepare_historical_window(
        _corpus(),
        _experiment(),
        BlockWindow(
            first_parent_block=25,
            last_parent_block=26,
        ),
        feature_state=preparation.feature_state,
        target_state=preparation.target_state,
    )
    assert [int(testing[index]["origin_block"]) for index in range(len(testing))] == [25, 26]
    assert [int(testing[index]["label"]) for index in range(len(testing))] == [1, 0]

    batches = list(DataLoader(preparation.training, batch_size=3))
    assert [batch["origin_block"].tolist() for batch in batches] == [[12, 13, 14], [15]]
    assert batches[0]["inputs"].shape == (3, 3, 2)
    assert batches[0]["base_fees"].shape == (3, 3)


def test_interval_feature_uses_a_real_predecessor_outside_the_context() -> None:
    blocks = np.arange(9, 30, dtype=np.int64)
    timestamps = np.cumsum(np.arange(10, 31, dtype=np.int64))
    frame = pl.DataFrame(
        {
            "block_number": blocks,
            "timestamp": timestamps,
            "chain_id": np.ones(blocks.size, dtype=np.int64),
            "base_fee_per_gas": np.concatenate((np.array([13]), _BASE_FEES)),
            "gas_used": 34 + np.arange(blocks.size, dtype=np.int64),
            "gas_limit": np.full(blocks.size, 100, dtype=np.int64),
            "tx_count": 19 + np.arange(blocks.size, dtype=np.int64),
            "effective_priority_fee_per_gas_p50": np.arange(blocks.size, dtype=np.int64),
            "effective_priority_fee_per_gas_p90": 2 * np.arange(blocks.size, dtype=np.int64),
        }
    )
    request = CorpusRequest(
        corpus_id=_CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=9, last_block=29),
    )
    corpus = Corpus(
        request=request,
        finalized_anchor=FinalizedAnchor(block_number=29, block_hash="a" * 64),
        blocks=BlockFrame(frame, request.definition),
    )
    experiment = _experiment().model_copy(
        update={
            "ordered_features": (
                "log1p_effective_priority_fee_per_gas_p50",
                "block_interval_seconds",
            )
        }
    )
    preparation = prepare_fit_history(corpus, experiment)
    raw = np.column_stack((np.log1p(np.arange(1, 7)), np.arange(11, 17))).astype(np.float64)
    expected = ((raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)).astype(np.float32)

    torch.testing.assert_close(preparation.training[0]["inputs"], torch.from_numpy(expected[:3]))

    without_predecessor = corpus.model_copy(
        update={
            "request": request.model_copy(
                update={"definition": CorpusDefinition(chain_id=1, first_block=10, last_block=29)}
            ),
            "blocks": corpus.blocks.select_range(10, 29),
        }
    )
    with pytest.raises(ValueError, match="within the BlockFrame definition"):
        prepare_fit_history(without_predecessor, experiment)


@pytest.mark.parametrize("corpus", (_corpus(first_block=11), _corpus(last_block=23)))
def test_fit_history_requires_complete_context_and_outcome_support(corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="within the BlockFrame definition"):
        prepare_fit_history(corpus, _experiment())


def test_testing_window_must_follow_complete_validation_outcomes() -> None:
    corpus = _corpus()
    experiment = _experiment()
    preparation = prepare_fit_history(corpus, experiment)

    with pytest.raises(ValueError, match="testing window must follow complete validation outcomes"):
        prepare_historical_window(
            corpus,
            experiment,
            BlockWindow(
                first_parent_block=24,
                last_parent_block=25,
            ),
            feature_state=preparation.feature_state,
            target_state=preparation.target_state,
        )
