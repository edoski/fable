from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import torch

from spice.config import coerce_feature_set_config, coerce_problem_spec
from spice.core.reporting import NullReporter
from spice.evaluation import coerce_evaluator_config, compile_evaluator_contract
from spice.features import compile_feature_contract
from spice.modeling.evaluation import run_prediction_evaluation
from spice.prediction import MetricSet
from spice.prediction.families.candidate_offset_selection.batch import CandidateSlateTargetBatch
from spice.prediction.families.candidate_offset_selection.loss import compute_selection_loss
from spice.prediction.families.candidate_offset_selection.metrics import best_epoch
from spice.temporal import (
    coerce_realization_policy_config,
    compile_realization_policy_contract,
)
from spice.temporal.contracts import compile_problem_contract


def _build_test_store():
    feature_contract = compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": "test_timestamp_native",
                "family": {"id": "time_native"},
                "outputs": ["seconds_since_previous_block", "elapsed_seconds"],
            }
        )
    )
    blocks = pl.DataFrame(
        {
            "block_number": np.arange(100, 107, dtype=np.int64),
            "timestamp": np.array([0, 5, 11, 18, 27, 29, 40], dtype=np.int64),
            "base_fee_per_gas": np.full(7, 1_000_000_000, dtype=np.int64),
            "gas_used": np.full(7, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(7, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(7, dtype=np.int64),
        }
    )
    feature_table = feature_contract.build_table(blocks)
    contract = compile_problem_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_native",
                "lookback_seconds": 10,
                "sample_count": 4,
                "max_delay_seconds": 12,
                "compiler": {"id": "timestamp_native"},
                "realization_policy": {"id": "strict_deadline_miss"},
            }
        ),
        feature_contract=feature_contract,
    )
    store, _ = contract.build_capability_store(feature_table)
    return store


def _realization_policy():
    return compile_realization_policy_contract(
        coerce_realization_policy_config({"id": "strict_deadline_miss"})
    )


def test_selection_loss_prefers_cheaper_candidates_and_ignores_masked_slots() -> None:
    candidate_log_fees = torch.tensor(
        [[math.log(10.0), math.log(5.0), math.log(1000.0)]],
        dtype=torch.float32,
    )
    candidate_mask = torch.tensor([[True, True, False]])
    logits_bad = torch.tensor([[4.0, -4.0, 100.0]], dtype=torch.float32)
    logits_good = torch.tensor([[-4.0, 4.0, 100.0]], dtype=torch.float32)

    targets = CandidateSlateTargetBatch(
        candidate_log_fees=candidate_log_fees,
        candidate_mask=candidate_mask,
    )

    bad_loss = compute_selection_loss(logits_bad, targets)
    good_loss = compute_selection_loss(logits_good, targets)

    assert good_loss.item() < bad_loss.item()


def test_prediction_selection_follows_validation_profit() -> None:
    history = [
        MetricSet(
            values={
                "total_loss": 0.90,
                "exact_optimum_hit_rate": 0.30,
                "cost_over_optimum": 0.25,
                "profit_over_baseline": 0.04,
            }
        ),
        MetricSet(
            values={
                "total_loss": 1.20,
                "exact_optimum_hit_rate": 0.20,
                "cost_over_optimum": 0.18,
                "profit_over_baseline": 0.11,
            }
        ),
        MetricSet(
            values={
                "total_loss": 0.70,
                "exact_optimum_hit_rate": 0.40,
                "cost_over_optimum": 0.20,
                "profit_over_baseline": 0.08,
            }
        ),
    ]

    assert best_epoch(history) == 2
    assert history[1].require("profit_over_baseline") == pytest.approx(0.11)


def test_poisson_replay_summary_uses_event_weighted_totals() -> None:
    store = _build_test_store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    predictions = [0] * store.n_samples

    evaluator = compile_evaluator_contract(
        coerce_evaluator_config(
            {
                "id": "poisson_replay",
                "window_seconds": 8,
                "arrival_rate_per_second": 0.3,
                "repetitions": 4,
                "seed": 2026,
            }
        )
    )
    summary = run_prediction_evaluation(
        evaluator,
        store,
        _realization_policy(),
        predictions,
        sample_indices=sample_indices,
        reporter=NullReporter(),
    )

    realized_fee_sum = sum(run.metrics["realized_fee_sum"] for run in summary.runs)
    baseline_fee_sum = sum(run.metrics["baseline_fee_sum"] for run in summary.runs)
    optimum_fee_sum = sum(run.metrics["optimum_fee_sum"] for run in summary.runs)

    assert summary.metrics.require("profit_over_baseline") == pytest.approx(
        (baseline_fee_sum - realized_fee_sum) / baseline_fee_sum
    )
    assert summary.metrics.require("cost_over_optimum") == pytest.approx(
        (realized_fee_sum - optimum_fee_sum) / optimum_fee_sum
    )
    assert summary.window_metrics["profit_over_baseline"].mean == pytest.approx(
        np.mean([run.metrics["profit_over_baseline"] for run in summary.runs])
    )
