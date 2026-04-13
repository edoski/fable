from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import torch

from spice.config import coerce_problem_spec
from spice.core.reporting import NullReporter
from spice.features import FeatureSelection, build_feature_table
from spice.prediction import MetricSet
from spice.prediction.families.candidate_slate_current.batch import CandidateSlateTargetBatch
from spice.prediction.families.candidate_slate_current.loss import compute_objective_loss
from spice.prediction.families.candidate_slate_current.metrics import (
    best_epoch,
    objective_value,
)
from spice.prediction.families.candidate_slate_current.replay import run_replay
from spice.temporal.contracts import resolve_feature_contract


def _build_test_store():
    selection = FeatureSelection(
        feature_set_id="test_timestamp_native",
        feature_family_id="time_native",
        feature_names=("seconds_since_previous_block", "elapsed_seconds"),
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
    feature_table = build_feature_table(blocks, selection=selection)
    contract = resolve_feature_contract(
        problem=coerce_problem_spec(
            {
                "id": "test_timestamp_native",
                "lookback_seconds": 10,
                "sample_count": 4,
                "max_supported_delay_seconds": 12,
                "compiler": {"id": "timestamp_native"},
            }
        ),
        selection=selection,
    )
    store, _ = contract.build_capability_store(feature_table)
    return store


def test_objective_loss_prefers_cheaper_candidates_and_ignores_masked_slots() -> None:
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

    bad_loss = compute_objective_loss(logits_bad, targets)
    good_loss = compute_objective_loss(logits_good, targets)

    assert good_loss.item() < bad_loss.item()


def test_prediction_selection_follows_validation_profit() -> None:
    history = [
        MetricSet(
            values={
                "objective_loss": 0.90,
                "exact_optimum_hit_rate": 0.30,
                "cost_over_optimum": 0.25,
                "profit_over_baseline": 0.04,
            }
        ),
        MetricSet(
            values={
                "objective_loss": 1.20,
                "exact_optimum_hit_rate": 0.20,
                "cost_over_optimum": 0.18,
                "profit_over_baseline": 0.11,
            }
        ),
        MetricSet(
            values={
                "objective_loss": 0.70,
                "exact_optimum_hit_rate": 0.40,
                "cost_over_optimum": 0.20,
                "profit_over_baseline": 0.08,
            }
        ),
    ]

    assert best_epoch(history) == 2
    assert objective_value(history[1]) == pytest.approx(0.11)


def test_replay_summary_uses_event_weighted_totals() -> None:
    store = _build_test_store()
    sample_indices = np.arange(store.n_samples, dtype=np.int64)
    predictions = [0] * store.n_samples

    summary = run_replay(
        store,
        predictions,
        sample_indices=sample_indices,
        window_seconds=8,
        arrival_rate_per_second=0.3,
        repetitions=4,
        seed=2026,
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
