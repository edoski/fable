from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from spice.core.console import NullReporter
from spice.data.datasets import build_temporal_store
from spice.data.io import load_block_frame
from spice.features import FeatureSelection, build_feature_table
from spice.modeling.artifacts import load_training_artifact
from spice.modeling.pipeline import prepare_training_dataset
from spice.modeling.torch_datasets import build_class_weights
from spice.modeling.training import evaluate_model
from spice.planning.geometry import DelayWindow
from spice.state.artifact import load_training_summary
from spice.workflows._shared import build_training_spec
from spice.workflows.train import run as run_train
from tests.support import load_test_train_config, model_workflow_override, seed_history_dataset


def test_temporal_store_uses_real_timestamps_for_context_and_candidates() -> None:
    selection = FeatureSelection(
        feature_set_id="test_timestamp_native",
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
    store = build_temporal_store(
        feature_table,
        window=DelayWindow(
            lookback_seconds=10,
            delay_seconds=12,
            feature_history_seconds=feature_table.feature_history_seconds,
        ),
    )

    np.testing.assert_array_equal(store.anchor_rows, np.array([2, 3, 4, 5], dtype=np.int64))
    np.testing.assert_array_equal(store.context_start_rows, np.array([1, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(
        store.candidate_end_rows - store.candidate_start_rows,
        np.array([1, 2, 1, 1], dtype=np.int64),
    )
    assert store.max_candidate_slots == 2


def test_training_summary_metrics_match_replayed_saved_artifact(tmp_path) -> None:
    config = load_test_train_config(tmp_path, override=model_workflow_override())
    seed_history_dataset(config)

    run_train(config, reporter=NullReporter())

    summary = load_training_summary(config.paths.artifact_state_db)
    assert summary is not None

    loaded_artifact = load_training_artifact(config.paths.artifact_root)
    spec = build_training_spec(config)
    prepared = prepare_training_dataset(load_block_frame(config.paths.history_dir), spec=spec)
    class_weights = build_class_weights(
        prepared.store.class_labels,
        prepared.split_indices.train,
        prepared.max_candidate_slots,
    )

    validation_metrics = evaluate_model(
        loaded_artifact.model,
        model_id=config.model.id,
        store=prepared.store,
        sample_indices=prepared.split_indices.validation,
        training_config=config.training,
        class_weights=class_weights,
        reporter=NullReporter(),
    )
    test_metrics = evaluate_model(
        loaded_artifact.model,
        model_id=config.model.id,
        store=prepared.store,
        sample_indices=prepared.split_indices.test,
        training_config=config.training,
        class_weights=class_weights,
        reporter=NullReporter(),
    )

    assert summary.best_validation_metrics.total_loss == pytest.approx(
        validation_metrics.total_loss
    )
    assert summary.best_validation_metrics.accuracy == pytest.approx(validation_metrics.accuracy)
    assert summary.best_validation_metrics.mean_profit_over_baseline == pytest.approx(
        validation_metrics.mean_profit_over_baseline
    )
    assert summary.test_metrics.total_loss == pytest.approx(test_metrics.total_loss)
    assert summary.test_metrics.accuracy == pytest.approx(test_metrics.accuracy)
    assert summary.test_metrics.mean_profit_over_baseline == pytest.approx(
        test_metrics.mean_profit_over_baseline
    )
