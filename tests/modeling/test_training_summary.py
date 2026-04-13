from __future__ import annotations

import pytest

from spice.core.reporting import NullReporter
from spice.corpus.io import load_block_frame
from spice.modeling.artifacts import load_training_artifact
from spice.modeling.pipeline import build_training_spec, prepare_training_dataset
from spice.modeling.training import evaluate_model
from spice.storage.artifact import list_training_epochs, load_training_summary
from spice.workflows.train import run as run_train


def test_training_summary_metrics_match_replayed_saved_artifact(
    tmp_path,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    config = load_test_train_config(tmp_path, override=model_workflow_override())
    seed_history_dataset(config)

    run_train(config, reporter=NullReporter())

    summary = load_training_summary(config.paths.artifact_state_db)
    assert summary is not None

    loaded_artifact = load_training_artifact(config.paths.artifact_root)
    spec = build_training_spec(config)
    prepared = prepare_training_dataset(load_block_frame(config.paths.history_dir), spec=spec)

    validation_metrics = evaluate_model(
        loaded_artifact.model,
        model_id=config.model.id,
        prediction_contract=spec.prediction_contract,
        store=prepared.store,
        sample_indices=prepared.split_indices.validation,
        training_config=config.training,
        reporter=NullReporter(),
    )
    test_metrics = evaluate_model(
        loaded_artifact.model,
        model_id=config.model.id,
        prediction_contract=spec.prediction_contract,
        store=prepared.store,
        sample_indices=prepared.split_indices.test,
        training_config=config.training,
        reporter=NullReporter(),
    )

    assert summary.objective_id == "profit_over_baseline"
    assert summary.representation_id == "sequence_inputs"
    assert summary.batch_planner_id == "signature_bucketed"
    assert summary.family_execution_id == "dense_recurrent_last_valid"
    assert summary.best_validation_metrics.require("objective_loss") == pytest.approx(
        validation_metrics.require("objective_loss")
    )
    assert summary.best_validation_metrics.require("exact_optimum_hit_rate") == pytest.approx(
        validation_metrics.require("exact_optimum_hit_rate")
    )
    assert summary.best_validation_metrics.require("profit_over_baseline") == pytest.approx(
        validation_metrics.require("profit_over_baseline")
    )
    assert summary.test_metrics.require("objective_loss") == pytest.approx(
        test_metrics.require("objective_loss")
    )
    assert summary.test_metrics.require("exact_optimum_hit_rate") == pytest.approx(
        test_metrics.require("exact_optimum_hit_rate")
    )
    assert summary.test_metrics.require("profit_over_baseline") == pytest.approx(
        test_metrics.require("profit_over_baseline")
    )

    epochs = list_training_epochs(config.paths.artifact_state_db)
    assert epochs
    assert epochs[summary.best_epoch - 1].epoch == summary.best_epoch
    assert epochs[summary.best_epoch - 1].validation_metrics.require(
        "profit_over_baseline"
    ) == pytest.approx(
        summary.best_validation_metrics.require("profit_over_baseline")
    )
