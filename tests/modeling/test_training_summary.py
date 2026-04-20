from __future__ import annotations

import pytest

from spice.core.reporting import NullReporter
from spice.corpus.io import load_block_frame
from spice.modeling.artifacts import load_training_artifact
from spice.modeling.pipeline import build_training_spec, prepare_training_dataset
from spice.modeling.representations import SEQUENCE_INPUT_REPRESENTATION_ID
from spice.modeling.training import evaluate_model
from spice.storage.artifact import list_training_epochs, load_training_summary
from spice.storage.layout import resolve_workflow_paths
from spice.workflows.train import run as run_train


def test_training_summary_metrics_match_replayed_saved_artifact(
    tmp_path,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    config = load_test_train_config(tmp_path, override=model_workflow_override())
    paths = resolve_workflow_paths(config)
    seed_history_dataset(config)

    run_train(config, reporter=NullReporter())

    summary = load_training_summary(paths.artifact_state_db)
    assert summary is not None

    loaded_artifact = load_training_artifact(paths.artifact_root)
    spec = build_training_spec(config)
    prepared = prepare_training_dataset(load_block_frame(paths.history_dir), spec=spec)
    prediction_training_state = spec.prediction_contract.fit_training_state(
        prepared.store,
        prepared.split_indices.train,
        realization_policy=prepared.realization_policy,
    )

    validation_metrics = evaluate_model(
        loaded_artifact.model,
        model_config=spec.model,
        prediction_contract=spec.prediction_contract,
        realization_policy=prepared.realization_policy,
        representation_contract=loaded_artifact.representation_contract,
        store=prepared.store,
        sample_indices=prepared.split_indices.validation,
        prediction_training_state=prediction_training_state,
        training_config=config.training,
        reporter=NullReporter(),
    )
    test_metrics = evaluate_model(
        loaded_artifact.model,
        model_config=spec.model,
        prediction_contract=spec.prediction_contract,
        realization_policy=prepared.realization_policy,
        representation_contract=loaded_artifact.representation_contract,
        store=prepared.store,
        sample_indices=prepared.split_indices.test,
        prediction_training_state=prediction_training_state,
        training_config=config.training,
        reporter=NullReporter(),
    )

    assert summary.manifest.prediction_id == config.prediction.id
    assert summary.manifest.representation_id == SEQUENCE_INPUT_REPRESENTATION_ID
    assert summary.runtime.batch_planner_id == "signature_bucketed"
    assert summary.runtime.best_validation_metrics.values == pytest.approx(
        validation_metrics.values
    )
    assert summary.runtime.test_metrics.values == pytest.approx(test_metrics.values)

    epochs = list_training_epochs(paths.artifact_state_db)
    assert epochs
    assert epochs[summary.runtime.best_epoch - 1].epoch == summary.runtime.best_epoch
    primary_metric_id = spec.prediction_contract.primary_metric_id
    best_epoch_metrics = epochs[summary.runtime.best_epoch - 1].validation_metrics
    assert best_epoch_metrics.require(primary_metric_id) == pytest.approx(
        summary.runtime.best_validation_metrics.require(primary_metric_id)
    )
