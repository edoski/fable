from __future__ import annotations

import pytest

from spice.config import (
    ArtifactVariant,
    TunedParameterSet,
    TunedPredictionParams,
    TunedProblemParams,
    TunedTrainingParams,
    WorkflowTask,
)
from spice.core.errors import ConfigResolutionError
from spice.core.reporting import NullReporter
from spice.modeling.families.lstm import LstmTunedModelParams
from spice.modeling.tuning import apply_study_best_params, apply_tuned_parameters
from spice.storage.layout import resolve_workflow_paths
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune


def _problem_prediction_tuning_override(
    model_workflow_override,
    *,
    prediction: str = "icdcs_2026",
) -> dict[str, object]:
    override = model_workflow_override(sample_count=20, lookback_seconds=240)
    override["prediction"] = prediction
    override["tuning_space"] = {
        "problem": {
            "lookback_seconds": [120, 240],
        },
        "prediction": {
            "classification_loss_weight": [0.5, 1.0],
            "regression_loss_weight": [0.25, 0.5],
        },
        "model": {
            "id": "lstm",
        },
    }
    return override


def _wide_tuning_override(model_workflow_override) -> dict[str, object]:
    override = _problem_prediction_tuning_override(model_workflow_override)
    override["tuning_space"]["training"] = {
        "learning_rate": [0.0001, 0.0003],
        "weight_decay": [0.0, 0.01],
        "batch_size": [8, 16],
    }
    override["tuning_space"]["model"] = {
        "id": "lstm",
        "hidden_size": [64, 128],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2],
    }
    return override


def test_tune_config_supports_problem_and_prediction_tuning_groups(
    tmp_path,
    load_test_tune_config,
    model_workflow_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=_problem_prediction_tuning_override(model_workflow_override),
    )

    assert config.tuning_space.problem is not None
    assert config.tuning_space.problem.lookback_seconds == [120, 240]
    assert config.tuning_space.prediction is not None
    assert config.tuning_space.prediction.classification_loss_weight == [0.5, 1.0]
    assert config.tuning_space.prediction.regression_loss_weight == [0.25, 0.5]


def test_tune_config_rejects_prediction_tuning_for_unsupported_family(
    tmp_path,
    load_test_tune_config,
    model_workflow_override,
) -> None:
    with pytest.raises(
        ConfigResolutionError,
        match="tuning_space\\.prediction fields are unsupported",
    ):
        load_test_tune_config(
            tmp_path,
            override=_problem_prediction_tuning_override(
                model_workflow_override,
                prediction="candidate_offset_selection",
            ),
        )


def test_apply_tuned_parameters_updates_problem_and_prediction_groups(
    tmp_path,
    load_test_train_config,
    model_workflow_override,
) -> None:
    config = load_test_train_config(
        tmp_path,
        override=model_workflow_override(),
    )

    tuned = apply_tuned_parameters(
        config,
        TunedParameterSet(
            problem=TunedProblemParams(lookback_seconds=240),
            prediction=TunedPredictionParams(
                classification_loss_weight=2.0,
                regression_loss_weight=0.25,
            ),
        ),
    )

    assert tuned.problem.lookback_seconds == 240
    assert tuned.prediction.family.classification_loss_weight == 2.0
    assert tuned.prediction.family.regression_loss_weight == 0.25


def test_apply_tuned_parameters_updates_training_and_model_groups(
    tmp_path,
    load_test_train_config,
    model_workflow_override,
) -> None:
    config = load_test_train_config(
        tmp_path,
        override=model_workflow_override(),
    )

    tuned = apply_tuned_parameters(
        config,
        TunedParameterSet(
            training=TunedTrainingParams(
                learning_rate=0.001,
                weight_decay=0.01,
                batch_size=16,
            ),
            model=LstmTunedModelParams(
                hidden_size=128,
                num_layers=1,
                dropout=0.1,
            ),
        ),
    )

    assert tuned.training.learning_rate == 0.001
    assert tuned.training.weight_decay == 0.01
    assert tuned.training.batch_size == 16
    assert tuned.model.hidden_size == 128
    assert tuned.model.num_layers == 1
    assert tuned.model.dropout == 0.1


def test_tune_then_train_tuned_round_trips_problem_and_prediction_params(
    tmp_path,
    load_test_tune_config,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    tune_config = load_test_tune_config(
        tmp_path,
        override=_problem_prediction_tuning_override(model_workflow_override),
    )
    seed_history_dataset(tune_config)

    run_tune(tune_config, reporter=NullReporter())

    tuned_override = model_workflow_override(sample_count=20, lookback_seconds=240)
    tuned_override["prediction"] = tune_config.prediction.id
    tuned_override["artifact"] = {"variant": ArtifactVariant.TUNED.value}
    tuned_override["study"] = tune_config.study.name
    tuned_train_config = load_test_train_config(tmp_path, override=tuned_override)

    run_train(tuned_train_config, reporter=NullReporter())
    resolved = apply_study_best_params(tuned_train_config)

    assert resolved.problem.lookback_seconds in {120, 240}
    assert resolved.prediction.family.classification_loss_weight in {0.5, 1.0}
    assert resolved.prediction.family.regression_loss_weight in {0.25, 0.5}
    assert resolve_workflow_paths(resolved).artifact_state_db.is_file()


def test_tune_then_train_tuned_round_trips_training_and_model_params(
    tmp_path,
    load_test_tune_config,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    tune_config = load_test_tune_config(
        tmp_path,
        override=_wide_tuning_override(model_workflow_override),
    )
    seed_history_dataset(tune_config)

    run_tune(tune_config, reporter=NullReporter())

    tuned_override = model_workflow_override(sample_count=20, lookback_seconds=240)
    tuned_override["prediction"] = tune_config.prediction.id
    tuned_override["artifact"] = {"variant": ArtifactVariant.TUNED.value}
    tuned_override["study"] = tune_config.study.name
    tuned_train_config = load_test_train_config(tmp_path, override=tuned_override)

    run_train(tuned_train_config, reporter=NullReporter())
    resolved = apply_study_best_params(tuned_train_config)

    assert resolved.training.batch_size in {8, 16}
    assert resolved.model.hidden_size in {64, 128}
    assert resolved.model.num_layers in {1, 2}
    assert resolved.model.dropout in {0.1, 0.2}
    assert resolve_workflow_paths(resolved).artifact_state_db.is_file()


def test_extensive_presets_resolve_cleanly(
    tmp_path,
    load_workflow_config,
) -> None:
    practical = load_workflow_config(
        WorkflowTask.TUNE,
        workspace=tmp_path,
        preset="icdcs_2026_economic_extensive",
    )
    paper = load_workflow_config(
        WorkflowTask.TUNE,
        workspace=tmp_path,
        preset="icdcs_2026_paper_truth_extensive",
    )
    transfer = load_workflow_config(
        WorkflowTask.TRAIN,
        workspace=tmp_path,
        preset="icdcs_2026_paper_truth_practical_seed",
    )

    assert practical.objective.id == "validation_evaluator_metric"
    assert practical.tuning_space.training is not None
    assert practical.tuning_space.training.batch_size == [128, 256, 512, 1024, 2048]
    assert paper.tuning_space.training is not None
    assert paper.tuning_space.training.batch_size == [64, 128, 256, 512, 1024]
    assert transfer.problem.lookback_seconds == 900
    assert transfer.training.batch_size == 512
