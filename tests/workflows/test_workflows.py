from __future__ import annotations

from typing import cast

import pytest

from spice.config import TrainConfig, WorkflowTask
from spice.core.errors import ConfigResolutionError, SpiceOperatorError
from spice.core.reporting import NullReporter
from spice.modeling.artifacts import load_training_artifact, validate_artifact_semantics
from spice.modeling.tuning import apply_study_best_params
from spice.storage.artifact import list_evaluation_runs, load_evaluation_summary
from spice.storage.catalog import list_artifact_records, list_study_records
from spice.storage.layout import resolve_workflow_paths
from spice.storage.study_optuna import load_study
from spice.workflows.evaluate import run as run_evaluate
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune


@pytest.mark.parametrize("compiler_id", ["timestamp_native", "estimated_block"])
def test_train_workflow_smoke(
    tmp_path,
    compiler_id,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    config = load_test_train_config(
        tmp_path,
        override=model_workflow_override(compiler_id=compiler_id),
    )
    paths = resolve_workflow_paths(config)
    seed_history_dataset(config)

    run_train(config, reporter=NullReporter())

    assert paths.artifact_state_db.is_file()
    assert (paths.artifact_root / "model.pt").is_file()
    artifacts = list_artifact_records(
        paths.catalog_db,
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        feature_set_id=config.feature_set.id,
        model_id=config.model.id,
        problem_id=config.problem.id,
        variant=config.artifact.variant.value,
    )
    assert len(artifacts) == 1
    assert artifacts[0].artifact_id == paths.artifact_id


def test_train_workflow_smoke_transformer_lstm_icdcs(
    tmp_path,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    override = model_workflow_override()
    override["model"] = "transformer_lstm"
    config = load_test_train_config(
        tmp_path,
        override=override,
    )
    paths = resolve_workflow_paths(config)
    seed_history_dataset(config)

    run_train(config, reporter=NullReporter())

    assert paths.artifact_state_db.is_file()
    assert (paths.artifact_root / "model.pt").is_file()


def test_train_configs_with_distinct_semantic_bundles_get_distinct_artifact_ids(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
) -> None:
    prod_config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=model_workflow_override(),
        ),
    )
    repro_config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=model_workflow_override(compiler_id="timestamp_native"),
        ),
    )

    assert resolve_workflow_paths(prod_config).artifact_id != resolve_workflow_paths(
        repro_config
    ).artifact_id
    assert resolve_workflow_paths(prod_config).artifact_root != resolve_workflow_paths(
        repro_config
    ).artifact_root
    assert prod_config.problem.compiler.id != repro_config.problem.compiler.id


def test_tune_then_train_tuned_smoke(
    tmp_path,
    load_test_train_config,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    paths = resolve_workflow_paths(config)
    seed_history_dataset(config)

    run_tune(config, reporter=NullReporter())

    assert paths.study_state_db.is_file()
    study = load_study(paths.study_state_db, study_name=config.study.name)
    assert len(study.trials) == config.tuning.trial_count
    studies = list_study_records(
        paths.catalog_db,
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        feature_set_id=config.feature_set.id,
        model_id=config.model.id,
        problem_id=config.problem.id,
        study_name=config.study.name,
    )
    assert len(studies) == 1
    assert studies[0].study_id == paths.study_id

    tuned_override = model_workflow_override()
    tuned_override["artifact"] = {"variant": "tuned"}
    tuned_override["study"] = config.study.name
    tuned_train_config = load_test_train_config(
        tmp_path,
        override=tuned_override,
    )
    run_train(tuned_train_config, reporter=NullReporter())
    resolved_tuned_config = apply_study_best_params(tuned_train_config)
    resolved_paths = resolve_workflow_paths(resolved_tuned_config)

    assert resolved_paths.artifact_state_db.is_file()
    assert (resolved_paths.artifact_root / "model.pt").is_file()


def test_tune_resume_same_study_tops_up_to_target(
    tmp_path,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    initial_config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    seed_history_dataset(initial_config)

    run_tune(initial_config, reporter=NullReporter())

    resumed_override = model_workflow_override() | tune_override()
    resumed_override["tuning"] = {
        "trial_count": 4,
        "timeout_seconds": None,
        "sampler_seed": 2026,
        "enable_pruning": False,
    }
    resumed_config = load_test_tune_config(tmp_path, override=resumed_override)

    run_tune(resumed_config, reporter=NullReporter())

    study = load_study(
        resolve_workflow_paths(resumed_config).study_state_db,
        study_name=resumed_config.study.name,
    )
    assert len(study.trials) == 4


def test_tune_rejects_lower_requested_trial_budget(
    tmp_path,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    seed_history_dataset(config)

    run_tune(config, reporter=NullReporter())

    lower_budget_override = model_workflow_override() | tune_override()
    lower_budget_override["tuning"] = {
        "trial_count": 1,
        "timeout_seconds": None,
        "sampler_seed": 2026,
        "enable_pruning": False,
    }
    lower_budget_config = load_test_tune_config(tmp_path, override=lower_budget_override)

    with pytest.raises(
        SpiceOperatorError,
        match="Requested trial_count is lower than existing study size",
    ):
        run_tune(lower_budget_config, reporter=NullReporter())


def test_tune_rejects_study_definition_drift(
    tmp_path,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    seed_history_dataset(config)

    run_tune(config, reporter=NullReporter())

    drifted_override = model_workflow_override() | tune_override()
    drifted_override["tuning_space"] = {
        "training": {
            "learning_rate": [0.0001, 0.001],
            "weight_decay": [0.0, 0.01],
        },
        "model": {
            "id": "lstm",
            "hidden_size": [64, 128],
            "dropout": [0.0, 0.1],
        },
    }
    drifted_config = load_test_tune_config(tmp_path, override=drifted_override)

    with pytest.raises(SpiceOperatorError, match="tuning_space"):
        run_tune(drifted_config, reporter=NullReporter())


def test_train_tuned_rejects_problem_drift_from_study(
    tmp_path,
    load_test_tune_config,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    tune_config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    seed_history_dataset(tune_config)
    run_tune(tune_config, reporter=NullReporter())
    tuned_override = model_workflow_override(lookback_seconds=240)
    tuned_override["artifact"] = {"variant": "tuned"}
    tuned_override["study"] = tune_config.study.name
    tuned_train_config = load_test_train_config(
        tmp_path,
        override=tuned_override,
    )

    with pytest.raises(SpiceOperatorError, match="problem"):
        run_train(tuned_train_config, reporter=NullReporter())


@pytest.mark.parametrize("compiler_id", ["timestamp_native", "estimated_block"])
def test_evaluate_workflow_smoke(
    tmp_path,
    compiler_id,
    load_test_evaluate_config,
    load_test_train_config,
    model_workflow_override,
    seed_evaluation_dataset,
    seed_history_dataset,
) -> None:
    override = model_workflow_override(compiler_id=compiler_id)
    train_config = load_test_train_config(tmp_path, override=override)
    evaluate_config = load_test_evaluate_config(tmp_path, override=override)
    seed_history_dataset(train_config)
    seed_evaluation_dataset(evaluate_config)
    run_train(train_config, reporter=NullReporter())
    evaluate_paths = resolve_workflow_paths(evaluate_config)

    run_evaluate(evaluate_config, reporter=NullReporter())

    assert evaluate_paths.artifact_state_db.is_file()
    runs = list_evaluation_runs(evaluate_paths.artifact_state_db)
    summary = load_evaluation_summary(evaluate_paths.artifact_state_db)
    assert runs
    assert summary is not None
    assert summary.runtime.runs == runs


def test_evaluate_workflow_supports_named_evaluation_override(
    tmp_path,
    load_test_evaluate_config,
    load_test_train_config,
    model_workflow_override,
    seed_evaluation_dataset,
    seed_history_dataset,
) -> None:
    override = model_workflow_override()
    override["evaluation"] = "paper_windowed_2h"
    train_config = load_test_train_config(tmp_path, override=override)
    evaluate_config = load_test_evaluate_config(tmp_path, override=override)
    seed_history_dataset(train_config)
    seed_evaluation_dataset(evaluate_config)

    run_train(train_config, reporter=NullReporter())
    run_evaluate(evaluate_config, reporter=NullReporter())

    summary = load_evaluation_summary(resolve_workflow_paths(evaluate_config).artifact_state_db)
    assert summary is not None
    assert summary.runtime.evaluator_id == "paper_windowed"
    assert summary.runtime.evaluator_config == {
        "id": "paper_windowed",
        "window_seconds": 7200,
        "repetitions": 50,
        "seed": 2026,
    }
    assert summary.runtime.runs
    assert summary.runtime.runs[0].metadata["mode"] == "fullset_fallback"


@pytest.mark.parametrize(
    ("prediction_name", "expected_family_id"),
    [
        ("candidate_offset_selection", "candidate_offset_selection"),
        ("icdcs_2026", "min_block_fee_multitask"),
    ],
)
def test_evaluate_workflow_supports_both_prediction_families(
    tmp_path,
    prediction_name,
    expected_family_id,
    load_test_evaluate_config,
    load_test_train_config,
    model_workflow_override,
    seed_evaluation_dataset,
    seed_history_dataset,
) -> None:
    override = model_workflow_override()
    override["prediction"] = prediction_name
    train_config = load_test_train_config(tmp_path, override=override)
    evaluate_config = load_test_evaluate_config(tmp_path, override=override)
    seed_history_dataset(train_config)
    seed_evaluation_dataset(evaluate_config)

    run_train(train_config, reporter=NullReporter())
    run_evaluate(evaluate_config, reporter=NullReporter())

    summary = load_evaluation_summary(resolve_workflow_paths(evaluate_config).artifact_state_db)
    assert summary is not None
    assert summary.manifest.prediction_family_id == expected_family_id
    assert summary.runtime.metric_descriptors


def test_evaluate_rejects_delay_request_above_capability(
    tmp_path,
    load_test_evaluate_config,
    model_workflow_override,
) -> None:
    override = model_workflow_override(max_delay_seconds=24)
    override["delay_seconds"] = 36
    with pytest.raises(
        ConfigResolutionError,
        match="delay_seconds must be <= problem.max_delay_seconds",
    ):
        load_test_evaluate_config(tmp_path, override=override)


@pytest.mark.parametrize(
    ("override", "error_pattern", "use_mismatch_feature_set", "use_mismatch_model"),
    [
        (
            {"feature_set": "time_native_baseline"},
            "Configured feature_set does not match the trained artifact semantics",
            True,
            False,
        ),
        (
            {"model": "transformer"},
            "Configured model does not match the trained artifact semantics",
            False,
            True,
        ),
    ],
)
def test_evaluate_validation_rejects_semantic_bundle_mismatch(
    tmp_path,
    override,
    error_pattern,
    use_mismatch_feature_set,
    use_mismatch_model,
    load_test_evaluate_config,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    base_override = model_workflow_override()
    train_config = load_test_train_config(tmp_path, override=base_override)
    evaluate_config = load_test_evaluate_config(tmp_path, override=base_override)
    mismatch_config = load_test_evaluate_config(
        tmp_path,
        override=base_override | override,
    )
    seed_history_dataset(train_config)
    run_train(train_config, reporter=NullReporter())
    loaded_artifact = load_training_artifact(resolve_workflow_paths(train_config).artifact_root)

    feature_set = (
        mismatch_config.feature_set if use_mismatch_feature_set else evaluate_config.feature_set
    )
    model = mismatch_config.model if use_mismatch_model else evaluate_config.model

    with pytest.raises(SpiceOperatorError, match=error_pattern):
        validate_artifact_semantics(
            loaded_artifact.manifest,
            problem=evaluate_config.problem,
            dataset_builder=evaluate_config.dataset_builder,
            feature_set=feature_set,
            prediction=evaluate_config.prediction,
            model=model,
        )
