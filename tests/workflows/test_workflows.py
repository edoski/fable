from __future__ import annotations

import pytest

from spice.core.reporting import NullReporter
from spice.modeling.artifacts import load_training_artifact, validate_artifact_semantics
from spice.storage.artifact import list_simulation_runs, load_simulation_summary
from spice.storage.catalog import list_artifact_records, list_study_records
from spice.storage.study import load_study
from spice.workflows.simulate import run as run_simulate
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
    seed_history_dataset(config)

    run_train(config, reporter=NullReporter())

    assert config.paths.artifact_state_db.is_file()
    assert (config.paths.artifact_root / "model.pt").is_file()
    artifacts = list_artifact_records(
        config.paths.catalog_db,
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        feature_set_id=config.feature_set.id,
        model_id=config.model.id,
        problem_id=config.problem.id,
        variant=config.artifact.variant.value,
    )
    assert len(artifacts) == 1
    assert artifacts[0].artifact_id == config.paths.artifact_id


def test_tune_then_train_tuned_smoke(
    tmp_path,
    deep_merge,
    load_test_train_config,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=deep_merge(model_workflow_override(), tune_override()),
    )
    seed_history_dataset(config)

    run_tune(config, reporter=NullReporter())

    assert config.paths.study_state_db.is_file()
    study = load_study(config.paths.study_state_db, study_name=config.study.name)
    assert len(study.trials) == config.tuning.trial_count
    studies = list_study_records(
        config.paths.catalog_db,
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        feature_set_id=config.feature_set.id,
        model_id=config.model.id,
        problem_id=config.problem.id,
        study_name=config.study.name,
    )
    assert len(studies) == 1
    assert studies[0].study_id == config.paths.study_id

    tuned_train_config = load_test_train_config(
        tmp_path,
        override=deep_merge(
            model_workflow_override(),
            {
                "artifact": {"variant": "tuned"},
                "study": config.study.name,
            },
        ),
    )
    run_train(tuned_train_config, reporter=NullReporter())

    assert tuned_train_config.paths.artifact_state_db.is_file()
    assert (tuned_train_config.paths.artifact_root / "model.pt").is_file()


def test_tune_resume_same_study_tops_up_to_target(
    tmp_path,
    deep_merge,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    initial_config = load_test_tune_config(
        tmp_path,
        override=deep_merge(model_workflow_override(), tune_override()),
    )
    seed_history_dataset(initial_config)

    run_tune(initial_config, reporter=NullReporter())

    resumed_override = deep_merge(
        deep_merge(model_workflow_override(), tune_override()),
        {"tuning": {"trial_count": 4, "enable_pruning": False}},
    )
    resumed_config = load_test_tune_config(tmp_path, override=resumed_override)

    run_tune(resumed_config, reporter=NullReporter())

    study = load_study(resumed_config.paths.study_state_db, study_name=resumed_config.study.name)
    assert len(study.trials) == 4


def test_tune_rejects_lower_requested_trial_budget(
    tmp_path,
    deep_merge,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=deep_merge(model_workflow_override(), tune_override()),
    )
    seed_history_dataset(config)

    run_tune(config, reporter=NullReporter())

    lower_budget_config = load_test_tune_config(
        tmp_path,
        override=deep_merge(
            deep_merge(model_workflow_override(), tune_override()),
            {"tuning": {"trial_count": 1, "enable_pruning": False}},
        ),
    )

    with pytest.raises(ValueError, match="Requested trial_count is lower than existing study size"):
        run_tune(lower_budget_config, reporter=NullReporter())


def test_tune_rejects_study_definition_drift(
    tmp_path,
    deep_merge,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=deep_merge(model_workflow_override(), tune_override()),
    )
    seed_history_dataset(config)

    run_tune(config, reporter=NullReporter())

    drifted_override = deep_merge(
        deep_merge(model_workflow_override(), tune_override()),
        {
            "tuning_space": {
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
        },
    )
    drifted_config = load_test_tune_config(tmp_path, override=drifted_override)

    with pytest.raises(ValueError, match="tuning_space"):
        run_tune(drifted_config, reporter=NullReporter())


def test_train_tuned_rejects_problem_drift_from_study(
    tmp_path,
    deep_merge,
    load_test_tune_config,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    tune_config = load_test_tune_config(
        tmp_path,
        override=deep_merge(model_workflow_override(), tune_override()),
    )
    seed_history_dataset(tune_config)
    run_tune(tune_config, reporter=NullReporter())

    tuned_train_config = load_test_train_config(
        tmp_path,
        override=deep_merge(
            model_workflow_override(lookback_seconds=240),
            {
                "artifact": {"variant": "tuned"},
                "study": tune_config.study.name,
            },
        ),
    )

    with pytest.raises(ValueError, match="problem"):
        run_train(tuned_train_config, reporter=NullReporter())


@pytest.mark.parametrize("compiler_id", ["timestamp_native", "estimated_block"])
def test_simulate_workflow_smoke(
    tmp_path,
    compiler_id,
    load_test_simulate_config,
    load_test_train_config,
    model_workflow_override,
    seed_evaluation_dataset,
    seed_history_dataset,
) -> None:
    override = model_workflow_override(compiler_id=compiler_id)
    train_config = load_test_train_config(tmp_path, override=override)
    simulate_config = load_test_simulate_config(tmp_path, override=override)
    seed_history_dataset(train_config)
    seed_evaluation_dataset(simulate_config)
    run_train(train_config, reporter=NullReporter())

    run_simulate(simulate_config, reporter=NullReporter())

    assert simulate_config.paths.artifact_state_db.is_file()
    runs = list_simulation_runs(simulate_config.paths.artifact_state_db)
    summary = load_simulation_summary(simulate_config.paths.artifact_state_db)
    assert runs
    assert summary is not None
    assert summary.runs == runs


@pytest.mark.parametrize(
    ("prediction_name", "expected_family_id"),
    [
        ("candidate_offset_selection", "candidate_offset_selection"),
        ("icdcs_2026_paper", "min_block_fee_multitask"),
    ],
)
def test_simulate_workflow_supports_both_prediction_families(
    tmp_path,
    deep_merge,
    prediction_name,
    expected_family_id,
    load_test_simulate_config,
    load_test_train_config,
    model_workflow_override,
    seed_evaluation_dataset,
    seed_history_dataset,
) -> None:
    override = deep_merge(
        model_workflow_override(),
        {"prediction": prediction_name},
    )
    train_config = load_test_train_config(tmp_path, override=override)
    simulate_config = load_test_simulate_config(tmp_path, override=override)
    seed_history_dataset(train_config)
    seed_evaluation_dataset(simulate_config)

    run_train(train_config, reporter=NullReporter())
    run_simulate(simulate_config, reporter=NullReporter())

    summary = load_simulation_summary(simulate_config.paths.artifact_state_db)
    assert summary is not None
    assert summary.prediction_family_id == expected_family_id
    assert summary.simulation_metric_descriptors


def test_simulate_rejects_execution_request_above_capability(
    tmp_path,
    deep_merge,
    load_test_simulate_config,
    model_workflow_override,
) -> None:
    override = deep_merge(
        model_workflow_override(max_supported_delay_seconds=24),
        {
            "execution": {
                "id": "too_large",
                "requested_delay_seconds": 36,
            }
        },
    )
    with pytest.raises(
        ValueError,
        match="execution.requested_delay_seconds must be <=",
    ):
        load_test_simulate_config(tmp_path, override=override)


@pytest.mark.parametrize(
    ("override", "error_pattern", "use_mismatch_feature_set", "use_mismatch_model"),
    [
        (
            {"feature_set": "icdcs_2026_time_native"},
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
def test_simulate_validation_rejects_semantic_bundle_mismatch(
    tmp_path,
    deep_merge,
    override,
    error_pattern,
    use_mismatch_feature_set,
    use_mismatch_model,
    load_test_simulate_config,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
) -> None:
    base_override = model_workflow_override()
    train_config = load_test_train_config(tmp_path, override=base_override)
    simulate_config = load_test_simulate_config(tmp_path, override=base_override)
    mismatch_config = load_test_simulate_config(
        tmp_path,
        override=deep_merge(base_override, override),
    )
    seed_history_dataset(train_config)
    run_train(train_config, reporter=NullReporter())
    loaded_artifact = load_training_artifact(train_config.paths.artifact_root)

    feature_set = (
        mismatch_config.feature_set
        if use_mismatch_feature_set
        else simulate_config.feature_set
    )
    model = mismatch_config.model if use_mismatch_model else simulate_config.model

    with pytest.raises(ValueError, match=error_pattern):
        validate_artifact_semantics(
            loaded_artifact.manifest,
            problem=simulate_config.problem,
            feature_set=feature_set,
            prediction=simulate_config.prediction,
            model=model,
        )
