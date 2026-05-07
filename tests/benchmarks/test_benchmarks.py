from __future__ import annotations

import pytest
import yaml

from spice.benchmarks.plan_materialization import materialize_benchmark_plan
from spice.config.models import EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.core.errors import ConfigResolutionError

ETH_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _write_benchmark(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "benchmark" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_benchmark_dimensions_expand_tuned_train_and_artifact_from(
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "dimension_case",
        {
            "cases": [
                {
                    "id": "window_sweep",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "training": "default",
                        "split": "default",
                        "tuning": "extensive",
                        "study": "window_sweep",
                    },
                    "dimensions": {
                        "data": [{"set": {"chain": "ethereum", "dataset_id": ETH_DATASET_ID}}],
                        "models": [
                            {"set": {"model": "lstm", "tuning_space": "lstm_large_capacity"}},
                            {
                                "set": {
                                    "model": "transformer",
                                    "tuning_space": "transformer_large_capacity",
                                }
                            },
                        ],
                        "problems": [
                            {"ref": "current_row_recent_median"},
                            {
                                "grid": {
                                    "base": "current_row_nominal",
                                    "fields": {
                                        "lookback_seconds": [600],
                                        "sample_count": [1000000],
                                    },
                                }
                            },
                        ],
                    },
                    "steps": [
                        {
                            "id": "tune",
                            "workflow": "tune",
                            "set": {
                                "objective": "validation_total_loss",
                                "evaluation": "poisson_replay",
                                "trial_count": 3,
                            },
                        },
                        {
                            "id": "train_tuned",
                            "workflow": "train",
                            "after": ["tune"],
                            "set": {"variant": "tuned"},
                        },
                        {
                            "id": "evaluate_tuned",
                            "workflow": "evaluate",
                            "after": ["train_tuned"],
                            "artifact_from": "train_tuned",
                            "set": {
                                "evaluation": "poisson_replay",
                                "delay_seconds": 36,
                            },
                        },
                    ],
                },
            ],
        },
    )

    plan = materialize_benchmark_plan("dimension_case")

    assert len(plan) == 12
    train = next(entry for entry in plan if entry.step_id == "train_tuned")
    grid_tune = next(
        entry
        for entry in plan
        if entry.step_id == "tune"
        and "problems-current_row_nominal__lookback_seconds-600__sample_count-1000000"
        in entry.run_id
    )
    evaluate = next(
        entry
        for entry in plan
        if entry.step_id == "evaluate_tuned"
        and "models-model-lstm__tuning_space-lstm_large_capacity" in entry.run_id
        and "problems-current_row_nominal__lookback_seconds-600__sample_count-1000000"
        in entry.run_id
    )
    assert isinstance(train.config, TrainConfig)
    assert train.config.study_id is not None
    assert isinstance(grid_tune.config, TuneConfig)
    assert grid_tune.config.problem.id == (
        "current_row_nominal__lookback_seconds-600__sample_count-1000000"
    )
    assert grid_tune.config.problem.lookback_seconds == 600
    assert grid_tune.config.problem.sample_count == 1000000
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.dependencies.artifact_from_run_id == evaluate.dependencies.local_run_ids[0]
    assert evaluate.config.dataset_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id.startswith("art_")
    assert evaluate.config.delay_seconds == 36


def test_packaged_benchmark_yamls_keep_expected_shapes() -> None:
    expected_counts = {
        "evaluator_objective_grid": 54,
        "large_capacity_hpo": 27,
        "priority_fee_ablation": 36,
        "unsafe_core_fee_dynamics_ablation": 36,
        "safe_baseline_grid": 18,
        "delay_degradation_sweep": 180,
        "elapsed_position_ablation": 36,
        "lookback_window_sweep": 54,
        "slot_spacing_sweep": 36,
    }

    for name, expected_count in expected_counts.items():
        plan = materialize_benchmark_plan(name)
        evaluate_entries = [
            entry for entry in plan if entry.workflow is WorkflowTask.EVALUATE
        ]
        train_entries = [entry for entry in plan if entry.workflow is WorkflowTask.TRAIN]

        assert len(plan) == expected_count
        assert evaluate_entries
        assert train_entries
        assert all(isinstance(entry.config, EvaluateConfig) for entry in evaluate_entries)
        assert all(
            entry.dependencies.artifact_from_run_id in entry.dependencies.local_run_ids
            for entry in evaluate_entries
        )


def test_evaluator_objective_grid_keeps_cross_evaluation_bindings() -> None:
    plan = materialize_benchmark_plan("evaluator_objective_grid")

    poisson_full = next(
        entry
        for entry in plan
        if entry.step_id == "evaluate_poisson_artifact_with_full"
        and "data-chain-ethereum__dataset_id-cor_9a73b1e88edb488afb1e" in entry.run_id
        and "models-model-lstm__tuning_space-lstm_large_capacity" in entry.run_id
    )
    full_poisson = next(
        entry
        for entry in plan
        if entry.step_id == "evaluate_full_artifact_with_poisson"
        and "data-chain-ethereum__dataset_id-cor_9a73b1e88edb488afb1e" in entry.run_id
        and "models-model-lstm__tuning_space-lstm_large_capacity" in entry.run_id
    )

    assert isinstance(poisson_full.config, EvaluateConfig)
    assert poisson_full.config.evaluation.id == "full_temporal_replay"
    assert poisson_full.selection.surface == "current_row_fee_dynamics"
    assert poisson_full.selection.model == "lstm"
    assert poisson_full.selection.problem == "current_row_nominal"
    assert poisson_full.root_facts.consumed_artifact_id == poisson_full.config.artifact_id
    assert poisson_full.dependencies.local_run_ids == (
        "evaluator_objective_grid."
        "data-chain-ethereum__dataset_id-cor_9a73b1e88edb488afb1e."
        "models-model-lstm__tuning_space-lstm_large_capacity."
        "problems-current_row_nominal."
        "train_poisson_objective",
    )
    assert isinstance(full_poisson.config, EvaluateConfig)
    assert full_poisson.config.evaluation.id == "poisson_replay"
    assert full_poisson.dependencies.local_run_ids == (
        "evaluator_objective_grid."
        "data-chain-ethereum__dataset_id-cor_9a73b1e88edb488afb1e."
        "models-model-lstm__tuning_space-lstm_large_capacity."
        "problems-current_row_nominal."
        "train_full_objective",
    )


def test_benchmark_rejects_step_local_evaluate_training_fields(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "bad_evaluate",
        {
            "cases": [
                {
                    "id": "bad",
                    "base": {"surface": "current_row_fee_dynamics"},
                    "steps": [
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "set": {"variant": "baseline"},
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(ConfigResolutionError, match="does not support fields: variant"):
        materialize_benchmark_plan("bad_evaluate")


def test_benchmark_rejects_invalid_problem_grid(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "invalid_grid",
        {
            "cases": [
                {
                    "id": "bad",
                    "base": {"surface": "current_row_fee_dynamics"},
                    "dimensions": {
                        "problems": [
                            {
                                "grid": {
                                    "base": "current_row_nominal",
                                    "fields": {"sample_count": [0]},
                                }
                            }
                        ]
                    },
                    "steps": [{"id": "tune", "workflow": "tune"}],
                }
            ]
        },
    )

    with pytest.raises(ConfigResolutionError, match="values must be positive"):
        materialize_benchmark_plan("invalid_grid")


def test_benchmark_rejects_step_dependency_cycles(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "cycle",
        {
            "cases": [
                {
                    "id": "bad",
                    "base": {"surface": "current_row_fee_dynamics"},
                    "steps": [
                        {"id": "train", "workflow": "train", "after": ["evaluate"]},
                        {"id": "evaluate", "workflow": "evaluate", "after": ["train"]},
                    ],
                }
            ]
        },
    )

    with pytest.raises(ConfigResolutionError, match="future step"):
        materialize_benchmark_plan("cycle")


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        (
            [
                {"id": "train", "workflow": "train"},
                {"id": "train", "workflow": "train"},
            ],
            "benchmark step ids must be unique",
        ),
        (
            [
                {"id": "evaluate", "workflow": "evaluate", "after": ["missing"]},
            ],
            "step evaluate depends on unknown step missing",
        ),
        (
            [
                {"id": "evaluate", "workflow": "evaluate", "after": ["evaluate"]},
            ],
            "step evaluate cannot depend on itself",
        ),
        (
            [
                {"id": "evaluate", "workflow": "evaluate", "artifact_from": "missing"},
            ],
            "step evaluate depends on unknown step missing",
        ),
        (
            [
                {"id": "evaluate", "workflow": "evaluate", "artifact_from": "train"},
                {"id": "train", "workflow": "train"},
            ],
            "step evaluate depends on future step train",
        ),
        (
            [
                {"id": "evaluate", "workflow": "evaluate", "artifact_from": "evaluate"},
            ],
            "step evaluate cannot depend on itself",
        ),
    ],
)
def test_benchmark_dependency_validation_keeps_specific_errors(
    isolate_conf_root,
    steps,
    error: str,
) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "bad_dependencies",
        {
            "cases": [
                {
                    "id": "bad",
                    "base": {"surface": "current_row_fee_dynamics"},
                    "steps": steps,
                }
            ]
        },
    )

    with pytest.raises(ConfigResolutionError, match=error):
        materialize_benchmark_plan("bad_dependencies")
