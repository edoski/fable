from __future__ import annotations

import pytest
import yaml

from spice.benchmarks import plan_benchmark
from spice.config.models import EvaluateConfig, TrainConfig, WorkflowTask
from spice.core.errors import ConfigResolutionError
from spice.storage.catalog.store import upsert_study_record
from spice.storage.engine import state_db_path
from spice.storage.layout import catalog_db_path
from spice.storage.root_consumer_paths import produced_artifact_id

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
                                "evaluation": "poisson_replay_2h",
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
                                "evaluation": "poisson_replay_2h",
                                "delay_seconds": 36,
                            },
                        },
                    ],
                },
            ],
        },
    )

    plan = plan_benchmark("dimension_case")

    assert len(plan) == 12
    train = next(entry for entry in plan if entry.step_id == "train_tuned")
    evaluate = next(
        entry
        for entry in plan
        if entry.step_id == "evaluate_tuned"
        and "models-model-lstm__tuning_space-lstm_large_capacity" in entry.run_id
        and "problems-current_row_nominal__lookback_seconds-600__sample_count-1000000"
        in entry.run_id
    )
    assert train.config.model_dump()["study_id"]
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.artifact_from == evaluate.depends_on[0]
    assert evaluate.config.dataset_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id.startswith("art_")
    assert evaluate.config.delay_seconds == 36


def test_packaged_benchmark_yamls_keep_expected_shapes() -> None:
    expected_counts = {
        "evaluator_objective_grid": 54,
        "large_capacity_hpo": 27,
        "priority_fee_trends_ablation": 36,
        "safe_baseline_grid": 18,
        "delay_degradation_sweep": 180,
        "elapsed_position_ablation": 36,
        "lookback_window_sweep": 54,
        "slot_spacing_sweep": 36,
    }

    for name, expected_count in expected_counts.items():
        plan = plan_benchmark(name)
        evaluate_entries = [
            entry for entry in plan if entry.workflow is WorkflowTask.EVALUATE
        ]
        train_entries = [entry for entry in plan if entry.workflow is WorkflowTask.TRAIN]

        assert len(plan) == expected_count
        assert evaluate_entries
        assert train_entries
        assert all(isinstance(entry.config, EvaluateConfig) for entry in evaluate_entries)
        assert all(entry.artifact_from in entry.depends_on for entry in evaluate_entries)


def test_evaluator_objective_grid_keeps_cross_evaluation_bindings() -> None:
    plan = plan_benchmark("evaluator_objective_grid")

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
    assert poisson_full.selection["surface"] == "current_row_fee_dynamics"
    assert poisson_full.selection["model"] == "lstm"
    assert poisson_full.selection["problem"] == "current_row_nominal"
    assert poisson_full.selection["artifact_id"] == poisson_full.config.artifact_id
    assert poisson_full.depends_on == (
        "evaluator_objective_grid."
        "data-chain-ethereum__dataset_id-cor_9a73b1e88edb488afb1e."
        "models-model-lstm__tuning_space-lstm_large_capacity."
        "problems-current_row_nominal."
        "train_poisson_objective",
    )
    assert isinstance(full_poisson.config, EvaluateConfig)
    assert full_poisson.config.evaluation.id == "poisson_replay_2h"
    assert full_poisson.depends_on == (
        "evaluator_objective_grid."
        "data-chain-ethereum__dataset_id-cor_9a73b1e88edb488afb1e."
        "models-model-lstm__tuning_space-lstm_large_capacity."
        "problems-current_row_nominal."
        "train_full_objective",
    )


def test_evaluate_plan_entries_keep_inherited_ledger_context(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "ledger_context",
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "training": "default",
                        "split": "default",
                        "study": "ledger_context",
                        "objective": "validation_total_loss",
                    },
                    "dimensions": {
                        "data": [
                            {
                                "set": {
                                    "chain": "ethereum",
                                    "dataset_id": ETH_DATASET_ID,
                                }
                            }
                        ],
                        "models": [
                            {
                                "set": {
                                    "model": "lstm",
                                    "tuning_space": "lstm_large_capacity",
                                }
                            }
                        ],
                        "problems": [{"ref": "current_row_nominal"}],
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {"variant": "baseline"},
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "artifact_from": "train",
                            "set": {"evaluation": "poisson_replay_2h"},
                        },
                    ],
                }
            ]
        },
    )

    evaluate = next(
        entry for entry in plan_benchmark("ledger_context") if entry.step_id == "evaluate"
    )

    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.selection["surface"] == "current_row_fee_dynamics"
    assert evaluate.selection["objective"] == "validation_total_loss"
    assert evaluate.selection["model"] == "lstm"
    assert evaluate.selection["problem"] == "current_row_nominal"
    assert evaluate.selection["dataset_id"] == ETH_DATASET_ID
    assert evaluate.selection["artifact_id"] == evaluate.config.artifact_id


def test_artifact_from_explicit_tuned_study_uses_catalog_dataset(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    storage_root = tmp_path / "outputs"
    study_root = storage_root / "studies" / "ethereum" / "std_existing"
    upsert_study_record(
        catalog_db_path(storage_root),
        study_id="std_existing",
        study_name="existing",
        dataset_id=ETH_DATASET_ID,
        dataset_name="icdcs_2026",
        chain_name="ethereum",
        features_id="core_fee_dynamics",
        prediction_id="icdcs_2026",
        model_id="lstm",
        problem_id="current_row_nominal",
        root_path=study_root,
        state_db_path=state_db_path(study_root),
    )
    _write_benchmark(
        conf_root,
        "explicit_tuned_study",
        {
            "cases": [
                {
                    "id": "case",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "storage_root": str(storage_root),
                        "training": "default",
                        "split": "default",
                        "study": "external",
                        "objective": "validation_total_loss",
                    },
                    "dimensions": {
                        "data": [{"set": {"chain": "ethereum"}}],
                        "models": [
                            {
                                "set": {
                                    "model": "lstm",
                                    "tuning_space": "lstm_large_capacity",
                                }
                            }
                        ],
                        "problems": [{"ref": "current_row_nominal"}],
                    },
                    "steps": [
                        {
                            "id": "train_tuned",
                            "workflow": "train",
                            "set": {
                                "variant": "tuned",
                                "study_id": "std_existing",
                            },
                        },
                        {
                            "id": "evaluate_tuned",
                            "workflow": "evaluate",
                            "artifact_from": "train_tuned",
                            "set": {"evaluation": "poisson_replay_2h"},
                        },
                    ],
                }
            ]
        },
    )

    plan = plan_benchmark("explicit_tuned_study")
    train = next(entry for entry in plan if entry.step_id == "train_tuned")
    evaluate = next(entry for entry in plan if entry.step_id == "evaluate_tuned")

    assert isinstance(train.config, TrainConfig)
    assert isinstance(evaluate.config, EvaluateConfig)
    assert evaluate.config.dataset_id == ETH_DATASET_ID
    assert evaluate.config.artifact_id == produced_artifact_id(
        train.config,
        dataset_id=ETH_DATASET_ID,
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
        plan_benchmark("bad_evaluate")


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
        plan_benchmark("invalid_grid")


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
        plan_benchmark("cycle")
