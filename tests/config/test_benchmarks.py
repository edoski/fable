from __future__ import annotations

from typing import cast

import pytest
import yaml

from spice.config.benchmarks import plan_benchmark
from spice.config.models import ModelWorkflowConfig
from spice.core.errors import ConfigResolutionError


def _write_benchmark(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "benchmark" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_benchmark_dimensions_expand_resolved_plan(isolate_conf_root) -> None:
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
                                        "lookback_seconds": [600, 900],
                                        "sample_count": [1000000, 2000000],
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
                                "evaluation": "fullset",
                                "trial_count": 3,
                            },
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "after": ["tune"],
                            "dimensions": {
                                "scoring": [
                                    {
                                        "set": {
                                            "objective": "profit_poisson_replay_2h_mean",
                                            "evaluation": "poisson_replay_2h_mean",
                                        }
                                    },
                                    {
                                        "set": {
                                            "objective": "profit_poisson_replay_2h_total",
                                            "evaluation": "poisson_replay_2h_total",
                                        }
                                    },
                                ],
                                "runtime": [
                                    {"set": {"variant": "tuned", "delay_seconds": 36}},
                                ],
                            },
                        },
                    ],
                },
            ],
        },
    )

    plan = plan_benchmark("dimension_case")

    assert len(plan) == 30
    tuned_problems = {
        entry.config.problem.id
        for entry in plan
        if entry.config.problem.id.startswith("current_row_nominal__")
    }
    assert (
        "current_row_nominal__lookback_seconds-600__sample_count-1000000"
        in tuned_problems
    )
    assert all(
        cast(ModelWorkflowConfig, entry.config).model.id == "lstm"
        for entry in plan
        if "models-model-lstm__tuning_space-lstm_large_capacity" in entry.run_id
    )
    evaluate_entries = [entry for entry in plan if entry.step_id == "evaluate"]
    assert evaluate_entries
    assert all(len(entry.depends_on) == 1 for entry in evaluate_entries)
    assert all(
        cast(ModelWorkflowConfig, entry.config).artifact.variant.value == "tuned"
        for entry in evaluate_entries
    )


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
