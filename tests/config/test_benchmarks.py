from __future__ import annotations

import pytest
import yaml

from spice.config.benchmarks import expand_benchmark_commands
from spice.core.errors import ConfigResolutionError


def _write_benchmark(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "benchmark" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_benchmark_expands_cases_in_order_and_matrix_axes(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "matrix_case",
        {
            "cases": [
                {
                    "surface": "block_open_lagged",
                    "workflow": "evaluate",
                    "feature_set": [
                        "block_open_lagged_no_time_since_start",
                        "block_open_lagged_calendar_only_time",
                    ],
                    "objective": "profit_poisson_replay_2h_mean",
                    "evaluation": "poisson_replay_2h_mean",
                    "delay_seconds": [12, 24],
                    "study": "safe_lstm_direct",
                    "variant": "baseline",
                },
                {
                    "surface": "same_block_closed",
                    "workflow": "train",
                    "study": "unsafe_lstm_direct",
                    "variant": "baseline",
                },
            ],
        },
    )

    commands = expand_benchmark_commands("matrix_case")

    assert commands == [
        (
            "spice evaluate --surface block_open_lagged "
            "--feature-set block_open_lagged_no_time_since_start "
            "--objective profit_poisson_replay_2h_mean --evaluation poisson_replay_2h_mean "
            "--study safe_lstm_direct --variant baseline --delay-seconds 12"
        ),
        (
            "spice evaluate --surface block_open_lagged "
            "--feature-set block_open_lagged_no_time_since_start "
            "--objective profit_poisson_replay_2h_mean --evaluation poisson_replay_2h_mean "
            "--study safe_lstm_direct --variant baseline --delay-seconds 24"
        ),
        (
            "spice evaluate --surface block_open_lagged "
            "--feature-set block_open_lagged_calendar_only_time "
            "--objective profit_poisson_replay_2h_mean --evaluation poisson_replay_2h_mean "
            "--study safe_lstm_direct --variant baseline --delay-seconds 12"
        ),
        (
            "spice evaluate --surface block_open_lagged "
            "--feature-set block_open_lagged_calendar_only_time "
            "--objective profit_poisson_replay_2h_mean --evaluation poisson_replay_2h_mean "
            "--study safe_lstm_direct --variant baseline --delay-seconds 24"
        ),
        (
            "spice train --surface same_block_closed --study unsafe_lstm_direct "
            "--variant baseline"
        ),
    ]


def test_benchmark_validates_all_rows_before_printing(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "invalid_case",
        {
            "cases": [
                {
                    "surface": "same_block_closed",
                    "workflow": "evaluate",
                    "delay_seconds": [12, 0],
                    "variant": "baseline",
                },
            ],
        },
    )

    with pytest.raises(
        ConfigResolutionError,
        match="benchmark invalid_case case 0 expanded row 1:",
    ):
        expand_benchmark_commands("invalid_case")


def test_benchmark_rejects_scheduler_shape(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "scheduler_field",
        {
            "cases": [
                {
                    "surface": "same_block_closed",
                    "workflow": "train",
                    "variant": "baseline",
                    "submit": True,
                },
            ],
        },
    )

    with pytest.raises(ConfigResolutionError, match="unknown benchmark case fields: submit"):
        expand_benchmark_commands("scheduler_field")
