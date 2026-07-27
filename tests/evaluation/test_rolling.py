from __future__ import annotations

import math
from pathlib import Path
from uuid import UUID

import polars as pl
import pytest

from fable.addresses import evaluation_directory
from fable.evaluation import OBSERVATION_SCHEMA, reduce_rolling

_EVALUATION_IDS = {
    horizon: UUID(f"20000000-0000-4000-8000-{horizon:012d}") for horizon in range(2, 6)
}
_RESULT_SCHEMA = pl.Schema(
    {
        "cell": pl.String,
        "one_shot_base_fee_savings": pl.Float64,
        "rolling_base_fee_savings": pl.Float64,
        "one_shot_p50_fee_inclusive_savings": pl.Float64,
        "rolling_p50_fee_inclusive_savings": pl.Float64,
        "one_shot_base_fee_optimality_gap": pl.Float64,
        "rolling_base_fee_optimality_gap": pl.Float64,
    }
)


def _publish_evaluation(
    storage_root: Path,
    horizon: int,
    *,
    first_origin: int,
    actions: list[int],
    selected_base_fees: list[int],
    selected_priority_fees: list[int],
) -> None:
    evaluation_id = _EVALUATION_IDS[horizon]
    rows = [
        {
            "origin_block": origin,
            "predicted_action_k": action,
            "predicted_minimum_log_base_fee": math.log(20),
            "minimum_action_k": 0,
            "immediate_base_fee_per_gas": 100,
            "immediate_effective_priority_fee_per_gas_p50": 10,
            "selected_base_fee_per_gas": selected_base_fee,
            "selected_effective_priority_fee_per_gas_p50": selected_priority_fee,
            "minimum_base_fee_per_gas": 20,
        }
        for origin, action, selected_base_fee, selected_priority_fee in zip(
            range(first_origin, first_origin + len(actions)),
            actions,
            selected_base_fees,
            selected_priority_fees,
            strict=True,
        )
    ]
    directory = evaluation_directory(storage_root, evaluation_id)
    directory.mkdir(parents=True)
    pl.DataFrame(rows, schema=OBSERVATION_SCHEMA).write_parquet(directory / "observations.parquet")


def _publish_rolling_evaluations(storage_root: Path) -> None:
    _publish_evaluation(
        storage_root,
        5,
        first_origin=100,
        actions=[0, 1, 1, 1, 1],
        selected_base_fees=[90, 80, 70, 60, 50],
        selected_priority_fees=[9, 8, 7, 6, 5],
    )
    _publish_evaluation(
        storage_root,
        4,
        first_origin=101,
        actions=[3, 0, 1, 1, 1],
        selected_base_fees=[1_000, 70, 1_000, 1_000, 1_000],
        selected_priority_fees=[100, 7, 100, 100, 100],
    )
    _publish_evaluation(
        storage_root,
        3,
        first_origin=102,
        actions=[2, 2, 0, 1, 1],
        selected_base_fees=[1_000, 1_000, 50, 1_000, 1_000],
        selected_priority_fees=[100, 100, 5, 100, 100],
    )
    _publish_evaluation(
        storage_root,
        2,
        first_origin=103,
        actions=[1, 1, 1, 0, 1],
        selected_base_fees=[1_000, 1_000, 1_000, 30, 20],
        selected_priority_fees=[100, 100, 100, 3, 2],
    )


def _roster() -> dict[str, dict[int, UUID]]:
    return {f"cell-{index}": dict(_EVALUATION_IDS) for index in range(9)}


def _observations_path(storage_root: Path, horizon: int) -> Path:
    return evaluation_directory(storage_root, _EVALUATION_IDS[horizon]) / "observations.parquet"


def test_reduce_rolling_reconstructs_every_stage_and_six_metrics(tmp_path: Path) -> None:
    _publish_rolling_evaluations(tmp_path)

    result = reduce_rolling(tmp_path, _roster())

    assert result.schema == _RESULT_SCHEMA
    assert result.shape == (9, 7)
    assert result["cell"].to_list() == [f"cell-{index}" for index in range(9)]
    for row in result.iter_rows(named=True):
        assert tuple(value for name, value in row.items() if name != "cell") == pytest.approx(
            (0.3, 0.48, 0.3, 0.48, 2.5, 1.6)
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("cells", "exactly nine named cells"),
        ("horizons", "must name exactly the K=2"),
        ("schema", "canonical ordered schema"),
        ("origins", "consecutive unique origins"),
        ("shift", "lacks required shifted origins"),
        ("action", "K=3 predicted_action_k values must be valid actions"),
    ],
)
def test_reduce_rolling_rejects_invalid_rosters_and_observations(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    _publish_rolling_evaluations(tmp_path)
    roster = _roster()
    if case == "cells":
        roster.pop("cell-8")
    elif case == "horizons":
        roster["cell-0"].pop(2)
    else:
        horizon = 2 if case == "shift" else 3
        observations_path = _observations_path(tmp_path, horizon)
        observations = pl.read_parquet(observations_path)
        if case == "schema":
            observations = observations.select(
                "predicted_action_k",
                *[name for name in OBSERVATION_SCHEMA if name != "predicted_action_k"],
            )
        elif case == "origins":
            observations = observations.with_columns(
                pl.when(pl.col("origin_block") == 103)
                .then(102)
                .otherwise(pl.col("origin_block"))
                .alias("origin_block")
            )
        elif case == "shift":
            observations = observations.filter(pl.col("origin_block") >= 104)
        elif case == "action":
            observations = observations.with_columns(
                pl.when(pl.col("origin_block") == 102)
                .then(3)
                .otherwise(pl.col("predicted_action_k"))
                .alias("predicted_action_k")
            )
        observations.write_parquet(observations_path)

    with pytest.raises(ValueError, match=message):
        reduce_rolling(tmp_path, roster)
