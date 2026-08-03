from __future__ import annotations

import math
from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl
import pytest

from kairos.addresses import evaluation_directory
from kairos.evaluation import reduce_rolling
from kairos.observations import OBSERVATION_SCHEMA

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
    base_fees: dict[int, int],
    priority_fees: dict[int, int],
) -> None:
    evaluation_id = _EVALUATION_IDS[horizon]
    rows = []
    for origin, action in zip(
        range(first_origin, first_origin + len(actions)), actions, strict=True
    ):
        outcome_blocks = range(origin + 1, origin + horizon + 1)
        outcome_base_fees = [base_fees[block] for block in outcome_blocks]
        minimum_action = int(np.argmin(outcome_base_fees))
        selected_block = origin + 1 + action
        deadline_block = origin + horizon
        rows.append(
            {
                "origin_block": origin,
                "predicted_action_k": action,
                "predicted_minimum_log_base_fee": math.log(outcome_base_fees[minimum_action]),
                "minimum_action_k": minimum_action,
                "immediate_base_fee_per_gas": base_fees[origin + 1],
                "immediate_effective_priority_fee_per_gas_p50": priority_fees[origin + 1],
                "selected_base_fee_per_gas": base_fees[selected_block],
                "selected_effective_priority_fee_per_gas_p50": priority_fees[selected_block],
                "deadline_base_fee_per_gas": base_fees[deadline_block],
                "deadline_effective_priority_fee_per_gas_p50": priority_fees[deadline_block],
                "minimum_base_fee_per_gas": outcome_base_fees[minimum_action],
            }
        )
    directory = evaluation_directory(storage_root, evaluation_id)
    directory.mkdir(parents=True)
    pl.DataFrame(rows, schema=OBSERVATION_SCHEMA).write_parquet(directory / "observations.parquet")


def _publish_all_terminal_evaluations(storage_root: Path) -> None:
    base_fees = {101: 100, 102: 80, 103: 60, 104: 40, 105: 20}
    priority_fees = {101: 10, 102: 8, 103: 6, 104: 4, 105: 2}
    actions_by_horizon = {5: [4], 4: [0, 3], 3: [0, 0, 2], 2: [0, 0, 0, 1]}
    for horizon, actions in actions_by_horizon.items():
        _publish_evaluation(
            storage_root,
            horizon,
            first_origin=100,
            actions=actions,
            base_fees=base_fees,
            priority_fees=priority_fees,
        )


def _roster() -> dict[str, dict[int, UUID]]:
    return {"lstm.ethereum": dict(_EVALUATION_IDS)}


def _observations_path(storage_root: Path, horizon: int) -> Path:
    return evaluation_directory(storage_root, _EVALUATION_IDS[horizon]) / "observations.parquet"


def test_reduce_rolling_all_terminal_actions_end_at_original_deadline(tmp_path: Path) -> None:
    _publish_all_terminal_evaluations(tmp_path)

    result = reduce_rolling(tmp_path, _roster())

    assert result.schema == _RESULT_SCHEMA
    assert result["cell"].to_list() == ["lstm.ethereum"]
    assert result.select(pl.exclude("cell")).row(0) == pytest.approx((0.8, 0.8, 0.8, 0.8, 0.0, 0.0))


def test_reduce_rolling_all_nonterminal_actions_keep_origin_and_k2_is_final(tmp_path: Path) -> None:
    base_fees = {101: 100, 102: 40, 103: 60, 104: 80, 105: 20}
    priority_fees = {101: 10, 102: 4, 103: 6, 104: 8, 105: 2}
    for horizon, actions in ((5, [3]), (4, [2]), (3, [1]), (2, [0])):
        _publish_evaluation(
            tmp_path,
            horizon,
            first_origin=100,
            actions=actions,
            base_fees=base_fees,
            priority_fees=priority_fees,
        )

    result = reduce_rolling(tmp_path, _roster())

    assert result.select(pl.exclude("cell")).row(0) == pytest.approx((0.2, 0.0, 0.2, 0.0, 3.0, 4.0))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("origins", "consecutive unique origins"),
        ("missing", "lacks required decision origins"),
        ("action", "K=3 predicted_action_k values must be valid actions"),
    ],
)
def test_reduce_rolling_rejects_invalid_observations(
    tmp_path: Path, case: str, message: str
) -> None:
    _publish_all_terminal_evaluations(tmp_path)
    horizon = 2 if case == "missing" else 3
    observations_path = _observations_path(tmp_path, horizon)
    observations = pl.read_parquet(observations_path)
    if case == "origins":
        observations = observations.with_columns(
            pl.when(pl.col("origin_block") == 102)
            .then(101)
            .otherwise(pl.col("origin_block"))
            .alias("origin_block")
        )
    elif case == "missing":
        observations = observations.filter(pl.col("origin_block") < 103)
    elif case == "action":
        observations = observations.with_columns(
            pl.when(pl.col("origin_block") == 102)
            .then(3)
            .otherwise(pl.col("predicted_action_k"))
            .alias("predicted_action_k")
        )
    observations.write_parquet(observations_path)

    with pytest.raises(ValueError, match=message):
        reduce_rolling(tmp_path, _roster())
