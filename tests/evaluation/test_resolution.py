from __future__ import annotations

import math
from pathlib import Path
from uuid import UUID

import polars as pl
import pytest

from fable.addresses import evaluation_directory
from fable.config import BlockWindow, EvaluateRequest
from fable.evaluation import reduce_baselines, reduce_evaluation

_EVALUATION_ID = UUID("10000000-0000-4000-8000-000000000001")
_OTHER_EVALUATION_ID = UUID("10000000-0000-4000-8000-000000000002")
_ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
_CORPUS_ID = UUID("30000000-0000-4000-8000-000000000001")

_OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "predicted_action_k": pl.Int64,
        "predicted_minimum_log_base_fee": pl.Float64,
        "minimum_action_k": pl.Int64,
        "deadline_action_k": pl.Int64,
        "immediate_base_fee_per_gas": pl.Int64,
        "immediate_effective_priority_fee_per_gas_p50": pl.Int64,
        "selected_base_fee_per_gas": pl.Int64,
        "selected_effective_priority_fee_per_gas_p50": pl.Int64,
        "deadline_base_fee_per_gas": pl.Int64,
        "deadline_effective_priority_fee_per_gas_p50": pl.Int64,
        "minimum_base_fee_per_gas": pl.Int64,
    }
)
_RESULT_SCHEMA = pl.Schema(
    {
        "accuracy": pl.Float64,
        "f1_macro": pl.Float64,
        "log_fee_mae": pl.Float64,
        "log_fee_mse": pl.Float64,
        "base_fee_savings": pl.Float64,
        "p50_fee_inclusive_savings": pl.Float64,
        "base_fee_optimality_gap": pl.Float64,
    }
)
_BASELINE_RESULT_SCHEMA = pl.Schema(
    {
        "policy": pl.String,
        "base_fee_savings": pl.Float64,
        "p50_fee_inclusive_savings": pl.Float64,
        "base_fee_optimality_gap": pl.Float64,
    }
)


def _request(
    *,
    evaluation_id: UUID = _EVALUATION_ID,
    testing_window: BlockWindow | None = None,
) -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=evaluation_id,
        artifact_id=_ARTIFACT_ID,
        corpus_id=_CORPUS_ID,
        testing_window=testing_window or BlockWindow(first_parent_block=20, last_parent_block=26),
    )


def _row(
    origin: int,
    predicted_action: int,
    predicted_log: float | None,
    minimum_action: int,
    immediate_fee: int,
    immediate_priority_fee_p50: int,
    selected_fee: int,
    selected_priority_fee_p50: int,
    deadline_fee: int,
    deadline_priority_fee_p50: int,
    minimum_fee: int,
) -> dict[str, int | float | None]:
    return {
        "origin_block": origin,
        "predicted_action_k": predicted_action,
        "predicted_minimum_log_base_fee": predicted_log,
        "minimum_action_k": minimum_action,
        "deadline_action_k": 3,
        "immediate_base_fee_per_gas": immediate_fee,
        "immediate_effective_priority_fee_per_gas_p50": immediate_priority_fee_p50,
        "selected_base_fee_per_gas": selected_fee,
        "selected_effective_priority_fee_per_gas_p50": selected_priority_fee_p50,
        "deadline_base_fee_per_gas": deadline_fee,
        "deadline_effective_priority_fee_per_gas_p50": deadline_priority_fee_p50,
        "minimum_base_fee_per_gas": minimum_fee,
    }


def _rows() -> list[dict[str, int | float | None]]:
    return [
        _row(20, 0, math.log(10) + 1.0, 0, 10, 0, 10, 0, 30, 3, 10),
        _row(21, 1, math.log(10) - 1.0, 2, 20, 0, 15, 5, 10, 0, 10),
        _row(22, 2, math.log(12) + 2.0, 2, 30, 10, 12, 8, 12, 8, 12),
        _row(23, 3, math.log(10) - 2.0, 1, 40, 0, 20, 20, 40, 0, 10),
        _row(24, 1, math.log(25), 1, 50, 0, 25, 25, 75, 25, 25),
        _row(25, 0, math.log(15) + 0.5, 3, 60, 10, 60, 10, 15, 10, 15),
        _row(26, 2, math.log(14) - 0.5, 0, 14, 6, 20, 0, 28, 2, 14),
    ]


def _observations(rows: list[dict[str, int | float | None]] | None = None) -> pl.DataFrame:
    return pl.DataFrame(rows or _rows(), schema=_OBSERVATION_SCHEMA)


def _publish_evaluation(
    storage_root: Path,
    request: EvaluateRequest,
    observations: pl.DataFrame,
) -> None:
    directory = evaluation_directory(storage_root, _EVALUATION_ID)
    directory.mkdir(parents=True)
    (directory / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")
    observations.write_parquet(directory / "observations.parquet")


def test_reduce_evaluation_derives_exact_metrics_from_self_contained_observations(
    tmp_path: Path,
) -> None:
    _publish_evaluation(tmp_path, _request(), _observations())

    result = reduce_evaluation(tmp_path, _EVALUATION_ID)

    assert result.schema == _RESULT_SCHEMA
    assert result.height == 1
    assert result.row(0) == pytest.approx(
        (
            3.0 / 7.0,
            0.375,
            1.0,
            1.5,
            199.0 / 980.0,
            1.0 / 14.0,
            69.0 / 98.0,
        )
    )


def test_reduce_baselines_derives_immediate_and_deadline_metrics(
    tmp_path: Path,
) -> None:
    _publish_evaluation(tmp_path, _request(), _observations())

    result = reduce_baselines(tmp_path, _EVALUATION_ID)

    assert result.schema == _BASELINE_RESULT_SCHEMA
    assert result["policy"].to_list() == ["immediate", "deadline"]
    assert result.select(pl.exclude("policy")).rows() == pytest.approx(
        [
            (
                0.0,
                0.0,
                19.0 / 14.0,
            ),
            (
                -33.0 / 140.0,
                -151.0 / 490.0,
                8.0 / 7.0,
            ),
        ]
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("uuid", "evaluation request ID must match the requested evaluation"),
        ("window", "observation origins must exactly match the ordered testing window"),
        ("schema", "observations must have the canonical ordered schema"),
        ("null", "observations must contain no null values"),
        ("origins", "observation origins must exactly match the ordered testing window"),
    ],
)
def test_reduce_evaluation_rejects_invalid_observation_contract(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    request = _request()
    rows = _rows()
    if case == "uuid":
        request = _request(evaluation_id=_OTHER_EVALUATION_ID)
    elif case == "window":
        rows.pop()
    elif case == "null":
        rows[0]["predicted_minimum_log_base_fee"] = None
    elif case == "origins":
        rows[1]["origin_block"] = 22

    observations = _observations(rows)
    if case == "schema":
        observations = observations.select(
            "predicted_action_k",
            *[name for name in _OBSERVATION_SCHEMA if name != "predicted_action_k"],
        )
    _publish_evaluation(tmp_path, request, observations)

    with pytest.raises(ValueError, match=message):
        reduce_evaluation(tmp_path, _EVALUATION_ID)


def test_reduce_evaluation_rejects_finite_inputs_that_overflow_a_metric(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["predicted_minimum_log_base_fee"] = 1e308
    _publish_evaluation(tmp_path, _request(), _observations(rows))

    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="only finite metrics"):
        reduce_evaluation(tmp_path, _EVALUATION_ID)
