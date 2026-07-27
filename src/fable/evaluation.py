"""Native artifact evaluation and scientific reduction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl
import torch
from torch import nn

from . import _runtime
from .addresses import (
    evaluation_directory,
    evaluation_json_path,
    evaluation_observations_path,
)
from .config import EvaluateRequest
from .corpus import load_corpus
from .min_block_fee import TargetState, decode_action
from .modeling import load_artifact
from .temporal import HistoricalDataset, prepare_historical_window

OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "predicted_action_k": pl.Int64,
        "predicted_minimum_log_base_fee": pl.Float64,
        "minimum_action_k": pl.Int64,
        "immediate_base_fee_per_gas": pl.Int64,
        "immediate_effective_priority_fee_per_gas_p50": pl.Int64,
        "selected_base_fee_per_gas": pl.Int64,
        "selected_effective_priority_fee_per_gas_p50": pl.Int64,
        "minimum_base_fee_per_gas": pl.Int64,
    }
)

_DEVICE = torch.device("cuda:0")
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
_ROLLING_HORIZONS = (2, 3, 4, 5)
_ROLLING_RESULT_SCHEMA = pl.Schema(
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


def evaluate(
    request: EvaluateRequest,
    storage_root: Path,
) -> None:
    """Publish canonical observations for one exact artifact/window request."""

    scratch = storage_root / "evaluations" / f".{request.evaluation_id}"
    scratch.mkdir(parents=True)

    corpus = load_corpus(storage_root, request.corpus_id)
    association, model = load_artifact(storage_root, request.artifact_id)
    if association.request.source.corpus_id != request.corpus_id:
        raise ValueError("artifact source Corpus must match the evaluation Corpus")
    experiment = association.training_definition.experiment
    testing_window = request.testing_window
    dataset = prepare_historical_window(
        corpus,
        experiment,
        testing_window,
        feature_state=association.feature_state,
        target_state=association.target_state,
    )
    first_outcome_block = testing_window.first_parent_block + 1
    outcome_priority_fees_p50 = (
        corpus.blocks.select_range(
            first_outcome_block,
            testing_window.last_parent_block + experiment.horizon_blocks,
        )
        .to_polars()["effective_priority_fee_per_gas_p50"]
        .to_numpy()
    )

    _runtime.configure_torch()
    observations = _collect_observations(
        dataset,
        model,
        target_state=association.target_state,
        outcome_priority_fees_p50=outcome_priority_fees_p50,
        first_outcome_block=first_outcome_block,
    )
    (scratch / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")
    observations.write_parquet(scratch / "observations.parquet")

    canonical = evaluation_directory(storage_root, request.evaluation_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    scratch.rename(canonical)


def _collect_observations(
    dataset: HistoricalDataset,
    model: nn.Module,
    *,
    target_state: TargetState,
    outcome_priority_fees_p50: np.ndarray,
    first_outcome_block: int,
) -> pl.DataFrame:
    count = len(dataset)
    columns = {
        name: np.empty(count, dtype=dtype.to_python()) for name, dtype in OBSERVATION_SCHEMA.items()
    }

    loader = _runtime.data_loader(
        dataset,
        batch_size=_runtime.EVALUATION_BATCH_SIZE,
        shuffle=False,
    )
    model.to(_DEVICE)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["inputs"].to(_DEVICE))
            actions = decode_action(output).cpu().numpy()
            minimum_actions_batch = batch["label"].numpy()
            base_fees = batch["base_fees"].numpy()

            rows = np.arange(actions.size, dtype=np.int64)
            immediate_batch = base_fees[:, 0]
            selected_batch = base_fees[rows, actions]
            minimum_batch = base_fees[rows, minimum_actions_batch]
            immediate_outcome_rows = batch["origin_block"].numpy() + 1 - first_outcome_block
            immediate_priority_fees_p50_batch = outcome_priority_fees_p50[immediate_outcome_rows]
            selected_priority_fees_p50_batch = outcome_priority_fees_p50[
                immediate_outcome_rows + actions
            ]
            predicted_logs_batch = target_state.mean + target_state.standard_deviation * (
                output.minimum_fee_z.cpu().numpy().astype(np.float64)
            )
            if not np.isfinite(predicted_logs_batch).all():
                raise ValueError("predicted minimum-log fees must be finite")

            size = actions.size
            destination = slice(cursor, cursor + size)
            columns["origin_block"][destination] = batch["origin_block"].numpy()
            columns["predicted_action_k"][destination] = actions
            columns["predicted_minimum_log_base_fee"][destination] = predicted_logs_batch
            columns["minimum_action_k"][destination] = minimum_actions_batch
            columns["immediate_base_fee_per_gas"][destination] = immediate_batch
            columns["immediate_effective_priority_fee_per_gas_p50"][destination] = (
                immediate_priority_fees_p50_batch
            )
            columns["selected_base_fee_per_gas"][destination] = selected_batch
            columns["selected_effective_priority_fee_per_gas_p50"][destination] = (
                selected_priority_fees_p50_batch
            )
            columns["minimum_base_fee_per_gas"][destination] = minimum_batch
            cursor += size

    return pl.DataFrame(columns, schema=OBSERVATION_SCHEMA)


def reduce_evaluation(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive one testing evaluation's seven metrics from its observations."""

    return _reduce(_load_evaluation(storage_root, evaluation_id))


def reduce_rolling(
    storage_root: Path,
    roster: Mapping[str, Mapping[int, UUID]],
) -> pl.DataFrame:
    """Compare one-shot and rolling economics for nine explicit K-study cells."""

    if len(roster) != 9 or any(not cell for cell in roster):
        raise ValueError("rolling roster must contain exactly nine named cells")

    rows = [
        _reduce_rolling_cell(storage_root, cell, evaluation_ids)
        for cell, evaluation_ids in roster.items()
    ]
    return pl.DataFrame(rows, schema=_ROLLING_RESULT_SCHEMA)


def _load_evaluation(
    storage_root: Path,
    evaluation_id: UUID,
) -> pl.DataFrame:
    request = EvaluateRequest.model_validate_json(
        evaluation_json_path(storage_root, evaluation_id).read_text(encoding="utf-8"),
        strict=True,
    )
    if request.evaluation_id != evaluation_id:
        raise ValueError("evaluation request ID must match the requested evaluation")
    return _load_observations(storage_root, request)


def _load_observations(storage_root: Path, request: EvaluateRequest) -> pl.DataFrame:
    path = evaluation_observations_path(storage_root, request.evaluation_id)
    observations = _read_observations(path)
    window = request.testing_window
    expected_origins = np.arange(
        window.first_parent_block,
        window.last_parent_block + 1,
        dtype=np.int64,
    )
    origins = observations["origin_block"].to_numpy()
    if not np.array_equal(origins, expected_origins):
        raise ValueError("observation origins must exactly match the ordered testing window")
    return observations


def _read_observations(path: Path) -> pl.DataFrame:
    if pl.read_parquet_schema(path) != OBSERVATION_SCHEMA:
        raise ValueError("observations must have the canonical ordered schema")
    observations = pl.read_parquet(path)
    if any(observations.null_count().row(0)):
        raise ValueError("observations must contain no null values")
    return observations


def _reduce(observations: pl.DataFrame) -> pl.DataFrame:
    predicted_actions = observations["predicted_action_k"].to_numpy()
    minimum_actions = observations["minimum_action_k"].to_numpy()
    predicted_logs = observations["predicted_minimum_log_base_fee"].to_numpy()
    immediate_fees = observations["immediate_base_fee_per_gas"].to_numpy()
    immediate_priority_fees_p50 = observations[
        "immediate_effective_priority_fee_per_gas_p50"
    ].to_numpy()
    selected_fees = observations["selected_base_fee_per_gas"].to_numpy()
    selected_priority_fees_p50 = observations[
        "selected_effective_priority_fee_per_gas_p50"
    ].to_numpy()
    minimum_fees = observations["minimum_base_fee_per_gas"].to_numpy()

    log_errors = predicted_logs - np.log(minimum_fees)
    classes = np.union1d(minimum_actions, predicted_actions)
    f1_by_class = [
        2.0
        * np.count_nonzero((minimum_actions == action) & (predicted_actions == action))
        / (
            np.count_nonzero(minimum_actions == action)
            + np.count_nonzero(predicted_actions == action)
        )
        for action in classes
    ]
    metrics: dict[str, float] = {
        "accuracy": float(np.mean(predicted_actions == minimum_actions)),
        "f1_macro": float(np.mean(f1_by_class)),
        "log_fee_mae": float(np.mean(np.abs(log_errors))),
        "log_fee_mse": float(np.mean(np.square(log_errors))),
    }
    metrics.update(
        _economic_metrics(
            immediate_fees,
            immediate_priority_fees_p50,
            selected_fees,
            selected_priority_fees_p50,
            minimum_fees,
        )
    )
    if not np.isfinite(tuple(metrics.values())).all():
        raise ValueError("evaluation reduction must contain only finite metrics")
    return pl.DataFrame(
        {name: [value] for name, value in metrics.items()},
        schema=_RESULT_SCHEMA,
    )


def _reduce_rolling_cell(
    storage_root: Path,
    cell: str,
    evaluation_ids: Mapping[int, UUID],
) -> dict[str, str | float]:
    if tuple(sorted(evaluation_ids)) != _ROLLING_HORIZONS:
        raise ValueError(f"{cell} must name exactly the K=2, K=3, K=4, and K=5 evaluations")

    evaluations = {
        horizon: _load_rolling_observations(storage_root, evaluation_ids[horizon])
        for horizon in _ROLLING_HORIZONS
    }

    for horizon, observations in evaluations.items():
        for column in ("predicted_action_k", "minimum_action_k"):
            actions = observations[column].to_numpy()
            if np.any((actions < 0) | (actions >= horizon)):
                raise ValueError(f"{cell} K={horizon} {column} values must be valid actions")

    initial = evaluations[5]
    initial_origins = initial["origin_block"].to_numpy()
    aligned = {
        horizon: _align_observations(
            evaluations[horizon],
            initial_origins + (5 - horizon),
            cell=cell,
            horizon=horizon,
        )
        for horizon in _ROLLING_HORIZONS
    }

    rolling_base_fees = aligned[5]["selected_base_fee_per_gas"].copy()
    rolling_priority_fees = aligned[5]["selected_effective_priority_fee_per_gas_p50"].copy()
    waiting = aligned[5]["predicted_action_k"] != 0
    for horizon in (4, 3, 2):
        selected = waiting & (aligned[horizon]["predicted_action_k"] == 0)
        rolling_base_fees[selected] = aligned[horizon]["selected_base_fee_per_gas"][selected]
        rolling_priority_fees[selected] = aligned[horizon][
            "selected_effective_priority_fee_per_gas_p50"
        ][selected]
        waiting &= aligned[horizon]["predicted_action_k"] != 0

    rolling_base_fees[waiting] = aligned[2]["selected_base_fee_per_gas"][waiting]
    rolling_priority_fees[waiting] = aligned[2]["selected_effective_priority_fee_per_gas_p50"][
        waiting
    ]

    immediate_base_fees = aligned[5]["immediate_base_fee_per_gas"]
    immediate_priority_fees = aligned[5]["immediate_effective_priority_fee_per_gas_p50"]
    minimum_base_fees = aligned[5]["minimum_base_fee_per_gas"]
    one_shot = _economic_metrics(
        immediate_base_fees,
        immediate_priority_fees,
        aligned[5]["selected_base_fee_per_gas"],
        aligned[5]["selected_effective_priority_fee_per_gas_p50"],
        minimum_base_fees,
    )
    rolling = _economic_metrics(
        immediate_base_fees,
        immediate_priority_fees,
        rolling_base_fees,
        rolling_priority_fees,
        minimum_base_fees,
    )
    metrics = {
        "cell": cell,
        **{f"one_shot_{name}": value for name, value in one_shot.items()},
        **{f"rolling_{name}": value for name, value in rolling.items()},
    }
    metric_values = tuple(value for value in metrics.values() if isinstance(value, float))
    if not np.isfinite(metric_values).all():
        raise ValueError(f"{cell} rolling comparison must contain only finite metrics")
    return metrics


def _load_rolling_observations(
    storage_root: Path,
    evaluation_id: UUID,
) -> pl.DataFrame:
    observations = _read_observations(evaluation_observations_path(storage_root, evaluation_id))
    origins = observations["origin_block"].to_numpy()
    if origins.size == 0 or np.any(np.diff(origins) != 1):
        raise ValueError("rolling observations must contain consecutive unique origins")
    return observations


def _align_observations(
    observations: pl.DataFrame,
    required_origins: np.ndarray,
    *,
    cell: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    origins = observations["origin_block"].to_numpy()
    positions = np.searchsorted(origins, required_origins)
    if np.any(positions == origins.size) or not np.array_equal(
        origins[positions],
        required_origins,
    ):
        raise ValueError(f"{cell} K={horizon} evaluation lacks required shifted origins")
    return {
        name: observations[name].to_numpy()[positions]
        for name in (
            "predicted_action_k",
            "immediate_base_fee_per_gas",
            "immediate_effective_priority_fee_per_gas_p50",
            "selected_base_fee_per_gas",
            "selected_effective_priority_fee_per_gas_p50",
            "minimum_base_fee_per_gas",
        )
    }


def _economic_metrics(
    immediate_base_fees: np.ndarray,
    immediate_priority_fees_p50: np.ndarray,
    selected_base_fees: np.ndarray,
    selected_priority_fees_p50: np.ndarray,
    minimum_base_fees: np.ndarray,
) -> dict[str, float]:
    return {
        "base_fee_savings": float(
            np.mean((immediate_base_fees - selected_base_fees) / immediate_base_fees)
        ),
        "p50_fee_inclusive_savings": float(
            np.mean(
                1.0
                - (selected_base_fees + selected_priority_fees_p50)
                / (immediate_base_fees + immediate_priority_fees_p50)
            )
        ),
        "base_fee_optimality_gap": float(
            np.mean((selected_base_fees - minimum_base_fees) / minimum_base_fees)
        ),
    }
