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
        "deadline_base_fee_per_gas": pl.Int64,
        "deadline_effective_priority_fee_per_gas_p50": pl.Int64,
        "minimum_base_fee_per_gas": pl.Int64,
    }
)

_DEVICE = torch.device("cuda:0")


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

    dataset = dataset.to(_DEVICE)
    loader = dataset.loader(
        batch_size=_runtime.EVALUATION_BATCH_SIZE,
        shuffle=False,
    )
    model.to(_DEVICE)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["inputs"])
            actions = decode_action(output).cpu().numpy()
            minimum_actions_batch = batch["label"].cpu().numpy()
            base_fees = batch["base_fees"].cpu().numpy()
            origin_blocks_batch = batch["origin_block"].cpu().numpy()

            rows = np.arange(actions.size, dtype=np.int64)
            immediate_batch = base_fees[:, 0]
            selected_batch = base_fees[rows, actions]
            deadline_batch = base_fees[:, -1]
            minimum_batch = base_fees[rows, minimum_actions_batch]
            immediate_outcome_rows = origin_blocks_batch + 1 - first_outcome_block
            immediate_priority_fees_p50_batch = outcome_priority_fees_p50[immediate_outcome_rows]
            selected_priority_fees_p50_batch = outcome_priority_fees_p50[
                immediate_outcome_rows + actions
            ]
            deadline_priority_fees_p50_batch = outcome_priority_fees_p50[
                immediate_outcome_rows + base_fees.shape[1] - 1
            ]
            predicted_logs_batch = target_state.mean + target_state.standard_deviation * (
                output.minimum_fee_z.cpu().numpy().astype(np.float64)
            )
            if not np.isfinite(predicted_logs_batch).all():
                raise ValueError("predicted minimum-log fees must be finite")

            size = actions.size
            destination = slice(cursor, cursor + size)
            batch_columns = {
                "origin_block": origin_blocks_batch,
                "predicted_action_k": actions,
                "predicted_minimum_log_base_fee": predicted_logs_batch,
                "minimum_action_k": minimum_actions_batch,
                "immediate_base_fee_per_gas": immediate_batch,
                "immediate_effective_priority_fee_per_gas_p50": (immediate_priority_fees_p50_batch),
                "selected_base_fee_per_gas": selected_batch,
                "selected_effective_priority_fee_per_gas_p50": (selected_priority_fees_p50_batch),
                "deadline_base_fee_per_gas": deadline_batch,
                "deadline_effective_priority_fee_per_gas_p50": (deadline_priority_fees_p50_batch),
                "minimum_base_fee_per_gas": minimum_batch,
            }
            for name in OBSERVATION_SCHEMA:
                columns[name][destination] = batch_columns[name]
            cursor += size

    return pl.DataFrame(columns, schema=OBSERVATION_SCHEMA)


def reduce_evaluation(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive one testing evaluation's seven metrics from its observations."""

    return _reduce(_load_evaluation(storage_root, evaluation_id))


def reduce_baselines(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive immediate and deadline policy metrics from one testing evaluation."""

    columns = _load_evaluation(storage_root, evaluation_id)
    rows = []
    for policy in ("immediate", "deadline"):
        metrics = _economic_metrics(columns, policy)
        if not np.isfinite(tuple(metrics.values())).all():
            raise ValueError("baseline reduction must contain only finite metrics")
        rows.append({"policy": policy, **metrics})
    return pl.DataFrame(rows)


def reduce_rolling(
    storage_root: Path,
    roster: Mapping[str, Mapping[int, UUID]],
) -> pl.DataFrame:
    """Compare one-shot and rolling economics for explicit K-study cells."""

    rows = [
        _reduce_rolling_cell(storage_root, cell, evaluation_ids)
        for cell, evaluation_ids in roster.items()
    ]
    return pl.DataFrame(rows)


def _load_evaluation(
    storage_root: Path,
    evaluation_id: UUID,
) -> dict[str, np.ndarray]:
    request = EvaluateRequest.model_validate_json(
        evaluation_json_path(storage_root, evaluation_id).read_text(encoding="utf-8"),
        strict=True,
    )
    if request.evaluation_id != evaluation_id:
        raise ValueError("evaluation request ID must match the requested evaluation")
    return _load_observations(storage_root, request)


def _load_observations(
    storage_root: Path,
    request: EvaluateRequest,
) -> dict[str, np.ndarray]:
    path = evaluation_observations_path(storage_root, request.evaluation_id)
    columns = _read_observations(path)
    window = request.testing_window
    expected_origins = np.arange(
        window.first_parent_block,
        window.last_parent_block + 1,
        dtype=np.int64,
    )
    origins = columns["origin_block"]
    if not np.array_equal(origins, expected_origins):
        raise ValueError("observation origins must exactly match the ordered testing window")
    return columns


def _read_observations(path: Path) -> dict[str, np.ndarray]:
    observations = pl.read_parquet(path)
    if observations.schema != OBSERVATION_SCHEMA:
        raise ValueError("observations must have the canonical ordered schema")
    if any(observations.null_count().row(0)):
        raise ValueError("observations must contain no null values")
    return {name: observations[name].to_numpy() for name in OBSERVATION_SCHEMA}


def _reduce(columns: Mapping[str, np.ndarray]) -> pl.DataFrame:
    log_errors = columns["predicted_minimum_log_base_fee"] - np.log(
        columns["minimum_base_fee_per_gas"]
    )
    metrics = {
        **_classification_metrics(
            columns["predicted_action_k"],
            columns["minimum_action_k"],
        ),
        "log_fee_mae": float(np.mean(np.abs(log_errors))),
        "log_fee_mse": float(np.mean(np.square(log_errors))),
        **_economic_metrics(columns, "selected"),
    }
    if not np.isfinite(tuple(metrics.values())).all():
        raise ValueError("evaluation reduction must contain only finite metrics")
    return pl.DataFrame({name: [value] for name, value in metrics.items()})


def _classification_metrics(
    predicted_actions: np.ndarray,
    minimum_actions: np.ndarray,
) -> dict[str, float]:
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
    return {
        "accuracy": float(np.mean(predicted_actions == minimum_actions)),
        "f1_macro": float(np.mean(f1_by_class)),
    }


def _reduce_rolling_cell(
    storage_root: Path,
    cell: str,
    evaluation_ids: Mapping[int, UUID],
) -> dict[str, str | float]:
    decision_origins: np.ndarray | None = None
    selections = []
    for horizon in range(5, 1, -1):
        columns = _load_rolling_observations(storage_root, evaluation_ids[horizon])
        if decision_origins is None:
            decision_origins = columns["origin_block"].copy()
        selection = _rolling_arrays(
            columns,
            decision_origins=decision_origins,
            cell=cell,
            horizon=horizon,
        )
        selections.append(selection)
        if horizon > 2:
            decision_origins += selection["predicted_action_k"] == horizon - 1

    initial = selections[0]
    final = selections[-1]

    one_shot = _economic_metrics(initial, "selected")
    rolling = _economic_metrics(initial, "selected", selected=final)
    metrics: dict[str, str | float] = {"cell": cell}
    for name in one_shot:
        metrics[f"one_shot_{name}"] = one_shot[name]
        metrics[f"rolling_{name}"] = rolling[name]
    metric_values = tuple(value for value in metrics.values() if isinstance(value, float))
    if not np.isfinite(metric_values).all():
        raise ValueError(f"{cell} rolling comparison must contain only finite metrics")
    return metrics


def _load_rolling_observations(
    storage_root: Path,
    evaluation_id: UUID,
) -> dict[str, np.ndarray]:
    columns = _read_observations(evaluation_observations_path(storage_root, evaluation_id))
    origins = columns["origin_block"]
    if origins.size == 0 or np.any(np.diff(origins) != 1):
        raise ValueError("rolling observations must contain consecutive unique origins")
    return columns


def _rolling_arrays(
    columns: Mapping[str, np.ndarray],
    *,
    decision_origins: np.ndarray,
    cell: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    actions = columns["predicted_action_k"]
    if np.any((actions < 0) | (actions >= horizon)):
        raise ValueError(f"{cell} K={horizon} predicted_action_k values must be valid actions")

    origins = columns["origin_block"]
    rows = decision_origins - int(origins[0])
    if np.any((rows < 0) | (rows >= origins.size)):
        raise ValueError(f"{cell} K={horizon} evaluation lacks required decision origins")
    return {name: values[rows] for name, values in columns.items()}


def _economic_metrics(
    columns: Mapping[str, np.ndarray],
    policy: str,
    *,
    selected: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    selected = columns if selected is None else selected
    immediate_base_fees = columns["immediate_base_fee_per_gas"]
    minimum_base_fees = columns["minimum_base_fee_per_gas"]
    selected_base_fees = selected[f"{policy}_base_fee_per_gas"]
    return {
        "base_fee_savings": float(
            np.mean((immediate_base_fees - selected_base_fees) / immediate_base_fees)
        ),
        "p50_fee_inclusive_savings": float(
            np.mean(
                1.0
                - (
                    selected_base_fees
                    + selected[f"{policy}_effective_priority_fee_per_gas_p50"]
                )
                / (
                    immediate_base_fees
                    + columns["immediate_effective_priority_fee_per_gas_p50"]
                )
            )
        ),
        "base_fee_optimality_gap": float(
            np.mean((selected_base_fees - minimum_base_fees) / minimum_base_fees)
        ),
    }
