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
from .addresses import evaluation_directory, evaluation_observations_path
from .config import EvaluateRequest
from .corpus import load_corpus_blocks
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


def evaluate(request: EvaluateRequest, storage_root: Path) -> None:
    """Publish canonical observations for one exact artifact/window request."""

    scratch = storage_root / "evaluations" / f".{request.evaluation_id}"
    scratch.mkdir(parents=True)

    blocks = load_corpus_blocks(storage_root, request.corpus_id)
    association, model = load_artifact(storage_root, request.artifact_id)
    if association.request.source.corpus_id != request.corpus_id:
        raise ValueError("artifact source Corpus must match the evaluation Corpus")
    experiment = association.training_definition.experiment
    testing_window = request.testing_window
    dataset = prepare_historical_window(
        blocks,
        experiment,
        testing_window,
        feature_state=association.feature_state,
        target_state=association.target_state,
    )
    first_outcome_block = testing_window.first_parent_block + 1
    outcomes = blocks.select_range(
        first_outcome_block, testing_window.last_parent_block + experiment.horizon_blocks
    ).to_polars()
    outcome_base_fees = outcomes["base_fee_per_gas"].to_numpy()
    outcome_priority_fees_p50 = outcomes["effective_priority_fee_per_gas_p50"].to_numpy()

    _runtime.configure_torch()
    observations = _collect_observations(
        dataset,
        model,
        target_state=association.target_state,
        outcome_base_fees=outcome_base_fees,
        outcome_priority_fees_p50=outcome_priority_fees_p50,
        first_outcome_block=first_outcome_block,
        horizon_blocks=experiment.horizon_blocks,
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
    outcome_base_fees: np.ndarray,
    outcome_priority_fees_p50: np.ndarray,
    first_outcome_block: int,
    horizon_blocks: int,
) -> pl.DataFrame:
    count = len(dataset)
    origin_blocks = np.empty(count, dtype=np.int64)
    predicted_actions = np.empty(count, dtype=np.int64)
    predicted_minimum_z = np.empty(count, dtype=np.float64)
    minimum_actions = np.empty(count, dtype=np.int64)

    dataset = dataset.to(_DEVICE)
    loader = dataset.loader(batch_size=_runtime.EVALUATION_BATCH_SIZE, shuffle=False)
    model.to(_DEVICE)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["inputs"])
            actions = decode_action(output).cpu().numpy()
            size = actions.size
            destination = slice(cursor, cursor + size)
            origin_blocks[destination] = batch["origin_block"].cpu().numpy()
            predicted_actions[destination] = actions
            predicted_minimum_z[destination] = output.minimum_fee_z.cpu().numpy()
            minimum_actions[destination] = batch["label"].cpu().numpy()
            cursor += size

    predicted_logs = target_state.mean + target_state.standard_deviation * predicted_minimum_z
    if not np.isfinite(predicted_logs).all():
        raise ValueError("predicted minimum-log fees must be finite")
    outcome_rows = origin_blocks + 1 - first_outcome_block
    return pl.DataFrame(
        {
            "origin_block": origin_blocks,
            "predicted_action_k": predicted_actions,
            "predicted_minimum_log_base_fee": predicted_logs,
            "minimum_action_k": minimum_actions,
            "immediate_base_fee_per_gas": outcome_base_fees[outcome_rows],
            "immediate_effective_priority_fee_per_gas_p50": outcome_priority_fees_p50[outcome_rows],
            "selected_base_fee_per_gas": outcome_base_fees[outcome_rows + predicted_actions],
            "selected_effective_priority_fee_per_gas_p50": outcome_priority_fees_p50[
                outcome_rows + predicted_actions
            ],
            "deadline_base_fee_per_gas": outcome_base_fees[outcome_rows + horizon_blocks - 1],
            "deadline_effective_priority_fee_per_gas_p50": outcome_priority_fees_p50[
                outcome_rows + horizon_blocks - 1
            ],
            "minimum_base_fee_per_gas": outcome_base_fees[outcome_rows + minimum_actions],
        },
        schema=OBSERVATION_SCHEMA,
    )


def reduce_evaluation(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive one testing evaluation's seven metrics from its observations."""

    columns = _read_observations(evaluation_observations_path(storage_root, evaluation_id))
    log_errors = columns["predicted_minimum_log_base_fee"] - np.log(
        columns["minimum_base_fee_per_gas"]
    )
    metrics = {
        **_classification_metrics(columns["predicted_action_k"], columns["minimum_action_k"]),
        "log_fee_mae": float(np.mean(np.abs(log_errors))),
        "log_fee_mse": float(np.mean(np.square(log_errors))),
        **_economic_metrics(columns, "selected"),
    }
    return pl.DataFrame([metrics])


def reduce_baselines(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive immediate and deadline policy metrics from one testing evaluation."""

    columns = _read_observations(evaluation_observations_path(storage_root, evaluation_id))
    rows = []
    for policy in ("immediate", "deadline"):
        metrics = _economic_metrics(columns, policy)
        rows.append({"policy": policy, **metrics})
    return pl.DataFrame(rows)


def reduce_rolling(storage_root: Path, roster: Mapping[str, Mapping[int, UUID]]) -> pl.DataFrame:
    """Compare one-shot and rolling economics for explicit K-study cells."""

    rows = [
        _reduce_rolling_cell(storage_root, cell, evaluation_ids)
        for cell, evaluation_ids in roster.items()
    ]
    return pl.DataFrame(rows)


def _read_observations(path: Path) -> dict[str, np.ndarray]:
    observations = pl.read_parquet(path)
    if observations.schema != OBSERVATION_SCHEMA:
        raise ValueError("observations must have the canonical ordered schema")
    return {name: observations[name].to_numpy() for name in OBSERVATION_SCHEMA}


def _classification_metrics(
    predicted_actions: np.ndarray, minimum_actions: np.ndarray
) -> dict[str, float]:
    matches = predicted_actions == minimum_actions
    class_count = max(int(predicted_actions.max()), int(minimum_actions.max())) + 1
    truth = np.bincount(minimum_actions, minlength=class_count)
    predictions = np.bincount(predicted_actions, minlength=class_count)
    true_positives = np.bincount(minimum_actions[matches], minlength=class_count)
    denominators = truth + predictions
    present = denominators > 0
    f1_by_class = 2.0 * true_positives[present] / denominators[present]
    return {"accuracy": float(np.mean(matches)), "f1_macro": float(np.mean(f1_by_class))}


def _reduce_rolling_cell(
    storage_root: Path, cell: str, evaluation_ids: Mapping[int, UUID]
) -> dict[str, str | float]:
    decision_origins: np.ndarray | None = None
    selections = []
    for horizon in range(5, 1, -1):
        columns = _load_rolling_observations(storage_root, evaluation_ids[horizon])
        if decision_origins is None:
            decision_origins = columns["origin_block"].copy()
        selection = _rolling_arrays(
            columns, decision_origins=decision_origins, cell=cell, horizon=horizon
        )
        selections.append(selection)
        if horizon > 2:
            decision_origins += selection["predicted_action_k"] == horizon - 1

    initial = selections[0]
    final = selections[-1]

    one_shot = _economic_metrics(initial, "selected")
    rolling = _economic_metrics(initial, "selected", selected=final)
    metrics = {}
    for name in one_shot:
        metrics[f"one_shot_{name}"] = one_shot[name]
        metrics[f"rolling_{name}"] = rolling[name]
    return {"cell": cell, **metrics}


def _load_rolling_observations(storage_root: Path, evaluation_id: UUID) -> dict[str, np.ndarray]:
    columns = _read_observations(evaluation_observations_path(storage_root, evaluation_id))
    origins = columns["origin_block"]
    if origins.size == 0 or np.any(np.diff(origins) != 1):
        raise ValueError("rolling observations must contain consecutive unique origins")
    return columns


def _rolling_arrays(
    columns: Mapping[str, np.ndarray], *, decision_origins: np.ndarray, cell: str, horizon: int
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
                - (selected_base_fees + selected[f"{policy}_effective_priority_fee_per_gas_p50"])
                / (immediate_base_fees + columns["immediate_effective_priority_fee_per_gas_p50"])
            )
        ),
        "base_fee_optimality_gap": float(
            np.mean((selected_base_fees - minimum_base_fees) / minimum_base_fees)
        ),
    }
