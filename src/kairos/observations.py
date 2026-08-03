"""Shared prediction observations and scientific reduction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from . import _runtime
from .config import BlockWindow
from .corpus import BlockFrame
from .min_block_fee import TargetState, decode_action
from .temporal import HistoricalDataset

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

_WEI_PER_GWEI = 1_000_000_000.0


def collect_observations(
    dataset: HistoricalDataset,
    model: nn.Module,
    blocks: BlockFrame,
    window: BlockWindow,
    *,
    target_state: TargetState,
    horizon_blocks: int,
    device: torch.device,
    batch_size: int,
    autocast_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """Run one ordered inference pass and own its prediction/outcome facts."""

    first_outcome_block = window.first_parent_block + 1
    outcomes = blocks.select_range(
        first_outcome_block, window.last_parent_block + horizon_blocks
    ).to_polars()
    outcome_base_fees = outcomes["base_fee_per_gas"].to_numpy()
    outcome_priority_fees_p50 = outcomes["effective_priority_fee_per_gas_p50"].to_numpy()

    count = len(dataset)
    origin_blocks = np.empty(count, dtype=np.int64)
    predicted_actions = np.empty(count, dtype=np.int64)
    predicted_minimum_z = np.empty(count, dtype=np.float64)
    minimum_actions = np.empty(count, dtype=np.int64)

    loader = _runtime.data_loader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            if autocast_dtype is None:
                output = model(inputs)
            else:
                with torch.autocast(device.type, dtype=autocast_dtype):
                    output = model(inputs)
            if (
                not torch.isfinite(output.action_logits).all()
                or not torch.isfinite(output.minimum_fee_z).all()
            ):
                raise ValueError("predictions must be finite")
            actions = decode_action(output).cpu().numpy()
            size = actions.size
            destination = slice(cursor, cursor + size)
            origin_blocks[destination] = batch["origin_block"].numpy()
            predicted_actions[destination] = actions
            predicted_minimum_z[destination] = output.minimum_fee_z.float().cpu().numpy()
            minimum_actions[destination] = batch["label"].numpy()
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


def read_observations(path: Path) -> dict[str, np.ndarray]:
    return observation_columns(pl.read_parquet(path))


def validate_observations(path: Path) -> None:
    if pl.read_parquet_schema(path) != OBSERVATION_SCHEMA:
        raise ValueError("observations must have the canonical ordered schema")


def observation_columns(observations: pl.DataFrame) -> dict[str, np.ndarray]:
    if observations.schema != OBSERVATION_SCHEMA:
        raise ValueError("observations must have the canonical ordered schema")
    return {name: observations[name].to_numpy() for name in OBSERVATION_SCHEMA}


def reduce_observations(path: Path) -> pl.DataFrame:
    """Derive the shared validation/testing metrics from canonical observations."""

    return reduce_observation_columns(read_observations(path))


def reduce_observation_frame(observations: pl.DataFrame) -> pl.DataFrame:
    return reduce_observation_columns(observation_columns(observations))


def reduce_observation_columns(columns: Mapping[str, np.ndarray]) -> pl.DataFrame:
    log_errors = columns["predicted_minimum_log_base_fee"] - np.log(
        columns["minimum_base_fee_per_gas"]
    )
    immediate = columns["immediate_base_fee_per_gas"]
    selected = columns["selected_base_fee_per_gas"]
    minimum = columns["minimum_base_fee_per_gas"]
    metrics = {
        **classification_metrics(columns["predicted_action_k"], columns["minimum_action_k"]),
        "log_fee_mae": float(np.mean(np.abs(log_errors))),
        "log_fee_mse": float(np.mean(np.square(log_errors))),
        **economic_metrics(columns, "selected"),
        "mean_immediate_base_fee_gwei": float(np.mean(immediate) / _WEI_PER_GWEI),
        "mean_selected_base_fee_gwei": float(np.mean(selected) / _WEI_PER_GWEI),
        "mean_minimum_base_fee_gwei": float(np.mean(minimum) / _WEI_PER_GWEI),
        "mean_selected_minus_minimum_base_fee_gwei": float(
            np.mean(selected - minimum) / _WEI_PER_GWEI
        ),
    }
    return pl.DataFrame([metrics])


def classification_metrics(
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


def economic_metrics(
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
