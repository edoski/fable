"""Direct evaluation of one native artifact over one historical window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import _runtime
from ..addresses import evaluation_directory
from ..config import EvaluateRequest
from ..corpus import load_corpus
from ..min_block_fee import TargetState, decode_action
from ..modeling import load_artifact
from ..temporal.history import HistoricalDataset, prepare_historical_window
from .contract import OBSERVATION_SCHEMA

_DEVICE = torch.device("cuda:0")


def evaluate(
    request: EvaluateRequest,
    storage_root: Path,
) -> None:
    """Publish canonical observations for one exact artifact/window request."""

    scratch = storage_root / "evaluations" / f".{request.evaluation_id}"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir()

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

    _configure_execution()
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


def _configure_execution() -> None:
    torch.use_deterministic_algorithms(_runtime.DETERMINISTIC)
    torch.backends.cudnn.deterministic = _runtime.DETERMINISTIC
    torch.backends.cudnn.benchmark = _runtime.BENCHMARK
    torch.set_float32_matmul_precision(_runtime.FLOAT32_MATMUL_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = _runtime.CUDA_MATMUL_ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = _runtime.CUDNN_ALLOW_TF32


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
        "origin_block": np.empty(count, dtype=np.int64),
        "predicted_action_k": np.empty(count, dtype=np.int64),
        "predicted_minimum_log_base_fee": np.empty(count, dtype=np.float64),
        "minimum_action_k": np.empty(count, dtype=np.int64),
        "immediate_base_fee_per_gas": np.empty(count, dtype=np.int64),
        "immediate_effective_priority_fee_per_gas_p50": np.empty(count, dtype=np.int64),
        "selected_base_fee_per_gas": np.empty(count, dtype=np.int64),
        "selected_effective_priority_fee_per_gas_p50": np.empty(count, dtype=np.int64),
        "minimum_base_fee_per_gas": np.empty(count, dtype=np.int64),
    }

    workers = _runtime.NUM_WORKERS
    loader = DataLoader(
        dataset,
        batch_size=_runtime.EVALUATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=_runtime.PIN_MEMORY,
        prefetch_factor=_runtime.PREFETCH_FACTOR if workers else None,
        persistent_workers=_runtime.PERSISTENT_WORKERS if workers else False,
    )
    model.to(_DEVICE)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["inputs"].to(_DEVICE))
            actions = decode_action(output).to(device="cpu", dtype=torch.int64).numpy()
            minimum_actions_batch = batch["label"].to(dtype=torch.int64).numpy()
            base_fees = batch["base_fees"].to(dtype=torch.int64).numpy()
            minimum_fee_z = output.minimum_fee_z

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
                minimum_fee_z.to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
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


__all__ = ["evaluate"]
