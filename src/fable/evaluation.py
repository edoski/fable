"""Native artifact evaluation and scientific reduction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl
import torch

from . import _runtime
from .addresses import evaluation_directory, evaluation_observations_path
from .config import EvaluateRequest
from .corpus import load_corpus_blocks
from .modeling import load_artifact
from .observations import (
    collect_observations,
    economic_metrics,
    read_observations,
    reduce_observations,
)
from .temporal import prepare_historical_window

_DEVICE = torch.device("cuda:0")
ROLLING_HORIZONS = (5, 4, 3, 2)


def evaluate(request: EvaluateRequest, storage_root: Path) -> None:
    """Publish canonical observations for one exact artifact/window request."""

    scratch = storage_root / "evaluations" / f".{request.evaluation_id}"
    scratch.mkdir(parents=True)

    blocks = load_corpus_blocks(storage_root, request.corpus_id)
    association, model = load_artifact(storage_root, request.artifact_id)
    if association.request.source.corpus_id != request.corpus_id:
        raise ValueError("artifact source Corpus must match the evaluation Corpus")
    experiment = association.training_definition.experiment
    dataset = prepare_historical_window(
        blocks,
        experiment,
        request.testing_window,
        feature_state=association.feature_state,
        target_state=association.target_state,
    )

    _runtime.configure_torch()
    observations = collect_observations(
        dataset,
        model,
        blocks,
        request.testing_window,
        target_state=association.target_state,
        horizon_blocks=experiment.horizon_blocks,
        device=_DEVICE,
        batch_size=_runtime.EVALUATION_BATCH_SIZE,
    )
    (scratch / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")
    observations.write_parquet(scratch / "observations.parquet")

    canonical = evaluation_directory(storage_root, request.evaluation_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    scratch.rename(canonical)


def reduce_evaluation(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive one testing evaluation's shared metrics from its observations."""

    return reduce_observations(evaluation_observations_path(storage_root, evaluation_id))


def reduce_baselines(storage_root: Path, evaluation_id: UUID) -> pl.DataFrame:
    """Derive immediate and deadline policy metrics from one testing evaluation."""

    columns = read_observations(evaluation_observations_path(storage_root, evaluation_id))
    rows = []
    for policy in ("immediate", "deadline"):
        rows.append({"policy": policy, **economic_metrics(columns, policy)})
    return pl.DataFrame(rows)


def reduce_rolling(storage_root: Path, roster: Mapping[str, Mapping[int, UUID]]) -> pl.DataFrame:
    """Compare one-shot and rolling economics for explicit K-study cells."""

    rows = [
        _reduce_rolling_cell(storage_root, cell, evaluation_ids)
        for cell, evaluation_ids in roster.items()
    ]
    return pl.DataFrame(rows)


def _reduce_rolling_cell(
    storage_root: Path, cell: str, evaluation_ids: Mapping[int, UUID]
) -> dict[str, str | float]:
    decision_origins: np.ndarray | None = None
    selections = []
    for horizon in ROLLING_HORIZONS:
        columns = _load_rolling_observations(storage_root, evaluation_ids[horizon])
        if decision_origins is None:
            decision_origins = columns["origin_block"].copy()
        selection = _rolling_arrays(
            columns, decision_origins=decision_origins, cell=cell, horizon=horizon
        )
        selections.append(selection)
        if horizon != ROLLING_HORIZONS[-1]:
            decision_origins += selection["predicted_action_k"] == horizon - 1

    initial = selections[0]
    final = selections[-1]
    one_shot = economic_metrics(initial, "selected")
    rolling = economic_metrics(initial, "selected", selected=final)
    metrics = {}
    for name in one_shot:
        metrics[f"one_shot_{name}"] = one_shot[name]
        metrics[f"rolling_{name}"] = rolling[name]
    return {"cell": cell, **metrics}


def _load_rolling_observations(storage_root: Path, evaluation_id: UUID) -> dict[str, np.ndarray]:
    columns = read_observations(evaluation_observations_path(storage_root, evaluation_id))
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
