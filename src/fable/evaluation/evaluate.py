"""Direct evaluation of one native artifact over one historical window."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader

from ..addresses import evaluation_directory
from ..config import BaselineSource, EvaluateRequest, ExperimentSemantics
from ..corpus import BlockFrame, load_corpus
from ..min_block_fee import decode_action, min_block_fee_loss
from ..modeling import ArtifactAssociation, load_artifact
from ..temporal.history import HistoricalDataset, prepare_historical_window

_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_DEVICE = torch.device("cuda:0")

_OBSERVATION_SCHEMA = pl.Schema(
    {
        "origin_block": pl.Int64,
        "origin_timestamp": pl.Int64,
        "selected_action_k": pl.Int64,
        "earliest_hindsight_action_k": pl.Int64,
        "classification_loss_contribution": pl.Float64,
        "predicted_hindsight_minimum_base_fee_z": pl.Float32,
        "previous_closed_parent_base_fee_per_gas": pl.Int64,
        "closed_parent_base_fee_per_gas": pl.Int64,
        "immediate_k0_base_fee_per_gas": pl.Int64,
        "selected_target_base_fee_per_gas": pl.Int64,
        "hindsight_minimum_base_fee_per_gas": pl.Int64,
        "selected_action_wait_seconds": pl.Int64,
        "full_horizon_elapsed_seconds": pl.Int64,
    }
)


class EvaluationDeployment(BaseModel):
    """Non-scientific execution facts for one evaluation invocation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )

    batch_size: _PositiveInt
    num_workers: _NonNegativeInt
    pin_memory: bool
    prefetch_factor: _PositiveInt | None
    persistent_workers: bool
    deterministic: bool | Literal["warn"]
    benchmark: bool
    float32_matmul_precision: Literal["highest", "high"]
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool


def evaluate(
    request: EvaluateRequest,
    storage_root: Path,
    deployment: EvaluationDeployment,
) -> None:
    """Publish canonical observations for one exact artifact/window request."""

    scratch = storage_root / "evaluations" / f".{request.evaluation_id}"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir()

    corpus = load_corpus(storage_root, request.corpus_id)
    association, model = load_artifact(storage_root, request.artifact_id)
    source = association.request.source
    if source.corpus_id != request.corpus_id:
        raise ValueError("artifact source Corpus must match the evaluation Corpus")
    experiment = (
        source.training_definition.experiment
        if isinstance(source, BaselineSource)
        else source.experiment
    )
    if request.window.first_parent_block <= corpus.request.definition.first_block:
        raise ValueError("Corpus must include the previous closed parent")
    dataset = prepare_historical_window(
        corpus,
        experiment,
        request.window,
        feature_state=association.feature_state,
        target_state=association.target_state,
    )

    _configure_execution(deployment)
    observations = _collect_observations(
        corpus.blocks.select_range(
            request.window.first_parent_block - 1,
            request.window.last_parent_block + experiment.horizon_blocks,
        ),
        dataset,
        experiment,
        association,
        model,
        deployment,
    )
    (scratch / "evaluation.json").write_text(
        request.model_dump_json(),
        encoding="utf-8",
    )
    observations.write_parquet(scratch / "observations.parquet")

    canonical = evaluation_directory(storage_root, request.evaluation_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    scratch.rename(canonical)


def _configure_execution(deployment: EvaluationDeployment) -> None:
    torch.use_deterministic_algorithms(
        deployment.deterministic is not False,
        warn_only=deployment.deterministic == "warn",
    )
    torch.backends.cudnn.deterministic = deployment.deterministic is not False
    torch.backends.cudnn.benchmark = deployment.benchmark
    torch.set_float32_matmul_precision(deployment.float32_matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = deployment.cuda_matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = deployment.cudnn_allow_tf32


def _collect_observations(
    blocks: BlockFrame,
    dataset: HistoricalDataset,
    experiment: ExperimentSemantics,
    association: ArtifactAssociation,
    model: nn.Module,
    deployment: EvaluationDeployment,
) -> pl.DataFrame:
    frame = blocks.to_polars()
    first_block = blocks.definition.first_block
    count = len(dataset)
    origin_blocks = np.empty(count, dtype=np.int64)
    selected_actions = np.empty(count, dtype=np.int64)
    hindsight_actions = np.empty(count, dtype=np.int64)
    classification = np.empty(count, dtype=np.float64)
    predicted_z = np.empty(count, dtype=np.float32)
    immediate_fees = np.empty(count, dtype=np.int64)
    selected_fees = np.empty(count, dtype=np.int64)
    hindsight_fees = np.empty(count, dtype=np.int64)

    loader = DataLoader(
        dataset,
        batch_size=deployment.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=deployment.num_workers,
        pin_memory=deployment.pin_memory,
        prefetch_factor=deployment.prefetch_factor,
        persistent_workers=deployment.persistent_workers,
    )
    model.to(_DEVICE)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["inputs"].to(_DEVICE)
            labels = batch["label"].to(_DEVICE)
            targets = batch["target"].to(_DEVICE)
            output = model(inputs)
            loss = min_block_fee_loss(
                output,
                label=labels,
                target=targets,
                loss_definition=experiment.loss,
                classification_state=association.classification_state,
            )
            actions = decode_action(output).to(device="cpu", dtype=torch.int64)
            contributions = loss.classification_by_origin.to(
                device="cpu",
                dtype=torch.float64,
            )
            minimum_fee_z = output.minimum_fee_z.to(device="cpu", dtype=torch.float32)

            size = actions.shape[0]
            destination = slice(cursor, cursor + size)
            outcomes = batch["base_fees"].numpy()
            cpu_labels = batch["label"].numpy()
            rows = np.arange(size)
            origin_blocks[destination] = batch["origin_block"].numpy()
            selected_actions[destination] = actions.numpy()
            hindsight_actions[destination] = cpu_labels
            classification[destination] = contributions.numpy()
            predicted_z[destination] = minimum_fee_z.numpy()
            immediate_fees[destination] = outcomes[:, 0]
            selected_fees[destination] = outcomes[rows, actions.numpy()]
            hindsight_fees[destination] = outcomes[rows, cpu_labels]
            cursor += size

    base_fees = cast(np.ndarray, frame["base_fee_per_gas"].to_numpy())
    timestamps = cast(np.ndarray, frame["timestamp"].to_numpy())
    origin_rows = origin_blocks - first_block
    return pl.DataFrame(
        {
            "origin_block": origin_blocks,
            "origin_timestamp": timestamps[origin_rows],
            "selected_action_k": selected_actions,
            "earliest_hindsight_action_k": hindsight_actions,
            "classification_loss_contribution": classification,
            "predicted_hindsight_minimum_base_fee_z": predicted_z,
            "previous_closed_parent_base_fee_per_gas": base_fees[origin_rows - 1],
            "closed_parent_base_fee_per_gas": base_fees[origin_rows],
            "immediate_k0_base_fee_per_gas": immediate_fees,
            "selected_target_base_fee_per_gas": selected_fees,
            "hindsight_minimum_base_fee_per_gas": hindsight_fees,
            "selected_action_wait_seconds": (
                timestamps[origin_rows + selected_actions] - timestamps[origin_rows]
            ),
            "full_horizon_elapsed_seconds": (
                timestamps[origin_rows + experiment.horizon_blocks] - timestamps[origin_rows]
            ),
        },
        schema=_OBSERVATION_SCHEMA,
    )


__all__ = ["EvaluationDeployment", "evaluate"]
