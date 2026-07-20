"""Ordered publication of sealed evaluation facts."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import polars as pl

from ..config import BaselineSource
from .resolution import resolve_evaluations

_CONTEXT_SCHEMA = pl.Schema(
    {
        "evaluation_id": pl.String,
        "artifact_id": pl.String,
        "corpus_id": pl.String,
        "chain_id": pl.Int64,
        "window_role": pl.String,
        "first_parent_block": pl.Int64,
        "last_parent_block": pl.Int64,
        "corpus_endpoint_block": pl.Int64,
        "testing_candidate_origin_count": pl.Int64,
        "testing_incomplete_kmax_outcome_exclusion_count": pl.Int64,
        "testing_elapsed_seconds": pl.Int64,
        "source_kind": pl.String,
        "study_id": pl.String,
        "study_result_index": pl.Int64,
        "model_family": pl.String,
        "context_blocks": pl.Int64,
        "horizon_blocks": pl.Int64,
        "ordered_features": pl.List(pl.String),
        "classification_loss": pl.String,
        "trainable_parameter_count": pl.Int64,
    }
)


def write_sealed_report(
    storage_root: Path,
    evaluation_ids: tuple[UUID, ...],
    destination: Path,
) -> None:
    """Compose and publish one ordered sealed-evaluation TSV."""

    if not evaluation_ids:
        raise ValueError("evaluation IDs must not be empty")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("evaluation IDs must not contain duplicates")

    rows: list[pl.DataFrame] = []
    resolved_evaluations = resolve_evaluations(storage_root, evaluation_ids)
    for resolved in resolved_evaluations:
        request = resolved.request
        if request.window.role != "testing":
            raise ValueError("sealed report evaluations must use testing windows")

        source = resolved.training_source
        definition = resolved.training_definition
        if isinstance(source, BaselineSource):
            study_id = None
            study_result_index = None
        else:
            study_id = source.study_id
            study_result_index = source.study_result_index

        corpus = resolved.corpus
        corpus_definition = corpus.request.definition
        first_parent_block = request.window.first_parent_block
        last_parent_block = request.window.last_parent_block
        corpus_endpoint_block = corpus_definition.last_block
        timestamps = corpus.blocks.select_range(
            first_parent_block,
            last_parent_block,
        ).to_polars()["timestamp"]
        testing_elapsed_seconds = int(timestamps[-1]) - int(timestamps[0])
        experiment = definition.experiment

        context = pl.DataFrame(
            [
                (
                    str(request.evaluation_id),
                    str(request.artifact_id),
                    str(request.corpus_id),
                    corpus_definition.chain_id,
                    request.window.role,
                    first_parent_block,
                    last_parent_block,
                    corpus_endpoint_block,
                    corpus_endpoint_block - first_parent_block + 1,
                    corpus_endpoint_block - last_parent_block,
                    testing_elapsed_seconds,
                    source.kind,
                    None if study_id is None else str(study_id),
                    study_result_index,
                    definition.model.family,
                    experiment.context_blocks,
                    experiment.horizon_blocks,
                    list(experiment.ordered_features),
                    experiment.loss.classification_weighting,
                    resolved.trainable_parameter_count,
                )
            ],
            schema=_CONTEXT_SCHEMA,
            orient="row",
        )
        rows.append(
            pl.concat([context, resolved.reduction.drop("evaluation_id")], how="horizontal")
        )

    hidden = destination.with_name(f".{destination.name}")
    if destination.exists():
        raise FileExistsError(destination)
    if hidden.exists():
        raise FileExistsError(hidden)

    report = pl.concat(rows, how="vertical")
    report.with_columns(
        pl.col("ordered_features", "selected_action_count_by_k").map_elements(
            lambda values: json.dumps(values.to_list(), separators=(",", ":")),
            return_dtype=pl.String,
        )
    ).write_csv(hidden, separator="\t", null_value="")
    hidden.rename(destination)


__all__ = ["write_sealed_report"]
