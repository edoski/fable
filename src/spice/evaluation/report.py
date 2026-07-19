"""Ordered publication of sealed evaluation facts."""

from __future__ import annotations

import json
import typing
from pathlib import Path
from uuid import UUID

import polars as pl

from ..addresses import evaluation_json_path
from ..config import BaselineSource, EvaluateRequest, Method, SelectedStudySource
from ..corpus import Corpus, load_corpus
from ..modeling.artifacts import load_artifact
from ..study import training_definition_from_method
from .reduction import reduce_evaluation

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

    corpora: dict[UUID, Corpus] = {}
    rows: list[pl.DataFrame] = []
    for evaluation_id in evaluation_ids:
        request = EvaluateRequest.model_validate_json(
            evaluation_json_path(storage_root, evaluation_id).read_text(encoding="utf-8"),
            strict=True,
        )
        if request.evaluation_id != evaluation_id:
            raise ValueError("evaluation request ID must match the requested evaluation")
        if request.window.role != "testing":
            raise ValueError("sealed report evaluations must use testing windows")

        reduced = reduce_evaluation(storage_root, evaluation_id)
        association, model = load_artifact(storage_root, request.artifact_id)
        source = association.request.source
        if source.corpus_id != request.corpus_id:
            raise ValueError("artifact source Corpus must match the evaluation Corpus")

        match source:
            case BaselineSource():
                definition = source.training_definition
                study_id = None
                study_result_index = None
            case SelectedStudySource():
                definition = training_definition_from_method(
                    source.experiment,
                    typing.cast(Method, association.method),
                )
                study_id = source.study_id
                study_result_index = association.study_result_index

        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        if request.corpus_id not in corpora:
            corpora[request.corpus_id] = load_corpus(storage_root, request.corpus_id)
        corpus = corpora[request.corpus_id]
        corpus_definition = corpus.request.definition
        first_parent_block = request.window.first_parent_block
        last_parent_block = request.window.last_parent_block
        corpus_endpoint_block = corpus_definition.last_block
        timestamp_column = corpus.blocks["timestamp"]
        first_offset = first_parent_block - corpus_definition.first_block
        last_offset = last_parent_block - corpus_definition.first_block
        testing_elapsed_seconds = int(timestamp_column[last_offset]) - int(
            timestamp_column[first_offset]
        )
        experiment = definition.experiment

        context = pl.DataFrame(
            [
                (
                    str(evaluation_id),
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
                    trainable_parameter_count,
                )
            ],
            schema=_CONTEXT_SCHEMA,
            orient="row",
        )
        rows.append(pl.concat([context, reduced[:, 1:]], how="horizontal"))

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
