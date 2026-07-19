"""Context-history sensitivity evidence publication."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast
from uuid import UUID

from spice.addresses import evaluation_json_path
from spice.config import (
    BaselineSource,
    EvaluateRequest,
    Method,
    OriginWindow,
    SelectedStudySource,
    TrainingDefinition,
)
from spice.corpus import Corpus, load_corpus
from spice.evaluation.reduction import reduce_evaluation
from spice.modeling.artifacts import ArtifactAssociation, load_artifact
from spice.study import training_definition_from_method

_CHAIN_IDS = (1, 137, 43_114)
_CONTEXT_BLOCKS = (50, 100, 200, 400)
_FINAL_K_HORIZONS = (2, 3, 4, 5, 10, 15, 30, 50, 100, 200)
_TRAINING_BATCH = 64
_Comparable = TypeVar("_Comparable")

_CHAIN_FEATURE_PREFIXES = {
    1: (
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_exact_forming_base_fee_per_gas",
    ),
    137: ("log_base_fee_per_gas", "gas_utilization"),
    43_114: ("log_base_fee_per_gas", "gas_utilization"),
}

_METRICS = (
    "earliest_hindsight_label_cross_entropy_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae",
    "hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse",
    "multitask_total_loss",
    "earliest_hindsight_label_accuracy",
    "earliest_hindsight_label_macro_f1",
    "finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0",
    "finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0",
    "signed_captured_hindsight_opportunity_ratio",
    "mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0",
    "mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k",
    "harmful_action_rate",
    "mean_extra_wait_block_opportunities_vs_immediate_k0",
    "mean_selected_action_wait_seconds",
    "mean_full_horizon_elapsed_seconds",
)

_COLUMNS = (
    "evaluation_id",
    "artifact_id",
    "corpus_id",
    "chain_id",
    "model_family",
    "context_blocks",
    "horizon_blocks",
    "ordered_features",
    "classification_loss",
    "training_first_parent_block",
    "training_last_parent_block",
    "training_origin_count",
    "training_examples_per_epoch",
    "training_minibatches_per_epoch",
    "training_optimizer_updates_per_epoch",
    "training_context_span_seconds_minimum",
    "training_context_span_seconds_median",
    "training_context_span_seconds_mean",
    "training_context_span_seconds_maximum",
    "validation_first_parent_block",
    "validation_last_parent_block",
    "validation_origin_count",
    "validation_context_span_seconds_minimum",
    "validation_context_span_seconds_median",
    "validation_context_span_seconds_mean",
    "validation_context_span_seconds_maximum",
    "testing_first_parent_block",
    "testing_last_parent_block",
    "testing_origin_count",
    "testing_context_span_seconds_minimum",
    "testing_context_span_seconds_median",
    "testing_context_span_seconds_mean",
    "testing_context_span_seconds_maximum",
    *(name for metric in _METRICS for name in (metric, f"{metric}_delta_vs_same_chain_c200")),
    "final_k_horizon_blocks",
    "final_k_artifact_ids",
)


@dataclass(frozen=True, slots=True)
class _ContextCell:
    chain_id: int
    context_blocks: int
    corpus_id: UUID
    training_window: OriginWindow
    ordered_features: tuple[str, ...]
    metrics: dict[str, float | None]
    row: dict[str, object]


def write_context_history_evidence(
    storage_root: Path,
    context_evaluation_ids: tuple[UUID, ...],
    final_k_artifact_ids: tuple[UUID, ...],
    destination: Path,
) -> None:
    """Publish the fixed context-history sensitivity matrix as one TSV."""

    if len(context_evaluation_ids) != len(_CHAIN_IDS) * len(_CONTEXT_BLOCKS):
        raise ValueError("context evaluation IDs must contain the exact twelve-cell matrix")
    if len(final_k_artifact_ids) != len(_CHAIN_IDS) * len(_FINAL_K_HORIZONS):
        raise ValueError("final-K artifact IDs must contain the exact thirty-cell matrix")

    corpora: dict[UUID, Corpus] = {}
    cells: list[_ContextCell] = []
    cells_by_coordinate: dict[tuple[int, int], _ContextCell] = {}
    chain_corpus_ids: dict[int, UUID] = {}
    chain_validation_windows: dict[int, OriginWindow] = {}
    chain_testing_windows: dict[int, OriginWindow] = {}
    chain_training_ends: dict[int, int] = {}
    selected_family: str | None = None
    selected_feature_route: tuple[str, ...] | None = None
    selected_classification_loss: str | None = None

    for index, evaluation_id in enumerate(context_evaluation_ids):
        expected_chain = _CHAIN_IDS[index // len(_CONTEXT_BLOCKS)]
        expected_context = _CONTEXT_BLOCKS[index % len(_CONTEXT_BLOCKS)]
        request = EvaluateRequest.model_validate_json(
            evaluation_json_path(storage_root, evaluation_id).read_text(encoding="utf-8"),
            strict=True,
        )
        if request.evaluation_id != evaluation_id:
            raise ValueError("context evaluation request ID must match its supplied ID")
        if request.window.role != "testing":
            raise ValueError("context evidence requires testing evaluations")

        reduced = reduce_evaluation(storage_root, evaluation_id).row(0, named=True)
        association, _ = load_artifact(storage_root, request.artifact_id)
        definition = _context_definition(request, association)
        experiment = definition.experiment

        corpus = corpora.get(request.corpus_id)
        if corpus is None:
            corpus = load_corpus(storage_root, request.corpus_id)
            corpora[request.corpus_id] = corpus
        chain_id = corpus.request.definition.chain_id
        if (chain_id, experiment.context_blocks) != (expected_chain, expected_context):
            raise ValueError("context evaluations must use exact chain-major C order")
        if experiment.horizon_blocks != 5:
            raise ValueError("context evaluations must use K=5")

        family = definition.model.family
        feature_route = _feature_route(chain_id, experiment.ordered_features)
        classification_loss = experiment.loss.classification_weighting
        selected_family = _require_same("model family", selected_family, family)
        selected_feature_route = _require_same(
            "ordered feature route",
            selected_feature_route,
            feature_route,
        )
        selected_classification_loss = _require_same(
            "classification loss",
            selected_classification_loss,
            classification_loss,
        )
        previous_corpus_id = chain_corpus_ids.setdefault(chain_id, request.corpus_id)
        if previous_corpus_id != request.corpus_id:
            raise ValueError("each chain must use one context Corpus")
        previous_validation = chain_validation_windows.setdefault(
            chain_id,
            experiment.validation_window,
        )
        if previous_validation != experiment.validation_window:
            raise ValueError("each chain must use one context validation window")
        previous_testing = chain_testing_windows.setdefault(chain_id, request.window)
        if previous_testing != request.window:
            raise ValueError("each chain must use one context testing window")
        previous_training_end = chain_training_ends.setdefault(
            chain_id,
            experiment.training_window.last_parent_block,
        )
        if previous_training_end != experiment.training_window.last_parent_block:
            raise ValueError("each chain must use one context training endpoint")
        expected_training_start = corpus.request.definition.first_block + expected_context - 1
        if experiment.training_window.first_parent_block != expected_training_start:
            raise ValueError("context training windows must use their natural starts")

        training_count = _origin_count(experiment.training_window)
        updates_per_epoch = math.ceil(training_count / _TRAINING_BATCH)
        row: dict[str, object] = {
            "evaluation_id": str(evaluation_id),
            "artifact_id": str(request.artifact_id),
            "corpus_id": str(request.corpus_id),
            "chain_id": chain_id,
            "model_family": family,
            "context_blocks": experiment.context_blocks,
            "horizon_blocks": experiment.horizon_blocks,
            "ordered_features": experiment.ordered_features,
            "classification_loss": classification_loss,
            "training_examples_per_epoch": training_count,
            "training_minibatches_per_epoch": updates_per_epoch,
            "training_optimizer_updates_per_epoch": updates_per_epoch,
            **_period_geometry(corpus, experiment.training_window, expected_context, "training"),
            **_period_geometry(
                corpus,
                experiment.validation_window,
                expected_context,
                "validation",
            ),
            **_period_geometry(corpus, request.window, expected_context, "testing"),
        }
        metrics = {name: cast(float | None, reduced[name]) for name in _METRICS}
        cell = _ContextCell(
            chain_id=chain_id,
            context_blocks=expected_context,
            corpus_id=request.corpus_id,
            training_window=experiment.training_window,
            ordered_features=experiment.ordered_features,
            metrics=metrics,
            row=row,
        )
        cells.append(cell)
        cells_by_coordinate[(chain_id, expected_context)] = cell

    final_ids_by_chain: dict[int, list[UUID]] = {chain_id: [] for chain_id in _CHAIN_IDS}
    for index, artifact_id in enumerate(final_k_artifact_ids):
        expected_chain = _CHAIN_IDS[index // len(_FINAL_K_HORIZONS)]
        expected_horizon = _FINAL_K_HORIZONS[index % len(_FINAL_K_HORIZONS)]
        association, _ = load_artifact(storage_root, artifact_id)
        source = association.request.source
        if not isinstance(source, SelectedStudySource):
            raise ValueError("final-K artifacts must use SelectedStudySource")

        c200 = cells_by_coordinate[(expected_chain, 200)]
        if source.corpus_id != c200.corpus_id:
            raise ValueError("final-K artifacts must use the same-chain C200 Corpus")
        definition = training_definition_from_method(
            source.experiment,
            cast(Method, association.method),
        )
        experiment = definition.experiment
        if (experiment.context_blocks, experiment.horizon_blocks) != (200, expected_horizon):
            raise ValueError("final-K artifacts must use exact chain-major K order at C200")
        if experiment.training_window != c200.training_window:
            raise ValueError("final-K training windows must match the same-chain C200 window")
        if definition.model.family != selected_family:
            raise ValueError("final-K artifacts must use the globally selected family")
        if experiment.ordered_features != c200.ordered_features:
            raise ValueError("final-K artifacts must use the same-chain selected features")
        if experiment.loss.classification_weighting != selected_classification_loss:
            raise ValueError("final-K artifacts must use the globally selected classification loss")
        final_ids_by_chain[expected_chain].append(artifact_id)

    for cell in cells:
        baseline = cells_by_coordinate[(cell.chain_id, 200)]
        for metric in _METRICS:
            value = cell.metrics[metric]
            reference = baseline.metrics[metric]
            cell.row[metric] = value
            cell.row[f"{metric}_delta_vs_same_chain_c200"] = (
                None if value is None or reference is None else value - reference
            )
        if cell.context_blocks == 200:
            cell.row["final_k_horizon_blocks"] = _FINAL_K_HORIZONS
            cell.row["final_k_artifact_ids"] = tuple(
                str(artifact_id) for artifact_id in final_ids_by_chain[cell.chain_id]
            )
        else:
            cell.row["final_k_horizon_blocks"] = ()
            cell.row["final_k_artifact_ids"] = ()

    hidden = destination.with_name(f".{destination.name}")
    if destination.exists():
        raise FileExistsError(destination)
    if hidden.exists():
        raise FileExistsError(hidden)
    with hidden.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for cell in cells:
            writer.writerow([_tsv_value(cell.row[name]) for name in _COLUMNS])
    hidden.rename(destination)


def _context_definition(
    request: EvaluateRequest,
    association: ArtifactAssociation,
) -> TrainingDefinition:
    source = association.request.source
    if source.corpus_id != request.corpus_id:
        raise ValueError("context artifact Corpus must match the evaluation Corpus")
    if not isinstance(source, BaselineSource):
        raise ValueError("context artifacts must use BaselineSource")
    return source.training_definition


def _feature_route(chain_id: int, ordered_features: tuple[str, ...]) -> tuple[str, ...]:
    prefix = _CHAIN_FEATURE_PREFIXES[chain_id]
    if ordered_features[: len(prefix)] != prefix:
        raise ValueError("context artifacts must use the accepted chain feature prefix")
    return ordered_features[len(prefix) :]


def _require_same(
    label: str,
    expected: _Comparable | None,
    actual: _Comparable,
) -> _Comparable:
    if expected is not None and expected != actual:
        raise ValueError(f"all context artifacts must use one {label}")
    return actual


def _origin_count(window: OriginWindow) -> int:
    return window.last_parent_block - window.first_parent_block + 1


def _period_geometry(
    corpus: Corpus,
    window: OriginWindow,
    context_blocks: int,
    role: str,
) -> dict[str, int | float]:
    definition = corpus.request.definition
    first_offset = window.first_parent_block - definition.first_block
    oldest_offset = first_offset - context_blocks + 1
    count = _origin_count(window)

    timestamps = corpus.blocks["timestamp"]
    spans = timestamps.slice(first_offset, count) - timestamps.slice(oldest_offset, count)
    return {
        f"{role}_first_parent_block": window.first_parent_block,
        f"{role}_last_parent_block": window.last_parent_block,
        f"{role}_origin_count": count,
        f"{role}_context_span_seconds_minimum": int(cast(int, spans.min())),
        f"{role}_context_span_seconds_median": float(cast(float, spans.median())),
        f"{role}_context_span_seconds_mean": float(cast(float, spans.mean())),
        f"{role}_context_span_seconds_maximum": int(cast(int, spans.max())),
    }


def _tsv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return json.dumps(value, separators=(",", ":"))
    return value


__all__ = ["write_context_history_evidence"]
