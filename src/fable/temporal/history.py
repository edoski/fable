"""Exact historical windows over accepted canonical block rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..config import ExperimentSemantics, OriginWindow
from ..corpus.contract import Corpus
from ..min_block_fee import (
    ClassificationLossState,
    TargetState,
    fit_classification_loss_state,
    fit_target_state,
    standardize_target,
)
from .features import FeatureState, fit_feature_state, transform_feature_rows

__all__ = [
    "HistoricalDataset",
    "HistoricalPreparation",
    "prepare_fit_history",
    "prepare_historical_window",
]

_OUTCOME_CHUNK_SIZE = 4_096
_HistoricalItem = dict[str, torch.Tensor]
_IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _HistoricalBacking:
    first_block: int
    inputs: torch.Tensor
    base_fees: torch.Tensor
    block_numbers: torch.Tensor


class HistoricalDataset(Dataset[_HistoricalItem]):
    """Lazy fixed-context dataset backed by contiguous CPU row tensors."""

    def __init__(
        self,
        backing: _HistoricalBacking,
        origin_rows: _IntVector,
        labels: _IntVector,
        targets: NDArray[np.float32],
        *,
        context_blocks: int,
        horizon_blocks: int,
    ) -> None:
        self._backing = backing
        self._origin_rows = torch.from_numpy(origin_rows)
        self._labels = torch.from_numpy(labels)
        self._targets = torch.from_numpy(targets)
        self._context_blocks = context_blocks
        self._horizon_blocks = horizon_blocks

    def __len__(self) -> int:
        return self._origin_rows.numel()

    def __getitem__(self, index: int) -> _HistoricalItem:
        if index < 0 or index >= len(self):
            raise IndexError("HistoricalDataset index out of range")
        origin = int(self._origin_rows[index])
        return {
            "inputs": self._backing.inputs[origin - self._context_blocks + 1 : origin + 1],
            "label": self._labels[index],
            "target": self._targets[index],
            "base_fees": self._backing.base_fees[origin + 1 : origin + 1 + self._horizon_blocks],
            "origin_block": self._backing.block_numbers[origin],
        }


@dataclass(frozen=True, slots=True)
class HistoricalPreparation:
    training: HistoricalDataset
    validation: HistoricalDataset
    feature_state: FeatureState
    target_state: TargetState
    classification_state: ClassificationLossState | None


def prepare_fit_history(
    corpus: Corpus,
    experiment: ExperimentSemantics,
) -> HistoricalPreparation:
    """Fit training-only state and prepare the authored fit windows."""

    training_window = experiment.training_window
    validation_window = experiment.validation_window
    _require_complete_support(corpus, experiment, training_window)
    _require_complete_support(corpus, experiment, validation_window)

    training_support = corpus.blocks.select_range(
        training_window.first_parent_block - experiment.context_blocks + 1,
        training_window.last_parent_block,
    )
    feature_state = fit_feature_state(
        training_support,
        ordered_features=experiment.ordered_features,
    )

    backing = _build_backing(
        corpus,
        first_block=training_window.first_parent_block - experiment.context_blocks + 1,
        last_block=validation_window.last_parent_block + experiment.horizon_blocks,
        ordered_features=experiment.ordered_features,
        feature_state=feature_state,
    )
    training_origins = _origin_rows(backing, training_window)
    training_labels, training_minima = _minimum_outcomes(
        backing.base_fees.numpy(),
        training_origins,
        horizon_blocks=experiment.horizon_blocks,
    )
    target_state = fit_target_state(training_minima)
    classification_state = fit_classification_loss_state(
        training_labels,
        horizon_blocks=experiment.horizon_blocks,
        loss_definition=experiment.loss,
    )

    return HistoricalPreparation(
        training=_build_dataset(
            backing,
            experiment,
            training_window,
            target_state,
            outcomes=(training_labels, training_minima),
        ),
        validation=_build_dataset(
            backing,
            experiment,
            validation_window,
            target_state,
        ),
        feature_state=feature_state,
        target_state=target_state,
        classification_state=classification_state,
    )


def prepare_historical_window(
    corpus: Corpus,
    experiment: ExperimentSemantics,
    window: OriginWindow,
    *,
    feature_state: FeatureState,
    target_state: TargetState,
) -> HistoricalDataset:
    """Prepare one validation or testing window without fitting state."""

    if window.role == "training":
        raise ValueError("prepare_historical_window accepts validation or testing windows")
    if window.role == "validation" and window != experiment.validation_window:
        raise ValueError("validation window must match experiment.validation_window")
    if (
        window.role == "testing"
        and experiment.validation_window.last_parent_block + experiment.horizon_blocks
        >= window.first_parent_block
    ):
        raise ValueError("testing window must follow complete validation outcomes")
    _require_complete_support(corpus, experiment, window)
    backing = _build_backing(
        corpus,
        first_block=window.first_parent_block - experiment.context_blocks + 1,
        last_block=window.last_parent_block + experiment.horizon_blocks,
        ordered_features=experiment.ordered_features,
        feature_state=feature_state,
    )
    return _build_dataset(backing, experiment, window, target_state)


def _require_complete_support(
    corpus: Corpus,
    experiment: ExperimentSemantics,
    window: OriginWindow,
) -> None:
    required_first = window.first_parent_block - experiment.context_blocks + 1
    required_last = window.last_parent_block + experiment.horizon_blocks
    available = corpus.request.definition
    if required_first < available.first_block or required_last > available.last_block:
        raise ValueError("Corpus must provide complete context and outcome support")


def _build_backing(
    corpus: Corpus,
    *,
    first_block: int,
    last_block: int,
    ordered_features: tuple[str, ...],
    feature_state: FeatureState,
) -> _HistoricalBacking:
    blocks = corpus.blocks.select_range(first_block, last_block)
    inputs = transform_feature_rows(
        blocks,
        ordered_features=ordered_features,
        state=feature_state,
    )
    frame = blocks.to_polars()
    base_fees = np.array(
        frame["base_fee_per_gas"].to_numpy(),
        dtype=np.int64,
        copy=True,
        order="C",
    )
    block_numbers = np.array(
        frame["block_number"].to_numpy(),
        dtype=np.int64,
        copy=True,
        order="C",
    )
    return _HistoricalBacking(
        first_block=first_block,
        inputs=torch.from_numpy(inputs),
        base_fees=torch.from_numpy(base_fees),
        block_numbers=torch.from_numpy(block_numbers),
    )


def _origin_rows(backing: _HistoricalBacking, window: OriginWindow) -> _IntVector:
    return np.arange(
        window.first_parent_block - backing.first_block,
        window.last_parent_block - backing.first_block + 1,
        dtype=np.int64,
    )


def _minimum_outcomes(
    base_fees: _IntVector,
    origin_rows: _IntVector,
    *,
    horizon_blocks: int,
) -> tuple[_IntVector, _IntVector]:
    labels = np.empty(origin_rows.size, dtype=np.int64)
    minima = np.empty(origin_rows.size, dtype=np.int64)
    offsets = np.arange(1, horizon_blocks + 1, dtype=np.int64)
    for start in range(0, origin_rows.size, _OUTCOME_CHUNK_SIZE):
        stop = min(start + _OUTCOME_CHUNK_SIZE, origin_rows.size)
        outcomes = base_fees[origin_rows[start:stop, None] + offsets]
        chunk_labels = outcomes.argmin(axis=1).astype(np.int64, copy=False)
        labels[start:stop] = chunk_labels
        minima[start:stop] = outcomes[np.arange(stop - start), chunk_labels]
    return labels, minima


def _build_dataset(
    backing: _HistoricalBacking,
    experiment: ExperimentSemantics,
    window: OriginWindow,
    target_state: TargetState,
    *,
    outcomes: tuple[_IntVector, _IntVector] | None = None,
) -> HistoricalDataset:
    origin_rows = _origin_rows(backing, window)
    labels, minima = (
        _minimum_outcomes(
            backing.base_fees.numpy(),
            origin_rows,
            horizon_blocks=experiment.horizon_blocks,
        )
        if outcomes is None
        else outcomes
    )
    return HistoricalDataset(
        backing,
        origin_rows,
        labels,
        standardize_target(minima, target_state),
        context_blocks=experiment.context_blocks,
        horizon_blocks=experiment.horizon_blocks,
    )
