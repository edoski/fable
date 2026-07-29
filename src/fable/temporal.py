"""Exact causal features and historical windows."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator
from torch.utils.data import DataLoader, Dataset

from .config import BlockWindow, ExperimentSemantics, FeatureName
from .corpus import BlockFrame, Corpus
from .min_block_fee import TargetState, fit_target_state, standardize_target
from .records import StrictFrozenRecord

_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_PositiveFiniteFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class FeatureState(StrictFrozenRecord):
    means: Annotated[tuple[_FiniteFloat, ...], Field(min_length=1)]
    standard_deviations: Annotated[
        tuple[_PositiveFiniteFloat, ...],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def validate_widths(self) -> Self:
        if len(self.means) != len(self.standard_deviations):
            raise ValueError("means and standard_deviations must have equal widths")
        return self


def fit_feature_state(
    training_support: BlockFrame,
    *,
    ordered_features: tuple[FeatureName, ...],
) -> FeatureState:
    raw = _raw_feature_rows(
        training_support.to_polars(),
        chain_id=training_support.definition.chain_id,
        ordered_features=ordered_features,
    )
    means = raw.mean(axis=0, dtype=np.float64)
    standard_deviations = raw.std(axis=0, ddof=0, dtype=np.float64)
    return FeatureState(
        means=tuple(float(value) for value in means),
        standard_deviations=tuple(float(value) for value in standard_deviations),
    )


def transform_feature_rows(
    blocks: BlockFrame,
    *,
    ordered_features: tuple[FeatureName, ...],
    state: FeatureState,
) -> NDArray[np.float32]:
    raw = _raw_feature_rows(
        blocks.to_polars(),
        chain_id=blocks.definition.chain_id,
        ordered_features=ordered_features,
    )
    means = np.asarray(state.means, dtype=np.float64)
    standard_deviations = np.asarray(state.standard_deviations, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.ascontiguousarray(
            (raw - means) / standard_deviations,
            dtype=np.float32,
        )
    if not np.isfinite(transformed).all():
        raise ValueError("transformed features must be finite float32 values")
    return transformed


def _raw_feature_rows(
    blocks: pl.DataFrame,
    *,
    chain_id: int,
    ordered_features: tuple[FeatureName, ...],
) -> NDArray[np.float64]:
    needs_predecessor = "block_interval_seconds" in ordered_features
    columns = []
    for feature_name in ordered_features:
        values = _feature_values(blocks, chain_id, feature_name)
        if needs_predecessor and feature_name != "block_interval_seconds":
            values = values[1:]
        columns.append(values)
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def _feature_values(
    blocks: pl.DataFrame,
    chain_id: int,
    feature_name: FeatureName,
) -> NDArray[np.float64]:
    match feature_name:
        case "log_base_fee_per_gas":
            return np.log(_float_column(blocks, "base_fee_per_gas"))
        case "gas_utilization":
            return _float_column(blocks, "gas_used") / _float_column(blocks, "gas_limit")
        case "log_exact_forming_base_fee_per_gas":
            if chain_id != 1:
                raise ValueError("log_exact_forming_base_fee_per_gas is Ethereum-only")
            return _forming_base_fee_logs(blocks)
        case "log_gas_limit":
            return np.log(_float_column(blocks, "gas_limit"))
        case "log1p_tx_count":
            return np.log1p(_float_column(blocks, "tx_count"))
        case "log1p_effective_priority_fee_per_gas_p50":
            return np.log1p(_float_column(blocks, "effective_priority_fee_per_gas_p50"))
        case "log1p_effective_priority_fee_per_gas_p90":
            return np.log1p(_float_column(blocks, "effective_priority_fee_per_gas_p90"))
        case "block_interval_seconds":
            timestamps = blocks["timestamp"].to_numpy().astype(np.int64, copy=False)
            intervals = np.diff(timestamps)
            return intervals.astype(np.float64, copy=False)
        case "hour_sin":
            return np.sin(_hour_angles(blocks))
        case "hour_cos":
            return np.cos(_hour_angles(blocks))
        case "dow_sin":
            return np.sin(_day_of_week_angles(blocks))
        case "dow_cos":
            return np.cos(_day_of_week_angles(blocks))


def _hour_angles(blocks: pl.DataFrame) -> NDArray[np.float64]:
    timestamps = blocks["timestamp"].to_numpy().astype(np.int64, copy=False)
    hours = (timestamps // 3_600) % 24
    return 2.0 * math.pi * hours.astype(np.float64, copy=False) / 24.0


def _day_of_week_angles(blocks: pl.DataFrame) -> NDArray[np.float64]:
    timestamps = blocks["timestamp"].to_numpy().astype(np.int64, copy=False)
    days = (timestamps // 86_400 + 4) % 7
    return 2.0 * math.pi * days.astype(np.float64, copy=False) / 7.0


def _forming_base_fee_logs(blocks: pl.DataFrame) -> NDArray[np.float64]:
    columns = [blocks[name].to_list() for name in ("base_fee_per_gas", "gas_used", "gas_limit")]
    return np.fromiter(
        (math.log(_forming_child_base_fee(*row)) for row in zip(*columns, strict=True)),
        dtype=np.float64,
        count=blocks.height,
    )


def _forming_child_base_fee(
    base_fee_per_gas: int,
    gas_used: int,
    gas_limit: int,
) -> int:
    gas_target = gas_limit // 2
    if gas_used == gas_target:
        return base_fee_per_gas
    if gas_used > gas_target:
        return base_fee_per_gas + max(
            base_fee_per_gas * (gas_used - gas_target) // gas_target // 8,
            1,
        )
    return base_fee_per_gas - (base_fee_per_gas * (gas_target - gas_used) // gas_target // 8)


def _float_column(blocks: pl.DataFrame, name: str) -> NDArray[np.float64]:
    return blocks[name].to_numpy().astype(np.float64, copy=False)


_OUTCOME_CHUNK_SIZE = 4_096
_HistoricalItem = dict[str, torch.Tensor]
_IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _HistoricalBacking:
    first_block: int
    inputs: torch.Tensor
    base_fees: torch.Tensor

    def to(self, device: torch.device) -> _HistoricalBacking:
        return _HistoricalBacking(
            first_block=self.first_block,
            inputs=self.inputs.to(device),
            base_fees=self.base_fees.to(device),
        )


class HistoricalDataset(Dataset[int]):
    """Batched history gathered from one contiguous tensor source."""

    def __init__(
        self,
        backing: _HistoricalBacking,
        first_origin_row: int,
        labels: torch.Tensor,
        targets: torch.Tensor,
        *,
        context_blocks: int,
        horizon_blocks: int,
    ) -> None:
        self._backing = backing
        self._first_origin_row = first_origin_row
        self._labels = labels
        self._targets = targets
        self._context_blocks = context_blocks
        self._horizon_blocks = horizon_blocks

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> int:
        return index

    def to(self, device: torch.device) -> HistoricalDataset:
        return self._to(self._backing.to(device), device)

    def loader(
        self,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> Iterable[_HistoricalItem]:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._batch,
        )

    def _to(self, backing: _HistoricalBacking, device: torch.device) -> HistoricalDataset:
        return HistoricalDataset(
            backing,
            self._first_origin_row,
            self._labels.to(device),
            self._targets.to(device),
            context_blocks=self._context_blocks,
            horizon_blocks=self._horizon_blocks,
        )

    def _batch(self, indexes: list[int]) -> _HistoricalItem:
        positions = torch.tensor(indexes, device=self._labels.device)
        origins = self._first_origin_row + positions
        inputs = self._backing.inputs.unfold(0, self._context_blocks, 1).transpose(1, 2)
        base_fees = self._backing.base_fees.unfold(0, self._horizon_blocks, 1)
        return {
            "inputs": inputs[origins - self._context_blocks + 1],
            "label": self._labels[positions],
            "target": self._targets[positions],
            "base_fees": base_fees[origins + 1],
            "origin_block": self._backing.first_block + origins,
        }


@dataclass(frozen=True, slots=True)
class HistoricalPreparation:
    training: HistoricalDataset
    validation: HistoricalDataset
    feature_state: FeatureState
    target_state: TargetState

    def to(self, device: torch.device) -> HistoricalPreparation:
        backing = self.training._backing.to(device)
        return HistoricalPreparation(
            training=self.training._to(backing, device),
            validation=self.validation._to(backing, device),
            feature_state=self.feature_state,
            target_state=self.target_state,
        )


def prepare_fit_history(
    corpus: Corpus,
    experiment: ExperimentSemantics,
) -> HistoricalPreparation:
    """Fit training-only state and prepare the authored fit windows."""

    training_window = experiment.training_window
    validation_window = experiment.validation_window

    training_first_block = training_window.first_parent_block - experiment.context_blocks + 1
    predecessor_blocks = _feature_predecessor_blocks(experiment.ordered_features)
    training_support = corpus.blocks.select_range(
        training_first_block - predecessor_blocks,
        training_window.last_parent_block,
    )
    feature_state = fit_feature_state(
        training_support,
        ordered_features=experiment.ordered_features,
    )

    backing = _build_backing(
        corpus,
        first_block=training_first_block,
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
    )


def prepare_historical_window(
    corpus: Corpus,
    experiment: ExperimentSemantics,
    window: BlockWindow,
    *,
    feature_state: FeatureState,
    target_state: TargetState,
) -> HistoricalDataset:
    """Prepare one testing window without fitting state."""

    if (
        experiment.validation_window.last_parent_block + experiment.horizon_blocks
        >= window.first_parent_block
    ):
        raise ValueError("testing window must follow complete validation outcomes")
    backing = _build_backing(
        corpus,
        first_block=window.first_parent_block - experiment.context_blocks + 1,
        last_block=window.last_parent_block + experiment.horizon_blocks,
        ordered_features=experiment.ordered_features,
        feature_state=feature_state,
    )
    return _build_dataset(backing, experiment, window, target_state)


def _build_backing(
    corpus: Corpus,
    *,
    first_block: int,
    last_block: int,
    ordered_features: tuple[FeatureName, ...],
    feature_state: FeatureState,
) -> _HistoricalBacking:
    predecessor_blocks = _feature_predecessor_blocks(ordered_features)
    blocks = corpus.blocks.select_range(first_block - predecessor_blocks, last_block)
    inputs = transform_feature_rows(
        blocks,
        ordered_features=ordered_features,
        state=feature_state,
    )
    frame = blocks.to_polars().slice(predecessor_blocks)
    base_fees = frame["base_fee_per_gas"].to_numpy(writable=True)
    return _HistoricalBacking(
        first_block=first_block,
        inputs=torch.from_numpy(inputs),
        base_fees=torch.from_numpy(base_fees),
    )


def _feature_predecessor_blocks(ordered_features: tuple[FeatureName, ...]) -> int:
    return 1 if "block_interval_seconds" in ordered_features else 0


def _origin_rows(backing: _HistoricalBacking, window: BlockWindow) -> _IntVector:
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
        chunk_labels = outcomes.argmin(axis=1)
        labels[start:stop] = chunk_labels
        minima[start:stop] = outcomes[np.arange(stop - start), chunk_labels]
    return labels, minima


def _build_dataset(
    backing: _HistoricalBacking,
    experiment: ExperimentSemantics,
    window: BlockWindow,
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
        int(origin_rows[0]),
        torch.from_numpy(labels),
        torch.from_numpy(standardize_target(minima, target_state)),
        context_blocks=experiment.context_blocks,
        horizon_blocks=experiment.horizon_blocks,
    )
