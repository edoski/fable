"""Disposable Issue 28 prototype for the fixed-context historical data seam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

Float64Matrix = NDArray[np.float64]
Int64Vector = NDArray[np.int64]
HistoricalItem = dict[str, torch.Tensor]

TARGET_ID = "hindsight_minimum_base_fee_per_gas_within_k"


@dataclass(frozen=True, slots=True)
class FeatureFitProvenance:
    corpus_id: str
    chain_id: int
    regime: str
    first_block: int
    last_block: int
    count: int


@dataclass(frozen=True, slots=True)
class TargetFitProvenance:
    corpus_id: str
    chain_id: int
    regime: str
    first_origin_block: int
    last_origin_block: int
    count: int
    k: int


@dataclass(frozen=True, slots=True)
class FeatureState:
    names: tuple[str, ...]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    provenance: FeatureFitProvenance

    def __post_init__(self) -> None:
        means = _readonly_float64_vector(self.means, "feature means")
        scales = _readonly_float64_vector(self.scales, "feature scales")
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be nonempty and unique")
        if means.shape != (len(self.names),) or scales.shape != means.shape:
            raise ValueError("feature state width must match ordered feature names")
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("feature state must be finite")
        if np.any(scales <= 0.0):
            raise ValueError("feature scales must be strictly positive")
        if self.provenance.count <= 0:
            raise ValueError("feature provenance count must be positive")
        if not self.provenance.corpus_id or not self.provenance.regime:
            raise ValueError("feature provenance identity must be nonempty")
        if (
            self.provenance.chain_id <= 0
            or self.provenance.first_block > self.provenance.last_block
        ):
            raise ValueError("feature provenance range must be valid")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)

    def transform(
        self,
        rows: Float64Matrix,
        *,
        names: tuple[str, ...],
    ) -> NDArray[np.float32]:
        _require_float64_feature_rows(rows, len(self.names))
        if names != self.names:
            raise ValueError("feature order does not match fitted feature state")
        transformed = (rows - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise ValueError("feature transform produced a nonfinite float64 value")
        model_rows = transformed.astype(np.float32)
        if not np.isfinite(model_rows).all():
            raise ValueError("feature transform produced a nonfinite float32 value")
        return np.ascontiguousarray(model_rows)


@dataclass(frozen=True, slots=True)
class TargetState:
    mean: np.float64
    scale: np.float64
    provenance: TargetFitProvenance
    target_id: str = TARGET_ID

    def __post_init__(self) -> None:
        if self.target_id != TARGET_ID:
            raise ValueError(f"target_id must equal {TARGET_ID!r}")
        if not isinstance(self.mean, np.float64) or not isinstance(self.scale, np.float64):
            raise ValueError("target mean and scale must be float64 scalars")
        if not np.isfinite(self.mean) or not np.isfinite(self.scale):
            raise ValueError("target state must be finite")
        if self.scale <= 0.0:
            raise ValueError("target scale must be strictly positive")
        if self.provenance.count <= 0 or self.provenance.k <= 0:
            raise ValueError("target provenance count and K must be positive")
        if not self.provenance.corpus_id or not self.provenance.regime:
            raise ValueError("target provenance identity must be nonempty")
        if (
            self.provenance.chain_id <= 0
            or self.provenance.first_origin_block > self.provenance.last_origin_block
        ):
            raise ValueError("target provenance range must be valid")

    def transform(self, raw_minima: Int64Vector) -> NDArray[np.float32]:
        _require_positive_int64_vector(raw_minima, "raw target minima")
        logged = np.log(raw_minima.astype(np.float64))
        standardized = (logged - self.mean) / self.scale
        if not np.isfinite(standardized).all():
            raise ValueError("target transform produced a nonfinite value")
        with np.errstate(over="ignore"):
            model_targets = standardized.astype(np.float32)
        if not np.isfinite(model_targets).all():
            raise ValueError("target transform produced a nonfinite float32 value")
        return np.ascontiguousarray(model_targets)


@dataclass(frozen=True, slots=True)
class HistoricalArrays:
    """Private shared runtime storage; it is not a combined persisted state."""

    _inputs: torch.Tensor
    _base_fees: torch.Tensor
    _block_numbers: torch.Tensor
    _feature_state: FeatureState
    _target_state: TargetState
    _names: tuple[str, ...]

    def __post_init__(self) -> None:
        row_count = self._inputs.shape[0]
        if self._inputs.device.type != "cpu" or self._inputs.dtype != torch.float32:
            raise ValueError("shared inputs must be CPU float32")
        if self._inputs.ndim != 2 or not self._inputs.is_contiguous():
            raise ValueError("shared inputs must be contiguous [R,F]")
        if self._base_fees.device.type != "cpu" or self._base_fees.dtype != torch.int64:
            raise ValueError("shared base fees must be CPU int64")
        if not self._base_fees.is_contiguous():
            raise ValueError("shared base fees must be contiguous [R]")
        if self._block_numbers.device.type != "cpu" or self._block_numbers.dtype != torch.int64:
            raise ValueError("shared block numbers must be CPU int64")
        if not self._block_numbers.is_contiguous():
            raise ValueError("shared block numbers must be contiguous [R]")
        if self._base_fees.shape != (row_count,) or self._block_numbers.shape != (row_count,):
            raise ValueError("shared row arrays must have equal length")
        if self._inputs.shape[1] != len(self._names) or self._names != self._feature_state.names:
            raise ValueError("shared input width/order must match feature state")
        if not torch.isfinite(self._inputs).all():
            raise ValueError("shared inputs must be finite")
        if torch.any(self._base_fees <= 0):
            raise ValueError("shared base fees must be positive")
        if row_count > 1 and torch.any(torch.diff(self._block_numbers) != 1):
            raise ValueError("shared block numbers must be consecutive and ordered")
        _require_matching_state_identity(self._feature_state, self._target_state)

    @property
    def row_count(self) -> int:
        return self._inputs.shape[0]

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self._inputs.shape)

    @property
    def base_fee_shape(self) -> tuple[int, ...]:
        return tuple(self._base_fees.shape)

    @property
    def shared_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._inputs, self._base_fees, self._block_numbers)
        )


def fit_feature_state(
    raw_rows: Float64Matrix,
    support_positions: Int64Vector,
    block_numbers: Int64Vector,
    *,
    names: tuple[str, ...],
    provenance: FeatureFitProvenance,
) -> FeatureState:
    """Fit once from unique training-visible physical rows in float64."""

    _require_float64_feature_rows(raw_rows, len(names))
    _require_int64_positions(support_positions, raw_rows.shape[0], "feature support")
    _require_consecutive_blocks(block_numbers, raw_rows.shape[0])
    if support_positions.size == 0 or np.any(np.diff(support_positions) != 1):
        raise ValueError("feature support must be one direct unique physical-row interval")
    actual = (int(block_numbers[support_positions[0]]), int(block_numbers[support_positions[-1]]))
    expected = (provenance.first_block, provenance.last_block)
    if actual != expected or support_positions.size != provenance.count:
        raise ValueError("feature fit population does not match declared provenance")
    population = raw_rows[support_positions]
    means = population.mean(axis=0, dtype=np.float64)
    scales = population.std(axis=0, ddof=0, dtype=np.float64)
    return FeatureState(names=names, means=means, scales=scales, provenance=provenance)


def fit_target_state(
    base_fees: Int64Vector,
    training_origins: Int64Vector,
    block_numbers: Int64Vector,
    *,
    k: int,
    provenance: TargetFitProvenance,
    chunk_size: int = 4_096,
) -> TargetState:
    """Fit separately from one exact raw minimum per training origin."""

    _require_consecutive_blocks(block_numbers, base_fees.shape[0])
    labels, minima = earliest_minimum(
        base_fees,
        training_origins,
        k=k,
        chunk_size=chunk_size,
    )
    del labels
    if provenance.k != k:
        raise ValueError("target provenance K does not match requested K")
    actual = (
        int(block_numbers[training_origins[0]]),
        int(block_numbers[training_origins[-1]]),
        int(training_origins.size),
    )
    expected = (
        provenance.first_origin_block,
        provenance.last_origin_block,
        provenance.count,
    )
    if actual != expected:
        raise ValueError("target fit population does not match declared provenance")
    logged = np.log(minima.astype(np.float64))
    mean = logged.mean(dtype=np.float64)
    scale = logged.std(ddof=0, dtype=np.float64)
    return TargetState(mean=mean, scale=scale, provenance=provenance)


def earliest_minimum(
    base_fees: Int64Vector,
    origins: Int64Vector,
    *,
    k: int,
    chunk_size: int = 4_096,
) -> tuple[Int64Vector, Int64Vector]:
    """Return first exact argmin labels and minima using bounded [chunk,K] copies."""

    _require_positive_int64_vector(base_fees, "base fees")
    _require_int64_positions(origins, base_fees.shape[0], "origins")
    if origins.size == 0 or np.any(np.diff(origins) <= 0):
        raise ValueError("origins must be nonempty, unique, and strictly increasing")
    if k <= 0 or chunk_size <= 0:
        raise ValueError("K and chunk_size must be positive")
    if int(origins[-1]) + k >= base_fees.shape[0]:
        raise ValueError("every origin must have K complete future outcomes")

    labels = np.empty(origins.shape, dtype=np.int64)
    minima = np.empty(origins.shape, dtype=np.int64)
    outcome_offsets = np.arange(1, k + 1, dtype=np.int64)
    for start in range(0, origins.size, chunk_size):
        stop = min(start + chunk_size, origins.size)
        window_positions = origins[start:stop, None] + outcome_offsets[None, :]
        windows = base_fees[window_positions]
        chunk_labels = windows.argmin(axis=1).astype(np.int64, copy=False)
        labels[start:stop] = chunk_labels
        minima[start:stop] = windows[np.arange(stop - start), chunk_labels]
    return labels, minima


def prepare_arrays(
    raw_rows: Float64Matrix,
    base_fees: Int64Vector,
    block_numbers: Int64Vector,
    *,
    names: tuple[str, ...],
    feature_state: FeatureState,
    target_state: TargetState,
    corpus_id: str,
    chain_id: int,
    regime: str,
    feature_support_positions: Int64Vector,
    training_origins: Int64Vector,
    k: int,
) -> HistoricalArrays:
    """Build the one shared CPU row/fee/block storage used by all roles."""

    _require_float64_feature_rows(raw_rows, len(names))
    _require_positive_int64_vector(base_fees, "base fees")
    _require_consecutive_blocks(block_numbers, raw_rows.shape[0])
    if base_fees.shape[0] != raw_rows.shape[0]:
        raise ValueError("raw feature and base-fee row counts must match")
    if not corpus_id or chain_id <= 0 or not regime:
        raise ValueError("canonical history identity must be valid")
    _require_int64_positions(
        feature_support_positions,
        raw_rows.shape[0],
        "canonical feature support",
    )
    if feature_support_positions.size == 0 or np.any(np.diff(feature_support_positions) != 1):
        raise ValueError("canonical feature support must be one direct unique-row interval")
    _require_int64_positions(training_origins, raw_rows.shape[0], "canonical training origins")
    if training_origins.size == 0 or np.any(np.diff(training_origins) <= 0):
        raise ValueError("canonical training origins must be unique and ordered")
    if k <= 0 or int(training_origins[-1]) + k >= raw_rows.shape[0]:
        raise ValueError("canonical training origins must have K complete outcomes")
    _require_matching_state_identity(feature_state, target_state)
    expected_feature_provenance = FeatureFitProvenance(
        corpus_id=corpus_id,
        chain_id=chain_id,
        regime=regime,
        first_block=int(block_numbers[feature_support_positions[0]]),
        last_block=int(block_numbers[feature_support_positions[-1]]),
        count=int(feature_support_positions.size),
    )
    expected_target_provenance = TargetFitProvenance(
        corpus_id=corpus_id,
        chain_id=chain_id,
        regime=regime,
        first_origin_block=int(block_numbers[training_origins[0]]),
        last_origin_block=int(block_numbers[training_origins[-1]]),
        count=int(training_origins.size),
        k=k,
    )
    if feature_state.provenance != expected_feature_provenance:
        raise ValueError("feature state provenance does not match the prepared history")
    if target_state.provenance != expected_target_provenance:
        raise ValueError("target state provenance does not match the prepared history")
    inputs = torch.from_numpy(feature_state.transform(raw_rows, names=names).copy())
    fees = torch.from_numpy(np.array(base_fees, dtype=np.int64, copy=True, order="C"))
    blocks = torch.from_numpy(np.array(block_numbers, dtype=np.int64, copy=True, order="C"))
    return HistoricalArrays(
        _inputs=inputs,
        _base_fees=fees,
        _block_numbers=blocks,
        _feature_state=feature_state,
        _target_state=target_state,
        _names=names,
    )


class HistoricalDataset(Dataset[HistoricalItem]):
    """Lazy fixed-context map-style dataset over one shared canonical row store."""

    def __init__(
        self,
        arrays: HistoricalArrays,
        origin_positions: Int64Vector,
        *,
        c: int,
        k: int,
        chunk_size: int = 4_096,
    ) -> None:
        _require_int64_positions(origin_positions, arrays.row_count, "origins")
        if c <= 0 or k <= 0:
            raise ValueError("C and K must be positive")
        if k != arrays._target_state.provenance.k:
            raise ValueError("dataset K must match target-state K")
        if origin_positions.size == 0 or np.any(np.diff(origin_positions) <= 0):
            raise ValueError("role origins must be nonempty, unique, and ordered")
        if int(origin_positions[0]) - c + 1 < 0:
            raise ValueError("every origin must have exactly C visible context rows")
        labels, minima = earliest_minimum(
            arrays._base_fees.numpy(),
            origin_positions,
            k=k,
            chunk_size=chunk_size,
        )
        targets = arrays._target_state.transform(minima)
        self._arrays = arrays
        self._origins = torch.from_numpy(np.array(origin_positions, copy=True))
        self._labels = torch.from_numpy(labels.copy())
        self._targets = torch.from_numpy(targets.copy())
        self._c = c
        self._k = k

    def __len__(self) -> int:
        return self._origins.numel()

    def __getitem__(self, index: int) -> HistoricalItem:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("HistoricalDataset indices must be integers")
        if index < 0 or index >= len(self):
            raise IndexError("HistoricalDataset index out of range")
        origin = int(self._origins[index])
        # Clones are transient. They keep the one shared backing store immutable to callers.
        return {
            "inputs": self._arrays._inputs[origin - self._c + 1 : origin + 1].clone(),
            "label": self._labels[index].clone(),
            "target": self._targets[index].clone(),
            "base_fees": self._arrays._base_fees[origin + 1 : origin + 1 + self._k].clone(),
            "origin_block": self._arrays._block_numbers[origin].clone(),
        }


def prepare_live_input(
    raw_context_rows: Float64Matrix,
    *,
    names: tuple[str, ...],
    state: FeatureState,
    c: int,
) -> torch.Tensor:
    """Separate live path: shared feature transform, no outcome or target fields."""

    if raw_context_rows.shape != (c, len(names)):
        raise ValueError("live input must contain exactly [C,F] ordered rows")
    inputs = torch.from_numpy(state.transform(raw_context_rows, names=names).copy())
    return inputs.unsqueeze(0)


def _readonly_float64_vector(value: NDArray[np.float64], label: str) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional float64 NumPy array")
    array = np.array(value, copy=True, order="C")
    array.setflags(write=False)
    return array


def _require_matching_state_identity(
    feature_state: FeatureState,
    target_state: TargetState,
) -> None:
    feature_identity = (
        feature_state.provenance.corpus_id,
        feature_state.provenance.chain_id,
        feature_state.provenance.regime,
    )
    target_identity = (
        target_state.provenance.corpus_id,
        target_state.provenance.chain_id,
        target_state.provenance.regime,
    )
    if feature_identity != target_identity:
        raise ValueError("feature and target state identities must match")


def _require_float64_feature_rows(rows: Float64Matrix, width: int) -> None:
    if not isinstance(rows, np.ndarray) or rows.dtype != np.float64 or rows.ndim != 2:
        raise ValueError("feature rows must be a float64 [R,F] NumPy array")
    if rows.shape[0] == 0 or rows.shape[1] != width:
        raise ValueError("feature rows must be nonempty with the declared width")
    if not np.isfinite(rows).all():
        raise ValueError("feature rows must be finite")


def _require_positive_int64_vector(values: Int64Vector, label: str) -> None:
    if not isinstance(values, np.ndarray) or values.dtype != np.int64 or values.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional int64 NumPy array")
    if values.size == 0 or np.any(values <= 0):
        raise ValueError(f"{label} must contain only positive integers")


def _require_int64_positions(positions: Int64Vector, row_count: int, label: str) -> None:
    if not isinstance(positions, np.ndarray) or positions.dtype != np.int64:
        raise ValueError(f"{label} must be an int64 NumPy array")
    if positions.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if positions.size and (np.any(positions < 0) or np.any(positions >= row_count)):
        raise ValueError(f"{label} contains an out-of-range row position")


def _require_consecutive_blocks(block_numbers: Int64Vector, row_count: int) -> None:
    if not isinstance(block_numbers, np.ndarray) or block_numbers.dtype != np.int64:
        raise ValueError("block numbers must be a one-dimensional int64 NumPy array")
    if block_numbers.shape != (row_count,):
        raise ValueError("block numbers must align one-for-one with rows")
    if row_count == 0 or np.any(np.diff(block_numbers) != 1):
        raise ValueError("block numbers must be strictly consecutive and ordered")
