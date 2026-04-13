"""Hamilton-backed feature graph helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import numpy as np
import polars as pl
from hamilton import driver
from hamilton.graph_types import HamiltonNode
from numpy.typing import NDArray

from . import base, rolling, trend

FEATURE_KIND_TAG = "spice_kind"
FEATURE_HISTORY_SECONDS_TAG = "spice_history_seconds"
FEATURE_KIND_VALUE = "feature"

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(slots=True, frozen=True)
class FeatureSelection:
    feature_set_id: str
    feature_names: tuple[str, ...]


def make_feature_selection(
    feature_set_id: str,
    feature_names: tuple[str, ...],
) -> FeatureSelection:
    selection = FeatureSelection(
        feature_set_id=feature_set_id,
        feature_names=feature_names,
    )
    validate_feature_selection(selection.feature_set_id, selection.feature_names)
    return selection


@dataclass(slots=True)
class FeatureTable:
    feature_set_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    feature_history_seconds: int
    block_numbers: IntVector
    timestamps: IntVector
    feature_matrix: FloatMatrix
    log_base_fees: FloatVector


@lru_cache(maxsize=1)
def build_feature_driver() -> driver.Driver:
    return driver.Builder().with_modules(base, rolling, trend).build()


@lru_cache(maxsize=1)
def _node_map() -> dict[str, HamiltonNode]:
    return {node.name: node for node in build_feature_driver().list_available_variables()}


@lru_cache(maxsize=1)
def feature_node_map() -> dict[str, HamiltonNode]:
    return {
        name: node
        for name, node in _node_map().items()
        if node.tags.get(FEATURE_KIND_TAG) == FEATURE_KIND_VALUE
    }


def validate_feature_selection(feature_set_id: str, feature_names: tuple[str, ...]) -> None:
    if not feature_set_id:
        raise ValueError("feature_set.id must be non-empty")
    if not feature_names:
        raise ValueError("feature_set.outputs must not be empty")
    duplicates = [name for name in dict.fromkeys(feature_names) if feature_names.count(name) > 1]
    if duplicates:
        raise ValueError(
            "feature_set.outputs must not contain duplicates: " + ", ".join(duplicates)
        )
    unknown = [name for name in feature_names if name not in feature_node_map()]
    if unknown:
        raise ValueError("Unknown feature outputs: " + ", ".join(unknown))


def feature_history_seconds(feature_names: tuple[str, ...]) -> int:
    validate_feature_selection("validated", feature_names)
    return max(
        int(feature_node_map()[name].tags[FEATURE_HISTORY_SECONDS_TAG])
        for name in feature_names
    )


def _dependency_closure(feature_names: tuple[str, ...]) -> list[HamiltonNode]:
    nodes = _node_map()
    stack = list(feature_names)
    visited: set[str] = set()
    ordered: list[HamiltonNode] = []
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        node = nodes[name]
        ordered.append(node)
        stack.extend(sorted(node.required_dependencies))
        stack.extend(sorted(node.optional_dependencies))
    return sorted(ordered, key=lambda item: item.name)


def feature_graph_fingerprint(feature_names: tuple[str, ...]) -> str:
    validate_feature_selection("validated", feature_names)
    digest = sha256()
    for name in feature_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    for node in _dependency_closure(feature_names):
        digest.update(node.name.encode("utf-8"))
        digest.update(b"\0")
        if node.is_external_input:
            continue
        for function in node.originating_functions:
            digest.update(function.__module__.encode("utf-8"))
            digest.update(b"\0")
            digest.update(function.__qualname__.encode("utf-8"))
            digest.update(b"\0")
            digest.update(inspect.getsource(function).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def build_feature_table(
    blocks: pl.DataFrame,
    *,
    selection: FeatureSelection,
) -> FeatureTable:
    validate_feature_selection(selection.feature_set_id, selection.feature_names)
    required_history_seconds = feature_history_seconds(selection.feature_names)
    graph_fingerprint = feature_graph_fingerprint(selection.feature_names)
    result = build_feature_driver().execute(
        list(selection.feature_names) + ["block_numbers", "timestamps", "log_base_fee"],
        inputs={
            "blocks": blocks,
        },
    )
    block_numbers = np.asarray(result["block_numbers"], dtype=np.int64)
    timestamps = np.asarray(result["timestamps"], dtype=np.int64)
    log_base_fees = np.asarray(result["log_base_fee"], dtype=np.float32)
    raw_feature_columns = [
        np.asarray(result[feature_name], dtype=np.float64)
        for feature_name in selection.feature_names
    ]
    if raw_feature_columns:
        feature_matrix = np.column_stack(raw_feature_columns).astype(np.float32, copy=False)
    else:  # pragma: no cover - guarded by config validation
        feature_matrix = np.empty((block_numbers.shape[0], 0), dtype=np.float32)
    return FeatureTable(
        feature_set_id=selection.feature_set_id,
        feature_names=selection.feature_names,
        feature_graph_fingerprint=graph_fingerprint,
        feature_history_seconds=required_history_seconds,
        block_numbers=block_numbers.astype(np.int64, copy=False),
        timestamps=timestamps.astype(np.int64, copy=False),
        feature_matrix=feature_matrix.astype(np.float32, copy=False),
        log_base_fees=log_base_fees.astype(np.float32, copy=False),
    )
