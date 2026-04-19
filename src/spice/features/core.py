"""Core feature execution and fingerprint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .families.base import FeatureFamilySpec, FeaturePrerequisites

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float64]
LogFeeVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(slots=True, frozen=True)
class CanonicalBlockSeries:
    block_numbers: IntVector
    timestamps: IntVector
    log_base_fees: LogFeeVector


@dataclass(slots=True)
class ResolvedFeatureTable:
    feature_set_id: str
    feature_family_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    feature_prerequisites: FeaturePrerequisites
    series: CanonicalBlockSeries
    feature_matrix: FloatMatrix


def validate_feature_names(
    feature_set_id: str,
    feature_names: tuple[str, ...],
    *,
    known_feature_names: tuple[str, ...],
) -> None:
    if not feature_set_id:
        raise ValueError("feature_set.id must be non-empty")
    if not feature_names:
        raise ValueError("feature_set.outputs must not be empty")
    duplicates = [name for name in dict.fromkeys(feature_names) if feature_names.count(name) > 1]
    if duplicates:
        raise ValueError(
            "feature_set.outputs must not contain duplicates: " + ", ".join(duplicates)
        )
    unknown = [name for name in feature_names if name not in known_feature_names]
    if unknown:
        raise ValueError("Unknown feature outputs: " + ", ".join(unknown))


def feature_prerequisites(
    spec: FeatureFamilySpec,
    feature_names: tuple[str, ...],
) -> FeaturePrerequisites:
    return FeaturePrerequisites(
        history_seconds=max(spec.features[name].history_seconds for name in feature_names),
        warmup_rows=max(spec.features[name].warmup_rows for name in feature_names),
    )


def feature_graph_fingerprint(
    feature_family_id: str,
    feature_names: tuple[str, ...],
    *,
    fingerprint_sources: tuple[Path, ...],
) -> str:
    digest = sha256()
    digest.update(feature_family_id.encode("utf-8"))
    digest.update(b"\0")
    for name in feature_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    for source in fingerprint_sources:
        resolved = source.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_feature_table(
    blocks: pl.DataFrame,
    *,
    feature_set_id: str,
    spec: FeatureFamilySpec,
    feature_names: tuple[str, ...],
) -> ResolvedFeatureTable:
    validate_feature_names(
        feature_set_id,
        feature_names,
        known_feature_names=tuple(spec.features),
    )
    sorted_blocks = blocks.sort("block_number")
    series = spec.build_series(sorted_blocks)
    resolved: dict[str, FloatVector] = {}
    for feature_name in _dependency_order(feature_names, spec=spec):
        definition = spec.features[feature_name]
        dependency_values = {
            dependency_name: resolved[dependency_name]
            for dependency_name in definition.dependencies
        }
        values = np.asarray(
            definition.compute(sorted_blocks, series, dependency_values),
            dtype=np.float64,
        )
        expected_shape = (series.timestamps.shape[0],)
        if values.shape != expected_shape:
            raise ValueError(
                f"Feature {feature_name} returned shape {values.shape}, expected {expected_shape}"
            )
        resolved[feature_name] = values.astype(np.float64, copy=False)
    feature_matrix = np.column_stack(
        [resolved[feature_name] for feature_name in feature_names]
    ).astype(np.float32, copy=False)
    return ResolvedFeatureTable(
        feature_set_id=feature_set_id,
        feature_family_id=spec.id,
        feature_names=feature_names,
        feature_graph_fingerprint=feature_graph_fingerprint(
            spec.id,
            feature_names,
            fingerprint_sources=spec.fingerprint_sources,
        ),
        feature_prerequisites=feature_prerequisites(spec, feature_names),
        series=series,
        feature_matrix=feature_matrix,
    )


def _dependency_order(
    feature_names: tuple[str, ...],
    *,
    spec: FeatureFamilySpec,
) -> tuple[str, ...]:
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(feature_name: str) -> None:
        if feature_name in visited:
            return
        if feature_name in visiting:
            raise ValueError(f"Cyclic feature dependency detected: {feature_name}")
        try:
            definition = spec.features[feature_name]
        except KeyError as exc:
            raise ValueError(f"Unknown feature dependency: {feature_name}") from exc
        visiting.add(feature_name)
        for dependency_name in definition.dependencies:
            _visit(dependency_name)
        visiting.remove(feature_name)
        visited.add(feature_name)
        ordered.append(feature_name)

    for feature_name in feature_names:
        _visit(feature_name)
    return tuple(ordered)
