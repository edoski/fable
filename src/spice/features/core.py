"""Core feature source/spec execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float64]
LogFeeVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


class FeatureConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FeaturePrerequisites(FeatureConfigModel):
    history_seconds: int = Field(default=0, ge=0)
    warmup_rows: int = Field(default=0, ge=0)


@dataclass(slots=True, frozen=True)
class CanonicalBlockSeries:
    block_numbers: IntVector
    timestamps: IntVector
    log_base_fees: LogFeeVector


@dataclass(slots=True)
class ResolvedFeatureTable:
    features_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    feature_prerequisites: FeaturePrerequisites
    series: CanonicalBlockSeries
    feature_matrix: FloatMatrix


class ComputeSourceFn(Protocol):
    def __call__(self, blocks: pl.DataFrame) -> FloatVector: ...


class ComputeFeatureFn(Protocol):
    def __call__(
        self,
        blocks: pl.DataFrame,
        series: CanonicalBlockSeries,
        sources: Mapping[str, FloatVector],
        features: Mapping[str, FloatVector],
    ) -> FloatVector: ...


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Available model-time source with explicit lag and null policy."""

    source_columns: tuple[str, ...]
    warmup_rows: int
    required_after_warmup: bool
    compute: ComputeSourceFn


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Formula over source/spec dependencies."""

    source_dependencies: tuple[str, ...]
    feature_dependencies: tuple[str, ...]
    history_seconds: int
    warmup_rows: int
    compute: ComputeFeatureFn


@dataclass(frozen=True, slots=True)
class FeatureCatalog:
    sources: dict[str, SourceSpec]
    features: dict[str, FeatureSpec]
    fingerprint_sources: tuple[Path, ...]


def validate_feature_names(
    features_id: str,
    feature_names: tuple[str, ...],
    *,
    known_feature_names: tuple[str, ...],
) -> None:
    if not features_id:
        raise ValueError("features.id must be non-empty")
    if not feature_names:
        raise ValueError("features.outputs must not be empty")
    duplicates = [name for name in dict.fromkeys(feature_names) if feature_names.count(name) > 1]
    if duplicates:
        raise ValueError("features.outputs must not contain duplicates: " + ", ".join(duplicates))
    unknown = [name for name in feature_names if name not in known_feature_names]
    if unknown:
        raise ValueError("Unknown feature outputs: " + ", ".join(unknown))


def feature_prerequisites(
    catalog: FeatureCatalog,
    feature_names: tuple[str, ...],
) -> FeaturePrerequisites:
    ordered = _dependency_order(feature_names, catalog=catalog)
    source_names = {
        source_name
        for feature_name in ordered
        for source_name in catalog.features[feature_name].source_dependencies
    }
    return FeaturePrerequisites(
        history_seconds=max(
            (catalog.features[name].history_seconds for name in ordered),
            default=0,
        ),
        warmup_rows=max(
            (
                *(catalog.features[name].warmup_rows for name in ordered),
                *(catalog.sources[name].warmup_rows for name in source_names),
            ),
            default=0,
        ),
    )


def feature_source_columns(
    catalog: FeatureCatalog,
    feature_names: tuple[str, ...],
) -> frozenset[str]:
    ordered = _dependency_order(feature_names, catalog=catalog)
    return frozenset(
        column
        for feature_name in ordered
        for source_name in catalog.features[feature_name].source_dependencies
        for column in catalog.sources[source_name].source_columns
    )


def feature_graph_fingerprint(
    features_id: str,
    feature_names: tuple[str, ...],
    *,
    fingerprint_sources: tuple[Path, ...],
) -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = sha256()
    digest.update(features_id.encode("utf-8"))
    digest.update(b"\0")
    for name in feature_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    for source in fingerprint_sources:
        resolved = source.resolve()
        digest.update(_fingerprint_source_id(resolved, package_root=package_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fingerprint_source_id(source: Path, *, package_root: Path) -> str:
    try:
        return source.relative_to(package_root).as_posix()
    except ValueError:
        return source.name


def build_feature_table(
    blocks: pl.DataFrame,
    *,
    features_id: str,
    catalog: FeatureCatalog,
    feature_names: tuple[str, ...],
    allowed_feature_names: tuple[str, ...],
) -> ResolvedFeatureTable:
    validate_feature_names(
        features_id,
        feature_names,
        known_feature_names=allowed_feature_names,
    )
    sorted_blocks = blocks.sort("block_number")
    ordered_features = _dependency_order(feature_names, catalog=catalog)
    required_source_names = tuple(
        dict.fromkeys(
            source_name
            for feature_name in ordered_features
            for source_name in catalog.features[feature_name].source_dependencies
        )
    )
    required_columns = {
        column
        for source_name in required_source_names
        for column in catalog.sources[source_name].source_columns
    }
    missing_columns = sorted(required_columns.difference(sorted_blocks.columns))
    if missing_columns:
        raise ValueError("Features require missing block columns: " + ", ".join(missing_columns))

    prerequisites = feature_prerequisites(catalog, feature_names)
    sources = _compute_sources(
        sorted_blocks,
        catalog=catalog,
        source_names=required_source_names,
        warmup_rows=prerequisites.warmup_rows,
    )
    series = _build_series(sorted_blocks, sources=sources, warmup_rows=prerequisites.warmup_rows)
    resolved: dict[str, FloatVector] = {}
    for feature_name in ordered_features:
        definition = catalog.features[feature_name]
        values = np.asarray(
            definition.compute(sorted_blocks, series, sources, resolved),
            dtype=np.float64,
        )
        _validate_vector_shape(feature_name, values, series.timestamps.shape[0])
        resolved[feature_name] = values.astype(np.float64, copy=False)

    matrix_columns = [
        _finite_feature_values(
            feature_name,
            resolved[feature_name],
            warmup_rows=prerequisites.warmup_rows,
        )
        for feature_name in feature_names
    ]
    feature_matrix = np.column_stack(matrix_columns).astype(np.float32, copy=False)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature matrix must contain finite values only")
    return ResolvedFeatureTable(
        features_id=features_id,
        feature_names=feature_names,
        feature_graph_fingerprint=feature_graph_fingerprint(
            features_id,
            feature_names,
            fingerprint_sources=catalog.fingerprint_sources,
        ),
        feature_prerequisites=prerequisites,
        series=series,
        feature_matrix=feature_matrix,
    )


def _compute_sources(
    blocks: pl.DataFrame,
    *,
    catalog: FeatureCatalog,
    source_names: tuple[str, ...],
    warmup_rows: int,
) -> dict[str, FloatVector]:
    sources: dict[str, FloatVector] = {}
    n_rows = blocks.height
    for source_name in source_names:
        source = catalog.sources[source_name]
        values = np.asarray(source.compute(blocks), dtype=np.float64)
        _validate_vector_shape(source_name, values, n_rows)
        if (
            source.required_after_warmup
            and not np.isfinite(values[source.warmup_rows :]).all()
        ):
            raise ValueError(
                f"Feature source {source_name} requires finite values after warmup"
            )
        cleaned = values.astype(np.float64, copy=True)
        if warmup_rows > 0:
            cleaned[:warmup_rows] = np.nan_to_num(
                cleaned[:warmup_rows],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        sources[source_name] = cleaned
    return sources


def _build_series(
    blocks: pl.DataFrame,
    *,
    sources: Mapping[str, FloatVector],
    warmup_rows: int,
) -> CanonicalBlockSeries:
    base_fee = sources.get("current_base_fee_per_gas")
    if base_fee is None:
        base_fee = _float_column(blocks, "base_fee_per_gas")
    if not np.isfinite(base_fee[warmup_rows:]).all():
        raise ValueError("base_fee_per_gas must be finite after warmup")
    cleaned_base_fee = base_fee.astype(np.float64, copy=True)
    if warmup_rows > 0:
        cleaned_base_fee[:warmup_rows] = np.nan_to_num(
            cleaned_base_fee[:warmup_rows],
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
    return CanonicalBlockSeries(
        block_numbers=blocks["block_number"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False),
        timestamps=blocks["timestamp"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False),
        log_base_fees=np.log(np.clip(cleaned_base_fee, 1.0, None)).astype(
            np.float32,
            copy=False,
        ),
    )


def _finite_feature_values(
    feature_name: str,
    values: FloatVector,
    *,
    warmup_rows: int,
) -> FloatVector:
    cleaned = values.astype(np.float64, copy=True)
    if warmup_rows > 0:
        cleaned[:warmup_rows] = np.nan_to_num(
            cleaned[:warmup_rows],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    if not np.isfinite(cleaned[warmup_rows:]).all():
        raise ValueError(f"Feature {feature_name} produced non-finite values after warmup")
    return cleaned


def _validate_vector_shape(name: str, values: FloatVector, n_rows: int) -> None:
    expected_shape = (n_rows,)
    if values.shape != expected_shape:
        raise ValueError(f"{name} returned shape {values.shape}, expected {expected_shape}")


def _dependency_order(
    feature_names: tuple[str, ...],
    *,
    catalog: FeatureCatalog,
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
            definition = catalog.features[feature_name]
        except KeyError as exc:
            raise ValueError(f"Unknown feature dependency: {feature_name}") from exc
        visiting.add(feature_name)
        for dependency_name in definition.feature_dependencies:
            _visit(dependency_name)
        visiting.remove(feature_name)
        visited.add(feature_name)
        ordered.append(feature_name)

    for feature_name in feature_names:
        _visit(feature_name)
    return tuple(ordered)


def _float_column(blocks: pl.DataFrame, column: str) -> FloatVector:
    return blocks[column].cast(pl.Float64).to_numpy().astype(np.float64, copy=False)


def shifted_column(blocks: pl.DataFrame, column: str, *, lag: int = 1) -> FloatVector:
    return shift(_float_column(blocks, column), lag=lag)


def shift(values: FloatVector, *, lag: int = 1) -> FloatVector:
    if lag <= 0:
        raise ValueError("lag must be positive")
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size > lag:
        result[lag:] = values[:-lag]
    return result


def rolling_stat(values: FloatVector, *, window: int, stat: str, ddof: int = 0) -> FloatVector:
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    series = pl.Series(values)
    if stat == "mean":
        result = series.rolling_mean(window_size=window, min_samples=window)
    elif stat == "std":
        result = series.rolling_std(window_size=window, min_samples=window, ddof=ddof)
    elif stat == "min":
        result = series.rolling_min(window_size=window, min_samples=window)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported rolling stat: {stat}")
    return result.to_numpy().astype(np.float64, copy=False)


def hour_sin(timestamps: IntVector) -> FloatVector:
    hours = (timestamps // 3600) % 24
    return np.sin(2.0 * math.pi * hours.astype(np.float64, copy=False) / 24.0)


def hour_cos(timestamps: IntVector) -> FloatVector:
    hours = (timestamps // 3600) % 24
    return np.cos(2.0 * math.pi * hours.astype(np.float64, copy=False) / 24.0)


def dow_sin(timestamps: IntVector) -> FloatVector:
    days = ((timestamps // 86_400) + 4) % 7
    return np.sin(2.0 * math.pi * days.astype(np.float64, copy=False) / 7.0)


def dow_cos(timestamps: IntVector) -> FloatVector:
    days = ((timestamps // 86_400) + 4) % 7
    return np.cos(2.0 * math.pi * days.astype(np.float64, copy=False) / 7.0)
