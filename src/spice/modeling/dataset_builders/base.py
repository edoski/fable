"""Dataset-preparation seam types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import Field, field_validator, model_validator

from ...core.validation import validate_path_segment
from ...modeling.families.base import ConfigModel
from ...semantics import DatasetBuilderSemantics

if TYPE_CHECKING:
    from ..pipeline import (
        InferencePreparationSpec,
        PreparedInferenceDataset,
        PreparedTrainingDataset,
        TrainingSpec,
    )


class DatasetBuilderConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="dataset_builder.id")


class StandardTemporalDatasetBuilderConfig(DatasetBuilderConfig):
    id: str = "standard_temporal"


class FixedContextTemporalDatasetBuilderConfig(DatasetBuilderConfig):
    id: str = "fixed_context_temporal"
    min_sequence_length: int = Field(default=64, gt=0)
    max_sequence_length: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def validate_sequence_bounds(self) -> FixedContextTemporalDatasetBuilderConfig:
        if self.max_sequence_length < self.min_sequence_length:
            raise ValueError("max_sequence_length must be >= min_sequence_length")
        return self


class BuilderRuntimeMetadata(ConfigModel):
    compiler_runtime_metadata: dict[str, object]


class StandardTemporalBuilderRuntimeMetadata(BuilderRuntimeMetadata):
    pass


class FixedContextTemporalBuilderRuntimeMetadata(BuilderRuntimeMetadata):
    sequence_length: int = Field(gt=0)
    median_dt_seconds: float = Field(gt=0.0)
    min_sequence_length: int = Field(gt=0)
    max_sequence_length: int = Field(gt=0)
    split_strategy: Literal["global_feature_table"] = "global_feature_table"

    @model_validator(mode="after")
    def validate_sequence_bounds(self) -> FixedContextTemporalBuilderRuntimeMetadata:
        if self.max_sequence_length < self.min_sequence_length:
            raise ValueError("max_sequence_length must be >= min_sequence_length")
        return self


PrepareTrainingFn = Callable[[pl.DataFrame, "TrainingSpec"], "PreparedTrainingDataset"]
PrepareInferenceFn = Callable[
    [pl.DataFrame, pl.DataFrame, "InferencePreparationSpec"],
    "PreparedInferenceDataset",
]


@dataclass(frozen=True, slots=True)
class CompiledDatasetBuilderContract:
    dataset_builder_id: str
    prepare_training_fn: PrepareTrainingFn
    prepare_inference_fn: PrepareInferenceFn

    @property
    def semantics(self) -> DatasetBuilderSemantics:
        return DatasetBuilderSemantics(dataset_builder_id=self.dataset_builder_id)

    def prepare_training_dataset(
        self,
        blocks: pl.DataFrame,
        *,
        spec: TrainingSpec,
    ) -> PreparedTrainingDataset:
        return self.prepare_training_fn(blocks, spec)

    def prepare_inference_dataset(
        self,
        history_blocks: pl.DataFrame,
        evaluation_blocks: pl.DataFrame,
        *,
        spec: InferencePreparationSpec,
    ) -> PreparedInferenceDataset:
        return self.prepare_inference_fn(history_blocks, evaluation_blocks, spec)


def _compile_standard_temporal(
    config: StandardTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    from .standard_temporal import compile_dataset_builder

    return compile_dataset_builder(config)


def _compile_fixed_context_temporal(
    config: FixedContextTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    from .fixed_context_temporal import compile_dataset_builder

    return compile_dataset_builder(config)


def standard_temporal_runtime_metadata(
    *,
    compiler_id: str,
    compiler_runtime_metadata: object,
) -> StandardTemporalBuilderRuntimeMetadata:
    from ...temporal.contracts import problem_runtime_metadata_payload

    return StandardTemporalBuilderRuntimeMetadata(
        compiler_runtime_metadata=problem_runtime_metadata_payload(
            compiler_id,
            compiler_runtime_metadata,
        )
    )


def fixed_context_temporal_runtime_metadata(
    *,
    compiler_id: str,
    compiler_runtime_metadata: object,
    sequence_length: int,
    median_dt_seconds: float,
    min_sequence_length: int,
    max_sequence_length: int,
) -> FixedContextTemporalBuilderRuntimeMetadata:
    from ...temporal.contracts import problem_runtime_metadata_payload

    return FixedContextTemporalBuilderRuntimeMetadata(
        compiler_runtime_metadata=problem_runtime_metadata_payload(
            compiler_id,
            compiler_runtime_metadata,
        ),
        sequence_length=sequence_length,
        median_dt_seconds=median_dt_seconds,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
    )


def compiler_runtime_metadata_from_builder_payload(
    payload: BuilderRuntimeMetadata,
    *,
    compiler_id: str,
) -> object:
    from ...temporal.contracts import problem_runtime_metadata_from_compiler_payload

    return problem_runtime_metadata_from_compiler_payload(
        compiler_id,
        payload.compiler_runtime_metadata,
    )


def coerce_builder_runtime_metadata(
    builder_id: str,
    payload: Mapping[str, object] | BuilderRuntimeMetadata,
) -> BuilderRuntimeMetadata:
    raw_payload = (
        payload.model_dump(mode="json")
        if isinstance(payload, BuilderRuntimeMetadata)
        else dict(payload)
    )
    if builder_id == "standard_temporal":
        return StandardTemporalBuilderRuntimeMetadata.model_validate(raw_payload)
    if builder_id == "fixed_context_temporal":
        return FixedContextTemporalBuilderRuntimeMetadata.model_validate(raw_payload)
    raise ValueError(
        "dataset_builder.id must be one of: standard_temporal, fixed_context_temporal"
    )


def coerce_dataset_builder_config(
    payload: Mapping[str, object] | DatasetBuilderConfig,
) -> DatasetBuilderConfig:
    if isinstance(payload, DatasetBuilderConfig):
        raw_payload = payload.model_dump(mode="json")
        builder_id = payload.id
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
        builder_id = raw_payload.get("id")
    else:
        raise TypeError("dataset_builder must be a mapping or config model")
    if builder_id == "standard_temporal":
        return StandardTemporalDatasetBuilderConfig.model_validate(raw_payload)
    if builder_id == "fixed_context_temporal":
        return FixedContextTemporalDatasetBuilderConfig.model_validate(raw_payload)
    raise ValueError(
        "dataset_builder.id must be one of: standard_temporal, fixed_context_temporal"
    )


def compile_dataset_builder_contract(
    config: DatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    if config.id == "standard_temporal":
        return _compile_standard_temporal(
            StandardTemporalDatasetBuilderConfig.model_validate(config)
        )
    if config.id == "fixed_context_temporal":
        return _compile_fixed_context_temporal(
            FixedContextTemporalDatasetBuilderConfig.model_validate(config)
        )
    raise ValueError(
        "dataset_builder.id must be one of: standard_temporal, fixed_context_temporal"
    )
