"""Dataset-preparation seam types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
from pydantic import Field, field_validator, model_validator

from ...core.validation import validate_path_segment
from ...core.errors import ConfigResolutionError
from ...modeling.families.base import ConfigModel
from ...semantics import DatasetBuilderSemantics

if TYPE_CHECKING:
    from ...temporal.contracts import ProblemRuntimeMetadata
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


class ProfessorTemporalDatasetBuilderConfig(DatasetBuilderConfig):
    id: str = "professor_temporal"
    min_sequence_length: int = Field(default=64, gt=0)
    max_sequence_length: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def validate_sequence_bounds(self) -> ProfessorTemporalDatasetBuilderConfig:
        if self.max_sequence_length < self.min_sequence_length:
            raise ValueError("max_sequence_length must be >= min_sequence_length")
        return self


BuilderRuntimeMetadata = dict[str, object]

_COMPILER_RUNTIME_METADATA_KEY = "compiler_runtime_metadata"


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


def _compile_professor_temporal(
    config: ProfessorTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    from .professor_temporal import compile_dataset_builder

    return compile_dataset_builder(config)


def builder_runtime_metadata(
    *,
    compiler_runtime_metadata: ProblemRuntimeMetadata,
    extra: Mapping[str, object] | None = None,
) -> BuilderRuntimeMetadata:
    from ...temporal.contracts import problem_runtime_metadata_payload

    payload: BuilderRuntimeMetadata = {
        _COMPILER_RUNTIME_METADATA_KEY: problem_runtime_metadata_payload(compiler_runtime_metadata)
    }
    if extra is not None:
        payload.update(dict(extra))
    return payload


def compiler_runtime_metadata_from_builder_payload(
    payload: Mapping[str, object],
    *,
    compiler_id: str,
) -> ProblemRuntimeMetadata:
    from ...temporal.contracts import problem_runtime_metadata_from_compiler_payload

    raw_payload = payload.get(_COMPILER_RUNTIME_METADATA_KEY)
    if not isinstance(raw_payload, Mapping):
        raise ConfigResolutionError("builder runtime metadata is missing compiler_runtime_metadata")
    return problem_runtime_metadata_from_compiler_payload(compiler_id, raw_payload)


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
    if builder_id == "professor_temporal":
        return ProfessorTemporalDatasetBuilderConfig.model_validate(raw_payload)
    raise ValueError(
        "dataset_builder.id must be one of: standard_temporal, professor_temporal"
    )


def compile_dataset_builder_contract(
    config: DatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    if config.id == "standard_temporal":
        return _compile_standard_temporal(
            StandardTemporalDatasetBuilderConfig.model_validate(config)
        )
    if config.id == "professor_temporal":
        return _compile_professor_temporal(
            ProfessorTemporalDatasetBuilderConfig.model_validate(config)
        )
    raise ValueError(
        "dataset_builder.id must be one of: standard_temporal, professor_temporal"
    )
