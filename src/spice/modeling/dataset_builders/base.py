"""Dataset-preparation seam types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl
from pydantic import Field, field_validator, model_validator

from ...core.errors import ConfigResolutionError
from ...core.specs import lookup_local_spec, require_mapping_id, require_spec_config
from ...core.validation import validate_path_segment
from ...modeling.families.base import ConfigModel
from ...semantics import DatasetBuilderSemantics
from .preparation import (
    ArtifactInferencePreparationSpec,
    CompiledInferenceDatasetPreparationSpec,
    PreparedInferenceDataset,
    PreparedTrainingDataset,
    TrainingDatasetPreparationSpec,
)


class DatasetBuilderConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="dataset_builder.id")


class FixedSequenceTemporalDatasetBuilderConfig(DatasetBuilderConfig):
    id: str = "fixed_sequence_temporal"
    min_sequence_length: int = Field(default=64, gt=0)
    max_sequence_length: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def validate_sequence_bounds(self) -> FixedSequenceTemporalDatasetBuilderConfig:
        if self.max_sequence_length < self.min_sequence_length:
            raise ValueError("max_sequence_length must be >= min_sequence_length")
        return self


class BuilderRuntimeMetadata(ConfigModel):
    compiler_runtime_metadata: dict[str, object]


class FixedSequenceTemporalBuilderRuntimeMetadata(BuilderRuntimeMetadata):
    sequence_length: int = Field(gt=0)
    median_dt_seconds: float = Field(gt=0.0)
    min_sequence_length: int = Field(gt=0)
    max_sequence_length: int = Field(gt=0)
    split_strategy: Literal["global_feature_table"] = "global_feature_table"

    @model_validator(mode="after")
    def validate_sequence_bounds(self) -> FixedSequenceTemporalBuilderRuntimeMetadata:
        if self.max_sequence_length < self.min_sequence_length:
            raise ValueError("max_sequence_length must be >= min_sequence_length")
        return self


PrepareTrainingFn = Callable[
    [pl.DataFrame, TrainingDatasetPreparationSpec],
    PreparedTrainingDataset,
]
PrepareInferenceFn = Callable[
    [pl.DataFrame, pl.DataFrame, CompiledInferenceDatasetPreparationSpec],
    PreparedInferenceDataset,
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
        spec: TrainingDatasetPreparationSpec,
    ) -> PreparedTrainingDataset:
        return self.prepare_training_fn(blocks, spec)

    def prepare_inference_dataset(
        self,
        history_blocks: pl.DataFrame,
        evaluation_blocks: pl.DataFrame,
        *,
        spec: ArtifactInferencePreparationSpec,
    ) -> PreparedInferenceDataset:
        builder_runtime_metadata = coerce_builder_runtime_metadata(
            self.dataset_builder_id,
            spec.builder_runtime_metadata,
        )
        compiler_runtime_metadata = compiler_runtime_metadata_from_builder_payload(
            builder_runtime_metadata,
            compiler_id=spec.problem_contract.compiler_id,
        )
        return self.prepare_inference_fn(
            history_blocks,
            evaluation_blocks,
            CompiledInferenceDatasetPreparationSpec(
                feature_contract=spec.feature_contract,
                problem_contract=spec.problem_contract,
                delay_seconds=spec.delay_seconds,
                builder_runtime_metadata=builder_runtime_metadata,
                compiler_runtime_metadata=compiler_runtime_metadata,
                scaler=spec.scaler,
                max_candidate_slots=spec.max_candidate_slots,
                window_start_timestamp=spec.evaluation_start_timestamp,
                window_end_timestamp=spec.evaluation_end_timestamp + 1,
            ),
        )


def _compile_fixed_sequence_temporal(
    config: FixedSequenceTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    from .fixed_sequence_temporal import compile_dataset_builder

    return compile_dataset_builder(config)


@dataclass(frozen=True, slots=True)
class DatasetBuilderSpec:
    config_type: type[DatasetBuilderConfig]
    runtime_metadata_type: type[BuilderRuntimeMetadata]
    compile_contract: Callable[[Any], CompiledDatasetBuilderContract]


_DATASET_BUILDER_SPECS: dict[str, DatasetBuilderSpec] = {
    "fixed_sequence_temporal": DatasetBuilderSpec(
        config_type=FixedSequenceTemporalDatasetBuilderConfig,
        runtime_metadata_type=FixedSequenceTemporalBuilderRuntimeMetadata,
        compile_contract=_compile_fixed_sequence_temporal,
    ),
}


def dataset_builder_spec(builder_id: str) -> DatasetBuilderSpec:
    return lookup_local_spec(_DATASET_BUILDER_SPECS, builder_id, "dataset_builder.id")


def fixed_sequence_temporal_runtime_metadata(
    *,
    compiler_id: str,
    compiler_runtime_metadata: object,
    sequence_length: int,
    median_dt_seconds: float,
    min_sequence_length: int,
    max_sequence_length: int,
) -> FixedSequenceTemporalBuilderRuntimeMetadata:
    from ...temporal.contracts import problem_runtime_metadata_payload

    return FixedSequenceTemporalBuilderRuntimeMetadata(
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


def validate_feature_prerequisites(
    actual,
    expected,
) -> None:
    if actual != expected:
        raise ValueError(
            "Resolved feature prerequisites do not match the current feature graph: "
            f"expected {expected.model_dump(mode='json')}, "
            f"got {actual.model_dump(mode='json')}"
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
    return dataset_builder_spec(builder_id).runtime_metadata_type.model_validate(raw_payload)


def coerce_dataset_builder_config(
    payload: Mapping[str, object] | DatasetBuilderConfig,
) -> DatasetBuilderConfig:
    if isinstance(payload, DatasetBuilderConfig):
        raw_payload = payload.model_dump(mode="json")
        builder_id = payload.id
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
        builder_id = require_mapping_id(raw_payload, "dataset_builder.id")
    else:
        raise ConfigResolutionError("dataset_builder must be a mapping or config model")
    return dataset_builder_spec(builder_id).config_type.model_validate(raw_payload)


def compile_dataset_builder_contract(
    config: DatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    spec = dataset_builder_spec(config.id)
    concrete_config = require_spec_config(config, spec.config_type, "dataset builder config")
    return spec.compile_contract(concrete_config)
