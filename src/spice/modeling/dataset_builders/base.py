"""Dataset-preparation seam types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
from pydantic import field_validator

from ...core.closed_dispatch import (
    config_payload_and_id,
    unknown_id_error,
    validate_path_segment,
)
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


def _require_standard_temporal(builder_id: str) -> None:
    if builder_id != "standard_temporal":
        raise unknown_id_error(
            field_name="dataset_builder.id",
            component_id=builder_id,
            known_ids=("standard_temporal",),
        )


def coerce_dataset_builder_config(
    payload: Mapping[str, object] | DatasetBuilderConfig,
) -> DatasetBuilderConfig:
    raw_payload, builder_id = config_payload_and_id(
        payload,
        config_type=DatasetBuilderConfig,
        field_name="dataset_builder.id",
        mapping_label="dataset_builder",
    )
    _require_standard_temporal(builder_id)
    return StandardTemporalDatasetBuilderConfig.model_validate(raw_payload)


def compile_dataset_builder_contract(
    config: DatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    _require_standard_temporal(config.id)
    return _compile_standard_temporal(StandardTemporalDatasetBuilderConfig.model_validate(config))
