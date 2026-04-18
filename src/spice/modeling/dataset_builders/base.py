"""Dataset-preparation seam types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import polars as pl
from pydantic import Field, field_validator

from ...modeling.families.base import ConfigModel
from ...semantics import DatasetBuilderSemantics

if TYPE_CHECKING:
    from ..pipeline import (
        InferencePreparationSpec,
        PreparedInferenceDataset,
        PreparedTrainingDataset,
        TrainingSpec,
    )


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


DatasetBuilderIdT = TypeVar("DatasetBuilderIdT", bound=str)


class DatasetBuilderConfig(ConfigModel, Generic[DatasetBuilderIdT]):
    id: DatasetBuilderIdT

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="dataset_builder.id")


class StandardTemporalDatasetBuilderConfig(
    DatasetBuilderConfig[Literal["standard_temporal"]]
):
    id: Literal["standard_temporal"] = Field(default="standard_temporal")


PrepareTrainingFn = Callable[[pl.DataFrame, "TrainingSpec"], "PreparedTrainingDataset"]
PrepareInferenceFn = Callable[
    [pl.DataFrame, pl.DataFrame, "InferencePreparationSpec"],
    "PreparedInferenceDataset",
]


@dataclass(frozen=True, slots=True)
class CompiledDatasetBuilderContract:
    dataset_builder_id: str
    config_payload: dict[str, object]
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


DatasetBuilderConfigT = TypeVar("DatasetBuilderConfigT", bound=DatasetBuilderConfig)


@dataclass(frozen=True, slots=True)
class DatasetBuilderSpec(Generic[DatasetBuilderConfigT]):
    id: str
    config_type: type[DatasetBuilderConfigT]
    compile: Callable[[DatasetBuilderConfigT], CompiledDatasetBuilderContract]
