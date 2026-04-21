"""Shared feature-family types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...core.validation import validate_path_segment

if TYPE_CHECKING:
    from ..core import CanonicalBlockSeries

FloatVector = NDArray[np.float64]

class FeatureConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FeatureFamilyConfig(FeatureConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="feature_set.family.id")


class FeaturePrerequisites(FeatureConfigModel):
    history_seconds: int = Field(default=0, ge=0)
    warmup_rows: int = Field(default=0, ge=0)


class ComputeFeatureFn(Protocol):
    def __call__(
        self,
        blocks: pl.DataFrame,
        series: CanonicalBlockSeries,
        resolved_dependencies: Mapping[str, FloatVector],
    ) -> FloatVector: ...


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    dependencies: tuple[str, ...]
    history_seconds: int
    warmup_rows: int
    compute: ComputeFeatureFn


class BuildSeriesFn(Protocol):
    def __call__(self, blocks: pl.DataFrame) -> CanonicalBlockSeries: ...


@dataclass(frozen=True, slots=True)
class FeatureFamily:
    features: dict[str, FeatureDefinition]
    fingerprint_sources: tuple[Path, ...]
    build_series: BuildSeriesFn
