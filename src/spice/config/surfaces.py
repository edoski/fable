"""Canonical surface frames."""

from __future__ import annotations

from ..core.config_model import ConfigModel
from .models import ArtifactConfig, ProblemSpec, StorageSpec, StudyConfig


class SurfaceAcquisitionFrame(ConfigModel):
    provider: str


class SurfaceTrainingFrame(ConfigModel):
    id: str
    split: str


class SurfaceTuningFrame(ConfigModel):
    id: str
    space: str | None = None


class SurfaceEvaluationFrame(ConfigModel):
    id: str | None = None


class SurfaceFrame(ConfigModel):
    chain: str
    dataset: str
    problem: str | ProblemSpec
    dataset_builder: str
    features: str | None = None
    prediction: str
    objective: str | None = None
    model: str | None = None
    acquisition: SurfaceAcquisitionFrame
    training: SurfaceTrainingFrame
    tuning: SurfaceTuningFrame
    evaluation: SurfaceEvaluationFrame
    storage: StorageSpec | None = None
    study: StudyConfig | None = None
    artifact: ArtifactConfig | None = None

    @property
    def provider(self) -> str:
        return self.acquisition.provider

    @property
    def training_id(self) -> str:
        return self.training.id

    @property
    def split(self) -> str:
        return self.training.split

    @property
    def tuning_id(self) -> str:
        return self.tuning.id

    @property
    def tuning_space_id(self) -> str | None:
        return self.tuning.space

    @property
    def evaluation_id(self) -> str | None:
        return self.evaluation.id
