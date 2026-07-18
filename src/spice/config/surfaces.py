"""Canonical surface frames."""

from __future__ import annotations

from ..core.config_model import ConfigModel
from .models import ArtifactConfig, ProblemSpec, StorageSpec, StudyConfig


class SurfaceTrainingFrame(ConfigModel):
    id: str
    split: str


class SurfaceTuningFrame(ConfigModel):
    id: str
    space: str | None = None


class SurfaceFrame(ConfigModel):
    chain: str
    corpus: str
    problem: str | ProblemSpec
    features: str | None = None
    prediction: str
    model: str | None = None
    training: SurfaceTrainingFrame
    tuning: SurfaceTuningFrame
    evaluations: str | None = None
    storage: StorageSpec | None = None
    study: StudyConfig | None = None
    artifact: ArtifactConfig | None = None
