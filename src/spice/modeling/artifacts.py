"""Training artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from ..core.config import ChainConfig, ModelConfig
from ..core.constants import (
    ARTIFACT_MANIFEST_FILENAME,
    MODEL_STATE_FILENAME,
)
from ..data.features import FEATURE_NAMES
from ..data.normalization import ScalerStats
from .models import TemporalModel, build_model
from .pipeline import PreparedTrainingDataset, TrainingSpec


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingArtifactManifest(ArtifactModel):
    kind: Literal["training_artifact"] = "training_artifact"
    chain: ChainConfig
    dataset_id: str
    max_delay_seconds: int
    lookback_seconds: int
    target_anchor_count: int
    lookback_steps: int
    max_extra_wait_steps: int
    action_count: int
    n_features: int
    feature_names: list[str]
    model: ModelConfig
    scaler: ScalerStats


@dataclass(slots=True)
class LoadedTrainingArtifact:
    manifest: TrainingArtifactManifest
    model: TemporalModel


def build_training_artifact_manifest(
    prepared: PreparedTrainingDataset,
    *,
    spec: TrainingSpec,
) -> TrainingArtifactManifest:
    return TrainingArtifactManifest(
        chain=spec.chain,
        dataset_id=spec.dataset_id,
        max_delay_seconds=spec.max_delay_seconds,
        lookback_seconds=spec.lookback_seconds,
        target_anchor_count=spec.target_anchor_count,
        lookback_steps=prepared.geometry.lookback_steps,
        max_extra_wait_steps=prepared.geometry.max_extra_wait_steps,
        action_count=prepared.geometry.action_count,
        n_features=prepared.n_features,
        feature_names=list(FEATURE_NAMES),
        model=spec.model,
        scaler=prepared.scaler,
    )


def write_training_artifact(
    artifact_dir: Path,
    *,
    manifest: TrainingArtifactManifest,
    model: TemporalModel,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ARTIFACT_MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(cpu_state, artifact_dir / MODEL_STATE_FILENAME)


def load_training_artifact(artifact_dir: Path) -> LoadedTrainingArtifact:
    manifest = TrainingArtifactManifest.model_validate_json(
        (artifact_dir / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    model = build_model(manifest.n_features, manifest.action_count, manifest.model)
    state_dict = torch.load(artifact_dir / MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedTrainingArtifact(manifest=manifest, model=model)
