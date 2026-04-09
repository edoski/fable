"""Training artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from spice_temporal.config import ChainConfig, ModelConfig
from spice_temporal.contracts import TemporalModel
from spice_temporal.features import FEATURE_NAMES
from spice_temporal.models import build_model
from spice_temporal.normalization import StandardScaler
from spice_temporal.pipeline import PreparedTrainingDataset

ARTIFACT_MANIFEST_FILENAME = "artifact.json"
MODEL_STATE_FILENAME = "model.pt"
TRAIN_REPORT_FILENAME = "train_report.json"
SIMULATION_REPORT_FILENAME = "simulation_report.json"


@dataclass(slots=True)
class TrainingArtifactManifest:
    chain: ChainConfig
    max_delay_seconds: int
    lookback_seconds: int
    target_anchor_count: int
    lookback_steps: int
    max_extra_wait_steps: int
    action_count: int
    n_features: int
    feature_names: list[str]
    model_config: ModelConfig
    scaler: StandardScaler


@dataclass(slots=True)
class LoadedTrainingArtifact:
    manifest: TrainingArtifactManifest
    model: TemporalModel


def build_training_artifact_manifest(
    prepared: PreparedTrainingDataset,
    *,
    chain: ChainConfig,
    max_delay_seconds: int,
    lookback_seconds: int,
    target_anchor_count: int,
    model_config: ModelConfig,
) -> TrainingArtifactManifest:
    return TrainingArtifactManifest(
        chain=chain,
        max_delay_seconds=max_delay_seconds,
        lookback_seconds=lookback_seconds,
        target_anchor_count=target_anchor_count,
        lookback_steps=prepared.geometry.lookback_steps,
        max_extra_wait_steps=prepared.geometry.max_extra_wait_steps,
        action_count=prepared.geometry.action_count,
        n_features=prepared.n_features,
        feature_names=list(FEATURE_NAMES),
        model_config=model_config,
        scaler=prepared.scaler,
    )


def write_training_artifact(
    artifact_dir: Path,
    *,
    manifest: TrainingArtifactManifest,
    model: TemporalModel,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / ARTIFACT_MANIFEST_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, ensure_ascii=True, indent=2)
    cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(cpu_state, artifact_dir / MODEL_STATE_FILENAME)


def load_training_artifact(artifact_dir: Path) -> LoadedTrainingArtifact:
    manifest_path = artifact_dir / ARTIFACT_MANIFEST_FILENAME
    model_path = artifact_dir / MODEL_STATE_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    manifest = TrainingArtifactManifest(
        chain=ChainConfig(**raw["chain"]),
        max_delay_seconds=int(raw["max_delay_seconds"]),
        lookback_seconds=int(raw["lookback_seconds"]),
        target_anchor_count=int(raw["target_anchor_count"]),
        lookback_steps=int(raw["lookback_steps"]),
        max_extra_wait_steps=int(raw["max_extra_wait_steps"]),
        action_count=int(raw["action_count"]),
        n_features=int(raw["n_features"]),
        feature_names=list(raw["feature_names"]),
        model_config=ModelConfig(**raw["model_config"]),
        scaler=StandardScaler(**raw["scaler"]),
    )
    model = build_model(manifest.n_features, manifest.action_count, manifest.model_config)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedTrainingArtifact(manifest=manifest, model=model)
