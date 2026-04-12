"""Training artifact models and feature-graph validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import ArtifactVariant, ModelConfig, StudyConfig
from ..core.constants import MODEL_STATE_FILENAME
from ..core.files import write_path_atomic
from ..data.normalization import ScalerStats
from ..features import (
    FeatureSelection,
    feature_graph_fingerprint,
    feature_warmup_blocks,
    validate_feature_selection,
)
from ..planning.geometry import (
    action_count_for_delay,
    lookback_steps_for_seconds,
    max_extra_wait_steps_for_delay,
    minimum_history_context_blocks,
)
from ..state.artifact import load_artifact_manifest, write_artifact_manifest
from .models import TemporalModel
from .pipeline import PreparedTrainingDataset, TrainingSpec
from .registry import build_model


@dataclass(frozen=True, slots=True)
class ArtifactChainMetadata:
    name: str
    block_time_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingArtifactManifest:
    chain: ArtifactChainMetadata
    dataset_id: str
    task_id: str
    variant: ArtifactVariant
    study: StudyConfig | None
    max_supported_delay_seconds: int
    lookback_seconds: int
    sample_count: int
    feature_set_id: str
    feature_names: list[str]
    feature_graph_fingerprint: str
    model: ModelConfig
    scaler: ScalerStats

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def lookback_steps(self) -> int:
        return lookback_steps_for_seconds(
            self.lookback_seconds,
            self.chain.block_time_seconds,
        )

    @property
    def max_extra_wait_steps(self) -> int:
        return max_extra_wait_steps_for_delay(
            self.max_supported_delay_seconds,
            self.chain.block_time_seconds,
        )

    @property
    def action_count(self) -> int:
        return action_count_for_delay(
            self.max_supported_delay_seconds,
            self.chain.block_time_seconds,
        )

    @property
    def required_history_context_blocks(self) -> int:
        return minimum_history_context_blocks(
            lookback_seconds=self.lookback_seconds,
            block_time_seconds=self.chain.block_time_seconds,
            feature_warmup_blocks=feature_warmup_blocks(tuple(self.feature_names)),
        )

    @property
    def history_context_blocks(self) -> int:
        return self.required_history_context_blocks


@dataclass(slots=True)
class LoadedTrainingArtifact:
    manifest: TrainingArtifactManifest
    model: TemporalModel


def feature_selection_from_manifest(manifest: TrainingArtifactManifest) -> FeatureSelection:
    selection = FeatureSelection(
        feature_set_id=manifest.feature_set_id,
        feature_names=tuple(manifest.feature_names),
    )
    validate_feature_selection(selection.feature_set_id, selection.feature_names)
    return selection


def validate_artifact_feature_graph(
    manifest: TrainingArtifactManifest,
    *,
    requested_feature_set_id: str | None = None,
) -> FeatureSelection:
    selection = feature_selection_from_manifest(manifest)
    if (
        requested_feature_set_id is not None
        and requested_feature_set_id != selection.feature_set_id
    ):
        raise ValueError(
            "Configured feature_set.id does not match the trained artifact: "
            f"expected {selection.feature_set_id}, got {requested_feature_set_id}"
        )
    current_fingerprint = feature_graph_fingerprint(selection.feature_names)
    if current_fingerprint != manifest.feature_graph_fingerprint:
        raise ValueError(
            "Current feature graph does not match the trained artifact manifest"
        )
    return selection


def build_training_artifact_manifest(
    prepared: PreparedTrainingDataset,
    *,
    spec: TrainingSpec,
) -> TrainingArtifactManifest:
    return TrainingArtifactManifest(
        chain=ArtifactChainMetadata(
            name=spec.chain.name,
            block_time_seconds=spec.chain.runtime.block_time_seconds,
        ),
        dataset_id=spec.dataset_id,
        task_id=spec.task.id,
        variant=spec.variant,
        study=spec.study,
        max_supported_delay_seconds=spec.task.max_supported_delay_seconds,
        lookback_seconds=spec.task.lookback_seconds,
        sample_count=spec.task.sample_count,
        feature_set_id=prepared.feature_set_id,
        feature_names=list(prepared.feature_names),
        feature_graph_fingerprint=prepared.feature_graph_fingerprint,
        model=spec.model,
        scaler=prepared.scaler,
    )


def load_training_artifact(artifact_dir: Path) -> LoadedTrainingArtifact:
    manifest = load_artifact_manifest(artifact_dir / ".spice" / "state.sqlite")
    model = build_model(manifest.n_features, manifest.action_count, manifest.model)
    state_dict = torch.load(artifact_dir / MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedTrainingArtifact(manifest=manifest, model=model)


def persist_training_artifact(
    artifact_dir: Path,
    *,
    manifest: TrainingArtifactManifest,
    root_kind: str,
    model: TemporalModel,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_manifest(
        artifact_dir / ".spice" / "state.sqlite",
        manifest=manifest,
        root_kind=root_kind,
    )
    cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _write(tmp_path: Path) -> None:
        torch.save(cpu_state, tmp_path)

    write_path_atomic(artifact_dir / MODEL_STATE_FILENAME, _write)
