"""Training artifact runtime helpers and feature-graph validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..core.constants import MODEL_STATE_FILENAME
from ..core.files import write_path_atomic
from ..features import FeatureSelection, feature_graph_fingerprint, make_feature_selection
from ..prediction import compile_prediction_contract
from ..storage.artifact import load_artifact_manifest, write_artifact_manifest
from ..storage.engine import RootKind
from .families.registry import build_model
from .models import TemporalModel
from .pipeline import PreparedTrainingDataset, TrainingSpec
from .results import ArtifactChainMetadata, TrainingArtifactManifest


@dataclass(slots=True)
class LoadedTrainingArtifact:
    manifest: TrainingArtifactManifest
    model: TemporalModel


def feature_selection_from_manifest(manifest: TrainingArtifactManifest) -> FeatureSelection:
    return make_feature_selection(
        feature_set_id=manifest.feature_set_id,
        feature_family_id=manifest.feature_family_id,
        feature_names=tuple(manifest.feature_names),
    )


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
    current_fingerprint = feature_graph_fingerprint(
        selection.feature_family_id,
        selection.feature_names,
    )
    if current_fingerprint != manifest.feature_graph_fingerprint:
        raise ValueError("Current feature graph does not match the trained artifact manifest")
    return selection


def build_training_artifact_manifest(
    prepared: PreparedTrainingDataset,
    *,
    spec: TrainingSpec,
) -> TrainingArtifactManifest:
    return TrainingArtifactManifest(
        artifact_id=spec.artifact_id,
        objective_id=spec.prediction_contract.objective_id,
        prediction=spec.prediction,
        metric_descriptors=list(spec.prediction_contract.metric_descriptors),
        chain=ArtifactChainMetadata(name=spec.chain.name),
        dataset_id=spec.dataset_id,
        dataset_name=spec.dataset_name,
        problem=spec.problem,
        variant=spec.variant,
        study=spec.study,
        study_id=spec.study_id,
        feature_family_id=prepared.feature_family_id,
        feature_prerequisites=prepared.feature_prerequisites,
        max_candidate_slots=prepared.max_candidate_slots,
        feature_set_id=prepared.feature_set_id,
        feature_names=list(prepared.feature_names),
        feature_graph_fingerprint=prepared.feature_graph_fingerprint,
        model=spec.model,
        scaler=prepared.scaler,
        compiler_runtime_metadata=prepared.compiler_runtime_metadata,
    )


def load_training_artifact(artifact_dir: Path) -> LoadedTrainingArtifact:
    manifest = load_artifact_manifest(artifact_dir / ".spice" / "state.sqlite")
    prediction_contract = compile_prediction_contract(
        prediction_id=manifest.prediction.id,
        family_config=manifest.prediction.family,
    )
    model = build_model(
        manifest.n_features,
        prediction_contract.build_output_spec(manifest.max_candidate_slots),
        manifest.model,
    )
    state_dict = torch.load(artifact_dir / MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedTrainingArtifact(manifest=manifest, model=model)


def persist_training_artifact(
    artifact_dir: Path,
    *,
    manifest: TrainingArtifactManifest,
    root_kind: RootKind,
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
