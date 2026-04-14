"""Training artifact runtime helpers and feature-graph validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import FeatureSetConfig, ModelConfig, PredictionConfig, ProblemSpec
from ..core.constants import MODEL_STATE_FILENAME
from ..core.files import write_path_atomic
from ..features import CompiledFeatureContract, compile_feature_contract
from ..prediction import compile_prediction_contract
from ..storage.artifact import load_artifact_manifest, write_artifact_manifest
from ..storage.engine import RootKind
from ._runtime import CompiledRepresentationContract
from .families.registry import build_model, compile_default_representation_contract
from .models import TemporalModel
from .pipeline import PreparedTrainingDataset, TrainingSpec
from .results import ArtifactChainMetadata, TrainingArtifactManifest


@dataclass(slots=True)
class LoadedTrainingArtifact:
    manifest: TrainingArtifactManifest
    model: TemporalModel
    representation_contract: CompiledRepresentationContract


def validate_artifact_semantics(
    manifest: TrainingArtifactManifest,
    *,
    problem: ProblemSpec,
    feature_set: FeatureSetConfig,
    prediction: PredictionConfig,
    model: ModelConfig,
) -> CompiledFeatureContract:
    if manifest.problem.model_dump(mode="json") != problem.model_dump(mode="json"):
        raise ValueError("Configured problem does not match the trained artifact semantics")
    if manifest.prediction.model_dump(mode="json") != prediction.model_dump(mode="json"):
        raise ValueError("Configured prediction does not match the trained artifact semantics")
    if manifest.model.model_dump(mode="json") != model.model_dump(mode="json"):
        raise ValueError("Configured model does not match the trained artifact semantics")
    if manifest.feature_set.model_dump(mode="json") != feature_set.model_dump(mode="json"):
        raise ValueError("Configured feature_set does not match the trained artifact semantics")
    feature_contract = compile_feature_contract(feature_set=feature_set)
    if feature_contract.feature_graph_fingerprint != manifest.feature_graph_fingerprint:
        raise ValueError("Current feature graph does not match the trained artifact manifest")
    if feature_contract.feature_prerequisites != manifest.feature_prerequisites:
        raise ValueError("Current feature prerequisites do not match the trained artifact manifest")
    return feature_contract


def build_training_artifact_manifest(
    prepared: PreparedTrainingDataset,
    *,
    spec: TrainingSpec,
) -> TrainingArtifactManifest:
    return TrainingArtifactManifest(
        artifact_id=spec.artifact_id,
        prediction=spec.prediction,
        training_metric_descriptors=list(spec.prediction_contract.training_metric_descriptors),
        chain=ArtifactChainMetadata(name=spec.chain.name),
        dataset_id=spec.dataset_id,
        dataset_name=spec.dataset_name,
        problem=spec.problem,
        variant=spec.variant,
        study=spec.study,
        study_id=spec.study_id,
        feature_set=spec.feature_set,
        feature_prerequisites=prepared.feature_prerequisites,
        max_candidate_slots=prepared.max_candidate_slots,
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
    return LoadedTrainingArtifact(
        manifest=manifest,
        model=model,
        representation_contract=compile_default_representation_contract(manifest.model.id),
    )


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
