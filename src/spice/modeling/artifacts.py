"""Training artifact runtime helpers and feature-graph validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import (
    DatasetBuilderConfig,
    FeatureSetConfig,
    ModelConfig,
    PredictionConfig,
    ProblemSpec,
)
from ..core.constants import MODEL_STATE_FILENAME
from ..core.errors import ConfigResolutionError
from ..core.files import write_path_atomic
from ..features import CompiledFeatureContract, compile_feature_contract
from ..prediction import compile_prediction_contract
from ..semantics import ArtifactSemantics
from ..storage.artifact import load_artifact_manifest, write_artifact_manifest
from ..storage.engine import RootKind
from ._runtime import CompiledRepresentationContract
from .dataset_builders import (
    CompiledDatasetBuilderContract,
    compile_dataset_builder_contract,
)
from .families.registry import build_model
from .models import TemporalModel
from .pipeline import PreparedTrainingDataset, TrainingSpec
from .representations import compile_representation_contract
from .results import TrainingArtifactManifest


@dataclass(slots=True)
class LoadedTrainingArtifact:
    manifest: TrainingArtifactManifest
    model: TemporalModel
    representation_contract: CompiledRepresentationContract
    dataset_builder_contract: CompiledDatasetBuilderContract


@dataclass(slots=True)
class ValidatedArtifactPreparation:
    feature_contract: CompiledFeatureContract
    dataset_builder_contract: CompiledDatasetBuilderContract


def validate_artifact_semantics(
    manifest: TrainingArtifactManifest,
    *,
    problem: ProblemSpec,
    dataset_builder: DatasetBuilderConfig,
    feature_set: FeatureSetConfig,
    prediction: PredictionConfig,
    model: ModelConfig,
) -> ValidatedArtifactPreparation:
    if manifest.problem.model_dump(mode="json") != problem.model_dump(mode="json"):
        raise ConfigResolutionError(
            "Configured problem does not match the trained artifact semantics"
        )
    if manifest.dataset_builder.model_dump(mode="json") != dataset_builder.model_dump(mode="json"):
        raise ConfigResolutionError(
            "Configured dataset_builder does not match the trained artifact semantics"
        )
    if manifest.prediction.model_dump(mode="json") != prediction.model_dump(mode="json"):
        raise ConfigResolutionError(
            "Configured prediction does not match the trained artifact semantics"
        )
    if manifest.model.model_dump(mode="json") != model.model_dump(mode="json"):
        raise ConfigResolutionError(
            "Configured model does not match the trained artifact semantics"
        )
    if manifest.feature_set.model_dump(mode="json") != feature_set.model_dump(mode="json"):
        raise ConfigResolutionError(
            "Configured feature_set does not match the trained artifact semantics"
        )
    feature_contract = compile_feature_contract(feature_set=feature_set)
    if feature_contract.feature_graph_fingerprint != manifest.feature_graph_fingerprint:
        raise ConfigResolutionError(
            "Current feature graph does not match the trained artifact manifest"
        )
    if feature_contract.feature_prerequisites != manifest.feature_prerequisites:
        raise ConfigResolutionError(
            "Current feature prerequisites do not match the trained artifact manifest"
        )
    return ValidatedArtifactPreparation(
        feature_contract=feature_contract,
        dataset_builder_contract=compile_dataset_builder_contract(manifest.dataset_builder),
    )


def build_training_artifact_manifest(
    prepared: PreparedTrainingDataset,
    *,
    spec: TrainingSpec,
) -> TrainingArtifactManifest:
    return TrainingArtifactManifest(
        artifact_id=spec.artifact_id,
        dataset_builder=spec.dataset_builder,
        prediction=spec.prediction,
        chain_name=spec.chain.name,
        dataset_id=spec.dataset_id,
        dataset_name=spec.dataset_name,
        problem=spec.problem,
        variant=spec.variant,
        study=spec.study,
        study_id=spec.study_id,
        feature_set=spec.feature_set,
        model=spec.model,
        scaler=prepared.scaler,
        builder_runtime_metadata=prepared.builder_runtime_metadata,
        semantics=ArtifactSemantics(
            problem=spec.contract.semantics,
            feature=spec.feature_contract.semantics,
            prediction=spec.prediction_contract.semantics,
            input_normalization=spec.input_normalization_contract.semantics,
            representation=spec.representation_contract.semantics,
            dataset_builder=spec.dataset_builder_contract.semantics,
            max_candidate_slots=prepared.max_candidate_slots,
        ),
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
        representation_contract=compile_representation_contract(manifest.representation_id),
        dataset_builder_contract=compile_dataset_builder_contract(manifest.dataset_builder),
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
