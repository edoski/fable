"""Training artifact runtime helpers and feature-graph validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel

from ..config.models import (
    FeatureSetConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    TrainingConfig,
)
from ..core.constants import MODEL_STATE_FILENAME
from ..core.errors import ConfigResolutionError
from ..core.files import write_path_atomic
from ..features import CompiledFeatureContract, compile_feature_contract
from ..objectives import ObjectiveConfig
from ..prediction import compile_prediction_contract
from ..semantics import ArtifactSemantics
from ..storage.artifact import load_artifact_manifest, write_artifact_manifest
from ..storage.engine import RootKind
from .dataset_builders import (
    CompiledDatasetBuilderContract,
    DatasetBuilderConfig,
    compile_dataset_builder_contract,
)
from .families.base import ModelConfig
from .families.registry import build_model
from .models import TemporalModel
from .pipeline import PreparedTrainingDataset, TrainingSpec
from .representations import (
    CompiledRepresentationContract,
    sequence_input_contract,
    validate_representation_id,
)
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
    objective: ObjectiveConfig,
    model: ModelConfig,
    split: SplitConfig,
    training: TrainingConfig,
) -> ValidatedArtifactPreparation:
    for label, stored_value, configured_value in (
        ("problem", manifest.problem, problem),
        ("dataset_builder", manifest.dataset_builder, dataset_builder),
        ("prediction", manifest.prediction, prediction),
        ("objective", manifest.objective, objective),
        ("model", manifest.model, model),
        ("feature_set", manifest.feature_set, feature_set),
        ("split", manifest.split, split),
        ("training", manifest.training, training),
    ):
        _require_matching_config(
            label=label,
            stored_value=stored_value,
            configured_value=configured_value,
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


def _require_matching_config(
    *,
    label: str,
    stored_value: BaseModel,
    configured_value: BaseModel,
) -> None:
    if stored_value.model_dump(mode="json") != configured_value.model_dump(mode="json"):
        raise ConfigResolutionError(
            f"Configured {label} does not match the trained artifact semantics"
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
        objective=spec.objective,
        chain_name=spec.chain.name,
        dataset_id=spec.dataset_id,
        dataset_name=spec.dataset_name,
        problem=spec.problem,
        variant=spec.variant,
        study=spec.study,
        study_id=spec.study_id,
        feature_set=spec.feature_set,
        model=spec.model,
        split=spec.split,
        training=spec.training,
        scaler=prepared.scaler,
        builder_runtime_metadata=prepared.builder_runtime_metadata,
        semantics=ArtifactSemantics(
            problem=spec.contract.semantics,
            realization_policy=spec.contract.realization_policy.semantics,
            objective=spec.objective_contract.semantics,
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
        family_id=manifest.prediction.family_id,
    )
    model = build_model(
        manifest.n_features,
        prediction_contract.build_output_spec(manifest.max_candidate_slots),
        manifest.model,
    )
    state_dict = torch.load(artifact_dir / MODEL_STATE_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    validate_representation_id(manifest.representation_id)
    return LoadedTrainingArtifact(
        manifest=manifest,
        model=model,
        representation_contract=sequence_input_contract(),
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
