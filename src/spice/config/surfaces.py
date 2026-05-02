"""Canonical surface frames and selection overlays."""

from __future__ import annotations

from typing import TypeVar, cast

from ..core.config_model import ConfigModel
from .models import ArtifactConfig, ProblemSpec, StorageSpec, StudyConfig
from .registry import load_surface_frame as _load_surface_frame
from .selections import (
    AcquireWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
)

ConfigT = TypeVar("ConfigT", bound=ConfigModel)


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


def load_surface_frame(name: str) -> SurfaceFrame:
    return _load_surface_frame(name)


SurfaceWorkflowSelection = AcquireWorkflowSelection | TrainWorkflowSelection | TuneWorkflowSelection


def apply_surface_selection_overrides(
    frame: SurfaceFrame,
    selection: SurfaceWorkflowSelection,
) -> SurfaceFrame:
    updates: dict[str, object] = {}
    if selection.chain is not None:
        updates["chain"] = selection.chain
    if selection.problem is not None:
        updates["problem"] = selection.problem
    if selection.features is not None:
        updates["features"] = selection.features
    objective = getattr(selection, "objective", None)
    if objective is not None:
        updates["objective"] = objective
    evaluation = getattr(selection, "evaluation", None)
    if evaluation is not None:
        updates["evaluation"] = _updated_model(frame.evaluation, id=evaluation)
    model = getattr(selection, "model", None)
    if model is not None:
        updates["model"] = model
    tuning_space = getattr(selection, "tuning_space", None)
    if tuning_space is not None:
        updates["tuning"] = _updated_model(
            cast(SurfaceTuningFrame, updates.get("tuning", frame.tuning)),
            space=tuning_space,
        )
    provider = getattr(selection, "provider", None)
    if provider is not None:
        updates["acquisition"] = _updated_model(
            cast(SurfaceAcquisitionFrame, updates.get("acquisition", frame.acquisition)),
            provider=provider,
        )
    training = getattr(selection, "training", None)
    if training is not None:
        updates["training"] = _updated_model(frame.training, id=training)
    split = getattr(selection, "split", None)
    if split is not None:
        updates["training"] = _updated_model(
            cast(SurfaceTrainingFrame, updates.get("training", frame.training)),
            split=split,
        )
    tuning = getattr(selection, "tuning", None)
    if tuning is not None:
        updates["tuning"] = _updated_model(
            cast(SurfaceTuningFrame, updates.get("tuning", frame.tuning)),
            id=tuning,
        )
    if selection.storage_root is not None:
        base_storage = frame.storage or StorageSpec()
        updates["storage"] = _updated_model(base_storage, root=selection.storage_root)
    study = getattr(selection, "study", None)
    if study is not None:
        base_study = frame.study or StudyConfig()
        updates["study"] = _updated_model(base_study, name=study)
    variant = getattr(selection, "variant", None)
    if variant is not None:
        base_artifact = frame.artifact or ArtifactConfig()
        updates["artifact"] = _updated_model(base_artifact, variant=variant)
    return frame.model_copy(update=updates)


def _updated_model(model: ConfigT, **updates: object) -> ConfigT:
    return type(model).model_validate(
        {
            **model.model_dump(mode="json", exclude_none=True),
            **updates,
        }
    )
