"""Canonical surface frame and request overlays."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

from pydantic import Field

from ..modeling.families.base import ConfigModel
from .models import ArtifactConfig, ProblemSpec, StorageSpec, StudyConfig
from .registry import load_named_group

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
    delay_seconds: int | None = Field(default=None, gt=0)


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

    @property
    def delay_seconds(self) -> int | None:
        return self.evaluation.delay_seconds


def load_surface_frame(name: str) -> SurfaceFrame:
    return SurfaceFrame.model_validate(load_named_group(name, "surface"))


def apply_request_overrides(
    frame: SurfaceFrame,
    *,
    chain: str | None,
    problem: str | ProblemSpec | None,
    features: str | None,
    objective: str | None,
    evaluation: str | None,
    model: str | None,
    tuning_space: str | None,
    provider: str | None,
    training: str | None,
    split: str | None,
    tuning: str | None,
    study: str | None,
    variant: str | None,
    delay_seconds: int | None,
    storage_root: Path | None,
) -> SurfaceFrame:
    updates: dict[str, object] = {}
    if chain is not None:
        updates["chain"] = chain
    if problem is not None:
        updates["problem"] = problem
    if features is not None:
        updates["features"] = features
    if objective is not None:
        updates["objective"] = objective
    if evaluation is not None:
        updates["evaluation"] = _updated_model(frame.evaluation, id=evaluation)
    if model is not None:
        updates["model"] = model
    if tuning_space is not None:
        updates["tuning"] = _updated_model(
            cast(SurfaceTuningFrame, updates.get("tuning", frame.tuning)),
            space=tuning_space,
        )
    if provider is not None:
        updates["acquisition"] = _updated_model(
            cast(SurfaceAcquisitionFrame, updates.get("acquisition", frame.acquisition)),
            provider=provider,
        )
    if training is not None:
        updates["training"] = _updated_model(frame.training, id=training)
    if split is not None:
        updates["training"] = _updated_model(
            cast(SurfaceTrainingFrame, updates.get("training", frame.training)),
            split=split,
        )
    if tuning is not None:
        updates["tuning"] = _updated_model(
            cast(SurfaceTuningFrame, updates.get("tuning", frame.tuning)),
            id=tuning,
        )
    if storage_root is not None:
        base_storage = frame.storage or StorageSpec()
        updates["storage"] = _updated_model(base_storage, root=storage_root)
    if study is not None:
        base_study = frame.study or StudyConfig()
        updates["study"] = _updated_model(base_study, name=study)
    if variant is not None:
        base_artifact = frame.artifact or ArtifactConfig()
        updates["artifact"] = _updated_model(base_artifact, variant=variant)
    if delay_seconds is not None:
        updates["evaluation"] = _updated_model(
            cast(SurfaceEvaluationFrame, updates.get("evaluation", frame.evaluation)),
            delay_seconds=delay_seconds,
        )
    return frame.model_copy(update=updates)


def _updated_model(model: ConfigT, **updates: object) -> ConfigT:
    return type(model).model_validate(
        {
            **model.model_dump(mode="json", exclude_none=True),
            **updates,
        }
    )
