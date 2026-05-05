# pyright: strict

"""Apply unresolved surface workflow selections to named surface frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, cast

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from .models import ArtifactConfig, StorageSpec, StudyConfig
from .selections import SurfaceWorkflowSelection
from .surfaces import (
    SurfaceAcquisitionFrame,
    SurfaceFrame,
    SurfaceTrainingFrame,
    SurfaceTuningFrame,
)
from .typed_registry import load_surface_frame

ConfigT = TypeVar("ConfigT", bound=ConfigModel)


@dataclass(frozen=True, slots=True)
class AppliedSurfaceSelection:
    surface_name: str
    selection: SurfaceWorkflowSelection
    frame: SurfaceFrame


def apply_surface_selection(
    selection: SurfaceWorkflowSelection,
) -> AppliedSurfaceSelection:
    if selection.surface is None:
        raise ConfigResolutionError("surface is required")
    return AppliedSurfaceSelection(
        surface_name=selection.surface,
        selection=selection,
        frame=_apply_selection_to_frame(load_surface_frame(selection.surface), selection),
    )


def _apply_selection_to_frame(
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
