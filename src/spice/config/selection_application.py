# pyright: strict

"""Apply unresolved surface workflow selections to named surface frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, overload

from ..core.errors import ConfigResolutionError
from .models import ArtifactConfig, ArtifactVariant, ProblemSpec, StorageSpec, StudyConfig
from .selections import (
    AcquireWorkflowSelection,
    SurfaceWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
)
from .surfaces import SurfaceFrame
from .typed_registry import load_surface_frame


@dataclass(frozen=True, slots=True)
class AppliedSurfaceBase:
    surface_name: str
    chain: str
    dataset: str
    problem: str | ProblemSpec
    features: str | None
    storage: StorageSpec | None


@dataclass(frozen=True, slots=True)
class AppliedAcquireSurfaceSelection(AppliedSurfaceBase):
    provider: str
    dry_run: bool | None


@dataclass(frozen=True, slots=True)
class AppliedModelSurfaceSelection(AppliedSurfaceBase):
    dataset_builder: str
    prediction: str
    objective: str | None
    model: str | None
    evaluation: str | None
    training: str
    split: str
    tuning: str
    tuning_space: str | None
    study: StudyConfig | None
    artifact: ArtifactConfig | None


@dataclass(frozen=True, slots=True)
class AppliedTrainSurfaceSelection(AppliedModelSurfaceSelection):
    dataset_id: str | None
    study_id: str | None


@dataclass(frozen=True, slots=True)
class AppliedTuneSurfaceSelection(AppliedModelSurfaceSelection):
    dataset_id: str | None
    trial_count: int | None


AppliedSurfaceSelection: TypeAlias = (
    AppliedAcquireSurfaceSelection
    | AppliedTrainSurfaceSelection
    | AppliedTuneSurfaceSelection
)


@overload
def apply_surface_selection(
    selection: AcquireWorkflowSelection,
) -> AppliedAcquireSurfaceSelection: ...


@overload
def apply_surface_selection(
    selection: TrainWorkflowSelection,
) -> AppliedTrainSurfaceSelection: ...


@overload
def apply_surface_selection(
    selection: TuneWorkflowSelection,
) -> AppliedTuneSurfaceSelection: ...


@overload
def apply_surface_selection(
    selection: SurfaceWorkflowSelection,
) -> AppliedSurfaceSelection:
    ...


def apply_surface_selection(selection: SurfaceWorkflowSelection) -> AppliedSurfaceSelection:
    if selection.surface is None:
        raise ConfigResolutionError("surface is required")
    frame = load_surface_frame(selection.surface)
    if isinstance(selection, AcquireWorkflowSelection):
        return _apply_acquire_surface(selection.surface, frame, selection)
    if isinstance(selection, TrainWorkflowSelection):
        return _apply_train_surface(selection.surface, frame, selection)
    return _apply_tune_surface(selection.surface, frame, selection)


def _apply_acquire_surface(
    surface_name: str,
    frame: SurfaceFrame,
    selection: AcquireWorkflowSelection,
) -> AppliedAcquireSurfaceSelection:
    base = _surface_base(surface_name, frame, selection)
    return AppliedAcquireSurfaceSelection(
        surface_name=base.surface_name,
        chain=base.chain,
        dataset=base.dataset,
        problem=base.problem,
        features=base.features,
        storage=base.storage,
        provider=selection.provider or frame.acquisition.provider,
        dry_run=selection.dry_run,
    )


def _apply_train_surface(
    surface_name: str,
    frame: SurfaceFrame,
    selection: TrainWorkflowSelection,
) -> AppliedTrainSurfaceSelection:
    model = _model_surface(surface_name, frame, selection)
    return AppliedTrainSurfaceSelection(
        surface_name=model.surface_name,
        chain=model.chain,
        dataset=model.dataset,
        problem=model.problem,
        features=model.features,
        storage=model.storage,
        dataset_builder=model.dataset_builder,
        prediction=model.prediction,
        objective=model.objective,
        model=model.model,
        evaluation=model.evaluation,
        training=model.training,
        split=model.split,
        tuning=model.tuning,
        tuning_space=model.tuning_space,
        study=model.study,
        artifact=model.artifact,
        dataset_id=selection.dataset_id,
        study_id=selection.study_id,
    )


def _apply_tune_surface(
    surface_name: str,
    frame: SurfaceFrame,
    selection: TuneWorkflowSelection,
) -> AppliedTuneSurfaceSelection:
    model = _model_surface(surface_name, frame, selection)
    return AppliedTuneSurfaceSelection(
        surface_name=model.surface_name,
        chain=model.chain,
        dataset=model.dataset,
        problem=model.problem,
        features=model.features,
        storage=model.storage,
        dataset_builder=model.dataset_builder,
        prediction=model.prediction,
        objective=model.objective,
        model=model.model,
        evaluation=model.evaluation,
        training=model.training,
        split=model.split,
        tuning=model.tuning,
        tuning_space=model.tuning_space,
        study=model.study,
        artifact=model.artifact,
        dataset_id=selection.dataset_id,
        trial_count=selection.trial_count,
    )


def _surface_base(
    surface_name: str,
    frame: SurfaceFrame,
    selection: SurfaceWorkflowSelection,
) -> AppliedSurfaceBase:
    return AppliedSurfaceBase(
        surface_name=surface_name,
        chain=selection.chain or frame.chain,
        dataset=frame.dataset,
        problem=selection.problem or frame.problem,
        features=selection.features or frame.features,
        storage=(
            frame.storage
            if selection.storage_root is None
            else StorageSpec(root=selection.storage_root)
        ),
    )


def _model_surface(
    surface_name: str,
    frame: SurfaceFrame,
    selection: TrainWorkflowSelection | TuneWorkflowSelection,
) -> AppliedModelSurfaceSelection:
    base = _surface_base(surface_name, frame, selection)
    artifact = frame.artifact
    if isinstance(selection, TrainWorkflowSelection) and selection.variant is not None:
        artifact = ArtifactConfig(variant=ArtifactVariant(selection.variant))
    return AppliedModelSurfaceSelection(
        surface_name=base.surface_name,
        chain=base.chain,
        dataset=base.dataset,
        problem=base.problem,
        features=base.features,
        storage=base.storage,
        dataset_builder=frame.dataset_builder,
        prediction=frame.prediction,
        objective=selection.objective or frame.objective,
        model=selection.model or frame.model,
        evaluation=selection.evaluation or frame.evaluation.id,
        training=selection.training or frame.training.id,
        split=selection.split or frame.training.split,
        tuning=selection.tuning or frame.tuning.id,
        tuning_space=selection.tuning_space or frame.tuning.space,
        study=frame.study if selection.study is None else StudyConfig(name=selection.study),
        artifact=artifact,
    )
