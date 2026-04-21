"""Canonical preset frame and request overlays."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..core.errors import ConfigResolutionError
from ..modeling.families.base import ConfigModel
from .models import (
    AcquisitionConfig,
    ArtifactConfig,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TrainingConfig,
    TuningConfig,
    WorkflowTask,
)
from .registry import load_named_group


class PresetFrame(ConfigModel):
    chain: str
    dataset: str
    provider: str
    problem: str
    dataset_builder: str
    feature_set: str
    prediction: str
    objective: str
    evaluation: str | None = None
    model: str
    tuning_space: str
    acquisition: AcquisitionConfig
    training: TrainingConfig
    split: SplitConfig
    delay_seconds: int = Field(gt=0)
    tuning: TuningConfig
    storage: StorageSpec | None = None
    study: StudyConfig | None = None
    artifact: ArtifactConfig | None = None


def load_preset_frame(name: str) -> PresetFrame:
    return PresetFrame.model_validate(load_named_group(name, "preset"))


def apply_request_overrides(
    frame: PresetFrame,
    *,
    workflow: WorkflowTask,
    chain: str | None,
    study: str | None,
    variant: str | None,
    delay_seconds: int | None,
    trial_count: int | None,
    storage_root: Path | None,
    dry_run: bool | None,
) -> PresetFrame:
    _reject_inapplicable_overrides(
        workflow=workflow,
        study=study,
        variant=variant,
        delay_seconds=delay_seconds,
        trial_count=trial_count,
        dry_run=dry_run,
    )
    updates: dict[str, object] = {}
    if chain is not None:
        updates["chain"] = chain
    if storage_root is not None:
        base_storage = frame.storage or StorageSpec()
        updates["storage"] = _updated_model(base_storage, root=storage_root)
    if workflow is WorkflowTask.ACQUIRE:
        if dry_run is not None:
            updates["acquisition"] = _updated_model(frame.acquisition, dry_run=dry_run)
        return frame.model_copy(update=updates)
    if study is not None:
        base_study = frame.study or StudyConfig()
        updates["study"] = _updated_model(base_study, name=study)
    if variant is not None:
        base_artifact = frame.artifact or ArtifactConfig()
        updates["artifact"] = _updated_model(base_artifact, variant=variant)
    if workflow is WorkflowTask.TUNE and trial_count is not None:
        updates["tuning"] = _updated_model(frame.tuning, trial_count=trial_count)
    if workflow is WorkflowTask.EVALUATE and delay_seconds is not None:
        updates["delay_seconds"] = delay_seconds
    return frame.model_copy(update=updates)


def _reject_inapplicable_overrides(
    *,
    workflow: WorkflowTask,
    study: str | None,
    variant: str | None,
    delay_seconds: int | None,
    trial_count: int | None,
    dry_run: bool | None,
) -> None:
    invalid: list[str] = []
    if workflow is WorkflowTask.ACQUIRE:
        if study is not None:
            invalid.append("study")
        if variant is not None:
            invalid.append("variant")
        if delay_seconds is not None:
            invalid.append("delay_seconds")
        if trial_count is not None:
            invalid.append("trial_count")
    elif workflow is WorkflowTask.TRAIN:
        if delay_seconds is not None:
            invalid.append("delay_seconds")
        if trial_count is not None:
            invalid.append("trial_count")
        if dry_run is not None:
            invalid.append("dry_run")
    elif workflow is WorkflowTask.TUNE:
        if variant is not None:
            invalid.append("variant")
        if delay_seconds is not None:
            invalid.append("delay_seconds")
        if dry_run is not None:
            invalid.append("dry_run")
    elif workflow is WorkflowTask.EVALUATE and dry_run is not None:
        invalid.append("dry_run")
    if invalid:
        joined = ", ".join(invalid)
        raise ConfigResolutionError(
            f"{workflow.value} request does not accept override fields: {joined}"
        )


def _updated_model(model: ConfigModel, **updates: object) -> ConfigModel:
    return type(model).model_validate(
        {
            **model.model_dump(mode="json", exclude_none=True),
            **updates,
        }
    )
