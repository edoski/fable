"""Operator-edge workflow command selection construction."""

from __future__ import annotations

from pathlib import Path

from .models import WorkflowTask
from .selections import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
    workflow_selection_payload,
)


def build_acquire_command_selection(
    *,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    provider: str | None,
    storage_root: Path | None,
    dry_run: bool | None,
) -> AcquireWorkflowSelection:
    return AcquireWorkflowSelection(
        surface=surface,
        chain=chain,
        problem=problem,
        features=features,
        provider=provider,
        storage_root=storage_root,
        dry_run=dry_run,
    )


def build_train_command_selection(
    *,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    objective: str | None,
    evaluation: str | None,
    model: str | None,
    tuning_space: str | None,
    training: str | None,
    split: str | None,
    tuning: str | None,
    study: str | None,
    dataset_id: str | None = None,
    study_id: str | None = None,
    variant: str | None = None,
) -> TrainWorkflowSelection:
    return TrainWorkflowSelection.model_validate(
        workflow_selection_payload(
            WorkflowTask.TRAIN,
            {
                "surface": surface,
                "chain": chain,
                "problem": problem,
                "features": features,
                "objective": objective,
                "evaluation": evaluation,
                "model": model,
                "tuning_space": tuning_space,
                "training": training,
                "split": split,
                "tuning": tuning,
                "study": study,
                "dataset_id": dataset_id,
                "study_id": study_id,
                "variant": variant,
            },
        )
    )


def build_tune_command_selection(
    *,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    objective: str | None,
    evaluation: str | None,
    model: str | None,
    tuning_space: str | None,
    training: str | None,
    split: str | None,
    tuning: str | None,
    study: str | None,
    dataset_id: str | None = None,
    trial_count: int | None = None,
) -> TuneWorkflowSelection:
    return TuneWorkflowSelection.model_validate(
        workflow_selection_payload(
            WorkflowTask.TUNE,
            {
                "surface": surface,
                "chain": chain,
                "problem": problem,
                "features": features,
                "objective": objective,
                "evaluation": evaluation,
                "model": model,
                "tuning_space": tuning_space,
                "training": training,
                "split": split,
                "tuning": tuning,
                "study": study,
                "dataset_id": dataset_id,
                "trial_count": trial_count,
            },
        )
    )


def build_evaluate_command_selection(
    *,
    artifact_id: str | None,
    dataset_id: str | None,
    evaluation: str | None,
    delay_seconds: int | None,
    batch_size: int | None,
) -> EvaluateWorkflowSelection:
    return EvaluateWorkflowSelection(
        storage_root=None,
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        evaluation=evaluation,
        delay_seconds=delay_seconds,
        batch_size=batch_size,
    )
