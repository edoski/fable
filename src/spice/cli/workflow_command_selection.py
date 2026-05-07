"""CLI-owned Workflow Command Selection mapping."""

from __future__ import annotations

from pathlib import Path

from ..config import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
)


def acquire_workflow_selection(
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


def train_workflow_selection(
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
    variant: str | None,
    dataset_id: str | None,
    study_id: str | None,
) -> TrainWorkflowSelection:
    return TrainWorkflowSelection(
        surface=surface,
        chain=chain,
        problem=problem,
        features=features,
        objective=objective,
        evaluation=evaluation,
        model=model,
        tuning_space=tuning_space,
        training=training,
        split=split,
        tuning=tuning,
        study=study,
        dataset_id=dataset_id,
        study_id=study_id,
        variant=variant,
    )


def tune_workflow_selection(
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
    dataset_id: str | None,
    trial_count: int | None,
) -> TuneWorkflowSelection:
    return TuneWorkflowSelection(
        surface=surface,
        chain=chain,
        problem=problem,
        features=features,
        objective=objective,
        evaluation=evaluation,
        model=model,
        tuning_space=tuning_space,
        training=training,
        split=split,
        tuning=tuning,
        study=study,
        dataset_id=dataset_id,
        trial_count=trial_count,
    )


def evaluate_workflow_selection(
    *,
    artifact_id: str | None,
    dataset_id: str | None,
    evaluation: str | None,
    delay_seconds: int | None,
    batch_size: int | None,
) -> EvaluateWorkflowSelection:
    return EvaluateWorkflowSelection(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        evaluation=evaluation,
        delay_seconds=delay_seconds,
        batch_size=batch_size,
    )
