"""Workflow active-config preparation."""

from __future__ import annotations

from typing import cast

from ..config.models import ArtifactVariant, TrainConfig
from ..modeling.tuning import apply_study_best_params
from ..storage.workflow_roots import TrainWorkflowRoots, TunedTrainWorkflowRoots


def active_train_config(config: TrainConfig, roots: TrainWorkflowRoots) -> TrainConfig:
    if config.artifact.variant is not ArtifactVariant.TUNED:
        return config
    assert isinstance(roots, TunedTrainWorkflowRoots)
    applied = apply_study_best_params(
        config,
        study=roots.study,
        corpus=roots.corpus,
    )
    return cast(TrainConfig, applied.config)
