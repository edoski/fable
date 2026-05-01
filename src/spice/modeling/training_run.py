"""Neutral training run result envelope."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset_builders import PreparedTrainingDataset
from .models import TemporalModel
from .training_runner import TrainingResult


@dataclass(slots=True)
class TrainingRunResult:
    model: TemporalModel
    prepared: PreparedTrainingDataset
    training_result: TrainingResult
    prediction_training_state: object | None
