"""Internal training fit-policy state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from ..core.errors import SpiceOperatorError
from ..metrics import MetricSet
from ..objectives import CompiledObjectiveContract
from .models import TemporalModel


@dataclass(frozen=True, slots=True)
class TrainingEpochProgress:
    epoch: int
    max_epochs: int
    train_metrics: MetricSet
    validation_metrics: MetricSet
    objective_metrics: MetricSet
    objective_metric_id: str
    direction: str
    best_epoch: int
    best_objective_value: float


@dataclass(frozen=True, slots=True)
class CompletedEpoch:
    epoch: int
    train_metrics: MetricSet
    validation_metrics: MetricSet
    objective_metrics: MetricSet


@dataclass(frozen=True, slots=True)
class FitPolicyDecision:
    should_stop: bool = False
    progress: TrainingEpochProgress | None = None
    early_stop: tuple[int, int] | None = None


@dataclass(slots=True)
class TrainingFitPolicy:
    objective_contract: CompiledObjectiveContract
    max_epochs: int
    patience: int
    min_delta: float
    train_history: list[MetricSet]
    validation_history: list[MetricSet]
    objective_history: list[MetricSet]
    best_state: dict[str, torch.Tensor] | None
    best_epoch: int
    epochs_without_improvement: int

    @classmethod
    def create(
        cls,
        *,
        objective_contract: CompiledObjectiveContract,
        max_epochs: int,
        patience: int,
        min_delta: float,
    ) -> TrainingFitPolicy:
        return cls(
            objective_contract=objective_contract,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            train_history=[],
            validation_history=[],
            objective_history=[],
            best_state=None,
            best_epoch=0,
            epochs_without_improvement=0,
        )

    def handle_nonfinite_metrics(
        self,
        *,
        epoch: int,
        phase: str,
        metrics: MetricSet,
    ) -> FitPolicyDecision | None:
        if _all_metrics_finite(metrics):
            return None
        if self.best_epoch > 0:
            return FitPolicyDecision(
                should_stop=True,
                early_stop=(epoch, self.best_epoch),
            )
        raise _nonfinite_metric_error(
            epoch=epoch,
            phase=phase,
            best_epoch=self.best_epoch,
        )

    def record_completed_epoch(
        self,
        completed: CompletedEpoch,
        *,
        model: TemporalModel,
    ) -> FitPolicyDecision:
        self.objective_history.append(completed.objective_metrics)
        self.train_history.append(completed.train_metrics)
        self.validation_history.append(completed.validation_metrics)

        if _is_improvement(
            current_epoch=completed.epoch,
            best_epoch=self.best_epoch,
            history=self.objective_history,
            direction=self.objective_contract.direction,
            metric_id=self.objective_contract.metric_id,
            min_delta=self.min_delta,
        ):
            self.best_state = _clone_cpu_state(_unwrap_compiled_model(model))
            self.best_epoch = completed.epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        best_value = self.objective_contract.value(
            self.objective_history[self.best_epoch - 1]
        )
        progress = TrainingEpochProgress(
            epoch=completed.epoch,
            max_epochs=self.max_epochs,
            train_metrics=completed.train_metrics,
            validation_metrics=completed.validation_metrics,
            objective_metrics=completed.objective_metrics,
            objective_metric_id=self.objective_contract.metric_id,
            direction=self.objective_contract.direction,
            best_epoch=self.best_epoch,
            best_objective_value=best_value,
        )
        if self.epochs_without_improvement >= self.patience:
            return FitPolicyDecision(
                should_stop=True,
                progress=progress,
                early_stop=(completed.epoch, self.best_epoch),
            )
        return FitPolicyDecision(progress=progress)

    def finalized_best(
        self,
        *,
        model: TemporalModel,
    ) -> tuple[int, dict[str, torch.Tensor], float]:
        if self.best_state is None:
            self.best_epoch = _best_epoch_from_objective_history(
                self.objective_history,
                objective_contract=self.objective_contract,
            )
            self.best_state = _clone_cpu_state(_unwrap_compiled_model(model))
        best_value = self.objective_contract.value(
            self.objective_history[self.best_epoch - 1]
        )
        return self.best_epoch, self.best_state, best_value


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


def _clone_cpu_state(model: TemporalModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _all_metrics_finite(metrics: MetricSet) -> bool:
    return all(np.isfinite(value) for value in metrics.values.values())


def _nonfinite_metric_error(
    *,
    epoch: int,
    phase: str,
    best_epoch: int,
) -> SpiceOperatorError:
    if best_epoch > 0:
        return SpiceOperatorError(
            f"Non-finite {phase} metrics at epoch {epoch}; "
            f"stopping and preserving best_epoch={best_epoch}"
        )
    return SpiceOperatorError(
        f"Non-finite {phase} metrics at epoch {epoch} before any valid best state"
    )


def _is_improvement(
    *,
    current_epoch: int,
    best_epoch: int,
    history: list[MetricSet],
    direction: str,
    metric_id: str,
    min_delta: float,
) -> bool:
    if best_epoch == 0:
        return True
    current_value = history[current_epoch - 1].require(metric_id)
    best_value = history[best_epoch - 1].require(metric_id)
    if direction == "maximize":
        return current_value > best_value + min_delta
    return current_value < best_value - min_delta


def _best_epoch_from_objective_history(
    history: list[MetricSet],
    *,
    objective_contract: CompiledObjectiveContract,
) -> int:
    if not history:
        return 1
    if objective_contract.direction == "maximize":
        winner = max(
            range(len(history)),
            key=lambda index: objective_contract.value(history[index]),
        )
    else:
        winner = min(
            range(len(history)),
            key=lambda index: objective_contract.value(history[index]),
        )
    return winner + 1
