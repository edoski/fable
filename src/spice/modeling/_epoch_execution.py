"""Internal epoch and training-step execution helpers."""

from __future__ import annotations

from typing import cast

import torch

from ..metrics import MetricSet
from ..prediction import CompiledPredictionContract
from ..prediction.contracts import PredictionBatch
from ._runtime import precision_context
from .batch_plan import BatchSource
from .models import TemporalModel


def execute_training_batch(
    model: TemporalModel,
    batch: PredictionBatch,
    *,
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None,
    zero_after_step: bool = False,
) -> object:
    device_batch = batch.to_device(resolved_device)
    optimizer.zero_grad(set_to_none=True)
    with precision_context(precision=precision):
        outputs = model(**device_batch.model_kwargs())
        loss, batch_state = prediction_contract.compute_batch_loss_and_state(
            outputs,
            device_batch.targets,
            training_state=prediction_training_state,
        )
    loss.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    if zero_after_step:
        optimizer.zero_grad(set_to_none=True)
    return batch_state


def execute_validation_batch(
    model: TemporalModel,
    batch: PredictionBatch,
    *,
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
) -> object:
    device_batch = batch.to_device(resolved_device)
    with precision_context(precision=precision):
        outputs = model(**device_batch.model_kwargs())
        _, batch_state = prediction_contract.compute_batch_loss_and_state(
            outputs,
            device_batch.targets,
            training_state=prediction_training_state,
        )
    return batch_state


def run_epoch(
    model: TemporalModel,
    *,
    loader: BatchSource[PredictionBatch],
    resolved_device: torch.device,
    precision: str,
    prediction_contract: CompiledPredictionContract,
    prediction_training_state: object | None,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
    training: bool,
) -> MetricSet:
    accumulator = prediction_contract.create_epoch_accumulator()
    if training:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                if optimizer is None:
                    raise RuntimeError("optimizer is required for training epochs")
                batch_state = execute_training_batch(
                    model,
                    batch,
                    resolved_device=resolved_device,
                    precision=precision,
                    prediction_contract=prediction_contract,
                    prediction_training_state=prediction_training_state,
                    optimizer=optimizer,
                    gradient_clip_norm=gradient_clip_norm,
                )
            else:
                batch_state = execute_validation_batch(
                    model,
                    batch,
                    resolved_device=resolved_device,
                    precision=precision,
                    prediction_contract=prediction_contract,
                    prediction_training_state=prediction_training_state,
                )
            accumulator.update(cast(object, batch_state))
    return accumulator.finalize()
