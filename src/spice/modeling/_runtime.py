"""Internal PyTorch runtime helpers shared by training and inference."""

from __future__ import annotations

import random

import numpy as np
import torch
from numpy.typing import NDArray

from ..config import CompileMode, ModelConfig, TrainingConfig, TrainingPrecision
from ..data.datasets import TemporalDatasetStore
from . import torch_datasets as _torch_datasets  # noqa: F401
from .registry import resolve_auto_compile, resolve_default_precision, resolve_input_representation
from .representations import RepresentationLoader, build_representation_loader

IntVector = NDArray[np.int64]
_TORCHINDUCTOR_MIN_CUDA_SMS_FOR_AUTO_COMPILE = 68


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_trainer_precision(
    training_config: TrainingConfig,
    *,
    device: torch.device,
    model_config: ModelConfig,
) -> str:
    precision = training_config.precision
    if precision is TrainingPrecision.AUTO:
        precision = resolve_default_precision(model_config.id, device)

    if precision is TrainingPrecision.FP32:
        return "32-true"
    if precision is TrainingPrecision.FP16_MIXED:
        return "16-mixed"
    if precision is TrainingPrecision.BF16_MIXED:
        return "bf16-mixed"
    raise ValueError(f"Unsupported training precision: {precision}")


def resolve_compile_enabled(
    training_config: TrainingConfig,
    *,
    device: torch.device,
    precision: str,
    model_config: ModelConfig,
) -> bool:
    compile_mode = training_config.compile
    if compile_mode is CompileMode.AUTO:
        enabled = device.type in {"mps", "cuda"}
    else:
        enabled = compile_mode is CompileMode.ON
    if not enabled:
        return False
    if not resolve_auto_compile(model_config.id, device, precision):
        return False
    if compile_mode is CompileMode.AUTO and not _device_supports_auto_compile(device):
        return False
    return True


def _device_supports_auto_compile(device: torch.device) -> bool:
    if device.type != "cuda" or torch.version.hip:
        return True
    device_index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(device_index)
    # Mirror TorchInductor's "big GPU" threshold so auto mode skips the path
    # that immediately warns and falls back on smaller CUDA parts.
    return properties.multi_processor_count >= _TORCHINDUCTOR_MIN_CUDA_SMS_FOR_AUTO_COMPILE


def build_model_loader(
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    *,
    model_id: str,
    batch_size: int,
    shuffle: bool = False,
) -> RepresentationLoader:
    _ = _torch_datasets
    return build_representation_loader(
        resolve_input_representation(model_id),
        store,
        sample_indices,
        batch_size=batch_size,
        shuffle=shuffle,
    )
