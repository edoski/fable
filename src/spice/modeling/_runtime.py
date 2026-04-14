"""Internal PyTorch runtime helpers shared by training and inference."""

from __future__ import annotations

import os
import random
import subprocess

import numpy as np
import torch
from numpy.typing import NDArray

from ..config import CompileMode, ModelConfig, TrainingConfig, TrainingPrecision
from ..prediction import CompiledPredictionContract, bind_prediction_representation
from ..temporal.problem_store import CompiledProblemStore
from .families.registry import (
    resolve_auto_compile,
    resolve_default_precision,
)
from .representations import (
    CompiledRepresentationContract,
    PreparedRepresentation,
    PreparedRepresentationLoader,
    RepresentationRuntimeContext,
)

try:
    import psutil
except ImportError:  # pragma: no cover - fallback path is covered below
    psutil = None

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


def build_representation_runtime_context(
    *,
    device: torch.device,
    batch_size: int,
) -> RepresentationRuntimeContext:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return RepresentationRuntimeContext(
        device_type=device.type,
        batch_size=batch_size,
        available_memory_bytes=_available_system_memory_bytes(),
    )


def prepare_model_representation(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    runtime_context: RepresentationRuntimeContext,
) -> PreparedRepresentation:
    return representation_contract.prepare(
        store,
        sample_indices,
        runtime_context=runtime_context,
    )


def prepare_prediction_representation(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    prediction_contract: CompiledPredictionContract,
    runtime_context: RepresentationRuntimeContext,
):
    prepared = prepare_model_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        runtime_context=runtime_context,
    )
    targets = prediction_contract.prepare_targets(store, sample_indices)
    return bind_prediction_representation(prepared, targets=targets)


def build_model_loader(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    runtime_context: RepresentationRuntimeContext,
    seed: int,
    shuffle: bool = False,
) -> PreparedRepresentationLoader:
    return representation_contract.build_loader(
        store,
        sample_indices,
        runtime_context=runtime_context,
        seed=seed,
        shuffle=shuffle,
    )


def build_prediction_loader(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    prediction_contract: CompiledPredictionContract,
    runtime_context: RepresentationRuntimeContext,
    seed: int,
    shuffle: bool = False,
) -> PreparedRepresentationLoader:
    prepared = prepare_prediction_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        runtime_context=runtime_context,
    )
    return PreparedRepresentationLoader(
        prepared,
        seed=seed,
        shuffle=shuffle,
    )
def _available_system_memory_bytes() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().available)

    page_size = _sysconf_int("SC_PAGE_SIZE")
    available_pages = _sysconf_int("SC_AVPHYS_PAGES")
    if page_size is not None and available_pages is not None:
        return page_size * available_pages

    physical_pages = _sysconf_int("SC_PHYS_PAGES")
    if page_size is not None and physical_pages is not None:
        return page_size * physical_pages

    if os.name == "posix":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
            ).strip()
            return int(output)
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass

    return 8 * 1024**3


def _sysconf_int(name: str) -> int | None:
    try:
        return int(os.sysconf(name))
    except (AttributeError, OSError, ValueError):
        return None
