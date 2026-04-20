"""Internal PyTorch runtime helpers shared by training and inference."""

from __future__ import annotations

import os
import random
import subprocess
from contextlib import contextmanager

import numpy as np
import torch
from numpy.typing import NDArray

from ..config import CompileMode, ModelConfig, TrainingConfig, TrainingPrecision
from ..core.errors import SpiceOperatorError
from ..prediction import (
    CompiledPredictionContract,
    bind_prediction_representation,
)
from ..prediction.contracts import PredictionPreparedRepresentation
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from .batch_sources import (
    BatchSourcePlan,
    ModelInputBatchSourcePlan,
    plan_batch_source,
    plan_model_input_batch_source,
    resolve_available_device_memory_budget,
)
from .families.registry import (
    resolve_auto_compile,
    resolve_default_precision,
)
from .representations import (
    CompiledRepresentationContract,
    PreparedRepresentation,
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


def ensure_device_runtime_ready(*, requested_device: str, resolved_device: torch.device) -> None:
    if resolved_device.type == "cuda":
        _ensure_cuda_runtime_ready(resolved_device)
        return
    if requested_device != "auto" or torch.version.cuda is None:
        return
    try:
        cuda_device_count = torch.cuda.device_count()
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - covered through _ensure_cuda_runtime_ready
        raise _cuda_runtime_error(exc) from exc
    if cuda_device_count > 0 and not cuda_available:
        raise SpiceOperatorError(
            "CUDA devices are visible but the PyTorch CUDA runtime is unusable. "
            + _cuda_runtime_details()
        )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def configure_torch_backends(
    *,
    resolved_device: torch.device,
    deterministic: bool | None,
):
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cuda_matmul_allow_tf32 = (
        torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends, "cuda") else None
    )
    previous_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if resolved_device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            if previous_cuda_matmul_allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if deterministic is not None:
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
        yield
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = previous_cudnn_allow_tf32
        if previous_cuda_matmul_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_cuda_matmul_allow_tf32


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
    if compile_mode is CompileMode.OFF:
        return False
    if compile_mode is CompileMode.ON:
        return True
    if device.type not in {"mps", "cuda"}:
        return False
    if not resolve_auto_compile(model_config.id, device, precision):
        return False
    if not _device_supports_auto_compile(device):
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
        available_host_memory_bytes=_available_system_memory_bytes(),
        available_device_memory_bytes=resolve_available_device_memory_budget(device),
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


def prepare_supervised_prediction_representation(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    prediction_contract: CompiledPredictionContract,
    realization_policy: CompiledRealizationPolicyContract,
    runtime_context: RepresentationRuntimeContext,
) -> PredictionPreparedRepresentation:
    prepared = prepare_model_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        runtime_context=runtime_context,
    )
    targets = prediction_contract.prepare_targets(
        store,
        sample_indices,
        realization_policy=realization_policy,
    )
    return bind_prediction_representation(prepared, targets=targets)


def build_prediction_batch_source(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    prediction_contract: CompiledPredictionContract,
    realization_policy: CompiledRealizationPolicyContract,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
    shuffle: bool = False,
) -> BatchSourcePlan:
    prepared = prepare_supervised_prediction_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        realization_policy=realization_policy,
        runtime_context=runtime_context,
    )
    return plan_batch_source(
        prepared,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=seed,
        shuffle=shuffle,
    )


def build_inference_batch_source(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
) -> ModelInputBatchSourcePlan:
    prepared = prepare_model_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        runtime_context=runtime_context,
    )
    return plan_model_input_batch_source(
        prepared,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=seed,
        shuffle=False,
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


def _ensure_cuda_runtime_ready(device: torch.device) -> None:
    try:
        if not torch.cuda.is_available():
            raise SpiceOperatorError(
                "Resolved CUDA device, but torch.cuda.is_available() is False. "
                + _cuda_runtime_details()
            )
        device_index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.get_device_properties(device_index)
    except SpiceOperatorError:
        raise
    except Exception as exc:
        raise _cuda_runtime_error(exc) from exc


def _cuda_runtime_error(exc: Exception) -> SpiceOperatorError:
    return SpiceOperatorError(
        "CUDA runtime initialization failed. "
        + _cuda_runtime_details()
        + f" Root cause: {exc}"
    )


def _cuda_runtime_details() -> str:
    try:
        cuda_device_count = str(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - defensive fallback
        cuda_device_count = f"error:{exc}"
    return (
        f"torch={torch.__version__}; "
        f"torch.version.cuda={torch.version.cuda}; "
        f"cuda_device_count={cuda_device_count}"
    )
