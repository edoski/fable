"""Internal PyTorch runtime helpers shared by training and inference."""

from __future__ import annotations

import os
import random
import subprocess
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.errors import SpiceOperatorError
from .batch_sources import BatchSource, resolve_available_device_memory_budget
from .models import ModelOutputs, TemporalModel
from .representations import RepresentationRuntimeContext

try:
    import psutil
except ImportError:  # pragma: no cover - fallback path is covered below
    psutil = None

IntVector = NDArray[np.int64]
_TORCHINDUCTOR_MIN_CUDA_SMS_FOR_AUTO_COMPILE = 68
ForwardBatchT = TypeVar("ForwardBatchT", bound="ForwardBatch")


@dataclass(frozen=True, slots=True)
class CudaModelingRuntime:
    resolved_device: torch.device
    representation_runtime_context: RepresentationRuntimeContext


class ForwardBatch(Protocol):
    def to_device(self, device: torch.device) -> ForwardBatch: ...

    def model_kwargs(self) -> Mapping[str, torch.Tensor]: ...


def resolve_cuda_device() -> torch.device:
    resolved_device = torch.device("cuda")
    require_cuda_device(resolved_device, requested_device="cuda")
    return resolved_device


def ensure_cuda_runtime_ready(device: torch.device) -> None:
    require_cuda_device(device, requested_device="cuda")
    _ensure_cuda_runtime_ready(device)


def build_cuda_modeling_runtime(*, batch_size: int) -> CudaModelingRuntime:
    resolved_device = resolve_cuda_device()
    ensure_cuda_runtime_ready(resolved_device)
    return CudaModelingRuntime(
        resolved_device=resolved_device,
        representation_runtime_context=build_representation_runtime_context(
            device=resolved_device,
            batch_size=batch_size,
        ),
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


def autocast_context(
    *,
    resolved_device: torch.device,
    precision: str,
):
    if precision == "32-true":
        return nullcontext()
    if precision == "16-mixed":
        if resolved_device.type != "cuda":
            raise ValueError("fp16 mixed precision is only supported on CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16-mixed":
        if resolved_device.type != "cuda":
            raise ValueError("bf16 mixed precision is only supported on CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported resolved precision: {precision}")


def require_cuda_device(
    device: torch.device,
    *,
    requested_device: str | None = None,
) -> None:
    if device.type != "cuda":
        resolved_label = requested_device if requested_device is not None else str(device)
        raise SpiceOperatorError(
            f"Modeling runtime requires CUDA devices. Received {resolved_label!r}."
        )
    if torch.version.hip is not None:
        raise SpiceOperatorError("Modeling runtime requires NVIDIA CUDA. ROCm/HIP is unsupported.")


def device_supports_auto_compile(device: torch.device) -> bool:
    require_cuda_device(device)
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
        batch_size=batch_size,
        available_host_memory_bytes=_available_system_memory_bytes(),
        available_device_memory_bytes=resolve_available_device_memory_budget(device),
    )


def run_model_forward_pass(
    model: TemporalModel,
    *,
    loader: BatchSource[ForwardBatchT],
    resolved_device: torch.device,
    precision: str,
    on_outputs: Callable[[ForwardBatchT, ModelOutputs], None],
) -> None:
    model.eval()
    with torch.no_grad():
        for batch in loader:
            device_batch = cast(ForwardBatchT, batch.to_device(resolved_device))
            with autocast_context(resolved_device=resolved_device, precision=precision):
                outputs = model(**device_batch.model_kwargs())
            on_outputs(device_batch, outputs)


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
