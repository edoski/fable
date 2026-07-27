"""Fixed runtime profile owned by the installed executable."""

import os
from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

_Item = TypeVar("_Item")

FIT_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 64

NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

DETERMINISTIC = True
BENCHMARK = False
FLOAT32_MATMUL_PRECISION = "high"
CUDA_MATMUL_ALLOW_TF32 = True
CUDNN_ALLOW_TF32 = True


def data_loader(
    dataset: Dataset[_Item],
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[_Item]:
    workers = NUM_WORKERS
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR if workers else None,
        persistent_workers=PERSISTENT_WORKERS if workers else False,
        generator=generator,
    )


def configure_torch() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(DETERMINISTIC)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = BENCHMARK
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = CUDA_MATMUL_ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = CUDNN_ALLOW_TF32
