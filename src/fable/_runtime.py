"""Fixed runtime profile owned by the installed executable."""

from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

_Item = TypeVar("_Item")

FIT_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512

NUM_WORKERS = 4


def data_loader(dataset: Dataset[_Item], *, batch_size: int, shuffle: bool) -> DataLoader[_Item]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if NUM_WORKERS else None,
        persistent_workers=bool(NUM_WORKERS),
    )


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
