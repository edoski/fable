"""Fixed runtime profile owned by the installed executable."""

from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

_Item = TypeVar("_Item")

FIT_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512

NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True


def data_loader(
    dataset: Dataset[_Item],
    *,
    batch_size: int,
    shuffle: bool,
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
    )


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
