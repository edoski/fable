"""Internal PyTorch runtime helpers shared by training and inference."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from spice_temporal.contracts import SequenceBatch
from spice_temporal.datasets import TemporalDatasetStore
from spice_temporal.torch_datasets import SequenceDataset

IntVector = NDArray[np.int64]


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


def choose_microbatch_size(effective_batch_size: int, device: torch.device) -> int:
    candidates = [effective_batch_size, 32, 16, 8]
    if device.type == "cpu":
        return min(effective_batch_size, 16)
    for candidate in candidates:
        if candidate <= effective_batch_size:
            return candidate
    return 8


def accumulation_steps(effective_batch_size: int, microbatch_size: int) -> int:
    return max(1, math.ceil(effective_batch_size / microbatch_size))


def build_sequence_loader(
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    *,
    lookback_steps: int,
    effective_batch_size: int,
    device: torch.device,
) -> DataLoader[SequenceBatch]:
    microbatch_size = choose_microbatch_size(effective_batch_size, device)
    return DataLoader(
        SequenceDataset(store, sample_indices, lookback_steps=lookback_steps),
        batch_size=microbatch_size,
        shuffle=False,
    )
