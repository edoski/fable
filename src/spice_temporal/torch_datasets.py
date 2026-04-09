"""PyTorch dataset adapters."""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset

from spice_temporal.records import SupervisedExample


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Simple tensor adapter for sequence examples."""

    def __init__(self, examples: list[SupervisedExample]) -> None:
        if not examples:
            raise ValueError("SequenceDataset requires at least one example")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "inputs": torch.tensor(example.inputs, dtype=torch.float32),
            "class_label": torch.tensor(example.class_label, dtype=torch.long),
            "target_log_fee": torch.tensor(example.target_log_fee, dtype=torch.float32),
            "future_log_fees": torch.tensor(example.future_log_fees, dtype=torch.float32),
            "next_block_log_fee": torch.tensor(example.next_block_log_fee, dtype=torch.float32),
            "optimal_log_fee": torch.tensor(example.optimal_log_fee, dtype=torch.float32),
        }


def build_class_weights(examples: list[SupervisedExample], n_classes: int) -> torch.Tensor:
    counts = Counter(example.class_label for example in examples)
    if not counts:
        raise ValueError("Cannot build class weights for an empty example list")
    total = sum(counts.values())
    weights: list[float] = []
    for class_index in range(n_classes):
        class_count = counts.get(class_index, 0)
        if class_count == 0:
            weights.append(4.0)
            continue
        raw = total / (n_classes * class_count)
        weights.append(min(raw, 4.0))
    mean_weight = sum(weights) / len(weights)
    normalized = [weight / mean_weight for weight in weights]
    return torch.tensor(normalized, dtype=torch.float32)
