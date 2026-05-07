# pyright: strict

"""Candidate-offset decoded-result ABI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from .decoding import DecodedPredictionResult, IntVector, coerce_cpu_int64_vector

OFFSET_DECODED_RESULT_ID = "candidate_offsets"


class DecodedOffsets:
    """Prediction-owned decoded offset buffer backed by a CPU int64 tensor."""

    __slots__ = ("_tensor",)
    decoded_result_id = OFFSET_DECODED_RESULT_ID

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = coerce_cpu_int64_vector(tensor, label="decoded_offsets")

    @classmethod
    def allocate(cls, sample_count: int) -> DecodedOffsets:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        return cls(torch.zeros(sample_count, dtype=torch.int64))

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def __len__(self) -> int:
        return int(self._tensor.shape[0])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DecodedOffsets):
            return torch.equal(self._tensor, other._tensor)
        if isinstance(other, torch.Tensor):
            return torch.equal(
                self._tensor,
                coerce_cpu_int64_vector(other, label="decoded_offsets"),
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            sequence = cast(Sequence[object], other)
            values = [int(self._tensor[index].item()) for index in range(len(self))]
            return values == list(sequence)
        return NotImplemented

    def write(self, sample_positions: torch.Tensor, decoded: torch.Tensor) -> None:
        positions = coerce_cpu_int64_vector(sample_positions, label="sample_positions")
        values = coerce_cpu_int64_vector(decoded, label="decoded")
        if positions.shape != values.shape:
            raise ValueError("sample_positions and decoded must have matching shape")
        self._tensor.index_copy_(0, positions, values)

    def select(self, sample_positions: torch.Tensor | IntVector) -> IntVector:
        positions = coerce_cpu_int64_vector(sample_positions, label="sample_positions")
        return self._tensor.index_select(0, positions).numpy()


def require_decoded_offsets(result: DecodedPredictionResult) -> DecodedOffsets:
    if not isinstance(result, DecodedOffsets):
        raise TypeError("Evaluator requires candidate offset decoded results")
    return result
