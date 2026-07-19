"""Historical evaluation values retained for read-only decoding."""

from __future__ import annotations

from dataclasses import dataclass

EvaluationMetadataValue = str | int | float


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, EvaluationMetadataValue]
