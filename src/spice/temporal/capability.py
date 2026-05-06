"""Temporal capability carried by trained artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.validation import validate_path_segment


@dataclass(frozen=True, slots=True)
class TemporalCapabilitySemantics:
    """Stable semantic projection of the trained artifact's temporal capability."""

    compiler_id: str
    max_delay_seconds: int
    action_width: int


@dataclass(frozen=True, slots=True)
class TemporalCapability:
    compiler_id: str
    max_delay_seconds: int
    action_width: int
    compiler_runtime_metadata: object

    def __post_init__(self) -> None:
        validate_path_segment(self.compiler_id, label="temporal_capability.compiler_id")
        if self.max_delay_seconds <= 0:
            raise ValueError("temporal_capability.max_delay_seconds must be positive")
        if self.action_width <= 0:
            raise ValueError("temporal_capability.action_width must be positive")

    @property
    def semantics(self) -> TemporalCapabilitySemantics:
        return TemporalCapabilitySemantics(
            compiler_id=self.compiler_id,
            max_delay_seconds=self.max_delay_seconds,
            action_width=self.action_width,
        )
