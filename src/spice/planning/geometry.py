"""Timestamp-native temporal planning helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DelayWindow:
    lookback_seconds: int
    delay_seconds: int
    feature_history_seconds: int

    @property
    def required_history_seconds(self) -> int:
        return self.lookback_seconds + self.feature_history_seconds
