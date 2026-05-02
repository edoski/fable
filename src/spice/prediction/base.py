"""Shared prediction-family output contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionHeadSpec:
    id: str
    size: int


@dataclass(frozen=True, slots=True)
class PredictionOutputSpec:
    heads: tuple[PredictionHeadSpec, ...]
