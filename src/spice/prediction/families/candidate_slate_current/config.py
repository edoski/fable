"""Current candidate-slate family config."""

from __future__ import annotations

from typing import Literal

from ...base import PredictionFamilyConfig


class CandidateSlateCurrentFamilyConfig(PredictionFamilyConfig):
    id: Literal["candidate_slate_current"] = "candidate_slate_current"
