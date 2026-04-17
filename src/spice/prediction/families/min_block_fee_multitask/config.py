"""Paper-faithful min-block-fee multitask family config."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...base import PredictionFamilyConfig


class MinBlockFeeMultitaskFamilyConfig(PredictionFamilyConfig):
    id: str = "min_block_fee_multitask"
    classification_loss_weight: float = Field(default=1.0, gt=0.0)
    regression_loss_weight: float = Field(default=1.0, gt=0.0)
    class_weighting: Literal["inverse_frequency"] = "inverse_frequency"
    fee_target_normalization: Literal["none", "zscore_train_split"] = "none"
