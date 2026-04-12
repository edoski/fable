"""Derived task contracts shared across workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import ChainSpec, FeatureSetConfig, TaskSpec
from ..features import feature_warmup_blocks
from .geometry import DatasetGeometry, derive_dataset_geometry, minimum_history_context_blocks


@dataclass(frozen=True, slots=True)
class ResolvedTaskContract:
    task_id: str
    feature_set_id: str
    block_time_seconds: float
    lookback_seconds: int
    sample_count: int
    max_supported_delay_seconds: int
    feature_warmup_blocks: int
    required_history_context_blocks: int
    capability_geometry: DatasetGeometry

    @property
    def required_history_blocks(self) -> int:
        return self.capability_geometry.required_block_count(self.sample_count)

    @property
    def action_count(self) -> int:
        return self.capability_geometry.action_count

    @property
    def lookback_steps(self) -> int:
        return self.capability_geometry.lookback_steps

    def geometry_for_delay(self, requested_delay_seconds: int) -> DatasetGeometry:
        if requested_delay_seconds <= 0:
            raise ValueError("requested_delay_seconds must be positive")
        if requested_delay_seconds > self.max_supported_delay_seconds:
            raise ValueError(
                "requested_delay_seconds exceeds task capability: "
                f"{requested_delay_seconds} > {self.max_supported_delay_seconds}"
            )
        return derive_dataset_geometry(
            lookback_seconds=self.lookback_seconds,
            max_delay_seconds=requested_delay_seconds,
            block_time_seconds=self.block_time_seconds,
            history_context_blocks=self.required_history_context_blocks,
        )


def resolve_feature_contract(
    *,
    chain: ChainSpec,
    task: TaskSpec,
    feature_set_id: str,
    feature_names: tuple[str, ...],
) -> ResolvedTaskContract:
    warmup_blocks = feature_warmup_blocks(feature_names)
    required_history_context_blocks = minimum_history_context_blocks(
        lookback_seconds=task.lookback_seconds,
        block_time_seconds=chain.runtime.block_time_seconds,
        feature_warmup_blocks=warmup_blocks,
    )
    capability_geometry = derive_dataset_geometry(
        lookback_seconds=task.lookback_seconds,
        max_delay_seconds=task.max_supported_delay_seconds,
        block_time_seconds=chain.runtime.block_time_seconds,
        history_context_blocks=required_history_context_blocks,
    )
    return ResolvedTaskContract(
        task_id=task.id,
        feature_set_id=feature_set_id,
        block_time_seconds=chain.runtime.block_time_seconds,
        lookback_seconds=task.lookback_seconds,
        sample_count=task.sample_count,
        max_supported_delay_seconds=task.max_supported_delay_seconds,
        feature_warmup_blocks=warmup_blocks,
        required_history_context_blocks=required_history_context_blocks,
        capability_geometry=capability_geometry,
    )


def resolve_task_contract(
    *,
    chain: ChainSpec,
    task: TaskSpec,
    feature_set: FeatureSetConfig,
) -> ResolvedTaskContract:
    return resolve_feature_contract(
        chain=chain,
        task=task,
        feature_set_id=feature_set.id,
        feature_names=tuple(feature_set.outputs),
    )
