"""Timestamp-native problem compiler."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

from ...features import (
    CompiledFeatureContract,
    ResolvedFeatureTable,
)
from ..contracts import CompiledProblemContract, ProblemRuntimeMetadata, TimestampRuntimeMetadata
from ..problem_store import CompiledProblemStore
from ..realization import CompiledRealizationPolicyContract
from ._shared import build_timestamp_window_store
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config.models import ChainRuntimeSpec, ProblemSpec


class TimestampNativeCompilerConfig(ProblemCompilerConfig):
    id: str = "timestamp_native"


@dataclass(frozen=True, slots=True)
class TimestampNativeCompiledProblemContract(CompiledProblemContract):
    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_delay_seconds
        if recent_block_interval_seconds is None or recent_block_interval_seconds <= 0:
            return minimum_window
        warmup_seconds = ceil(self.warmup_rows * recent_block_interval_seconds)
        return max(
            minimum_window,
            self.required_history_seconds
            + self.max_delay_seconds
            + warmup_seconds
            + ceil(self.sample_count * recent_block_interval_seconds),
        )

    def count_valid_capability_samples(self, feature_table: ResolvedFeatureTable) -> int:
        store, _ = self.build_capability_store(feature_table)
        return store.n_samples

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> tuple[CompiledProblemStore, TimestampRuntimeMetadata]:
        return (
            build_timestamp_window_store(
                feature_table,
                feature_prerequisites=self.feature_prerequisites,
                lookback_seconds=self.lookback_seconds,
                delay_seconds=self.max_delay_seconds,
                requires_post_window_row=self.realization_policy.requires_post_window_row,
            ),
            TimestampRuntimeMetadata(),
        )

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        compiler_runtime_metadata: ProblemRuntimeMetadata,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        if not isinstance(compiler_runtime_metadata, TimestampRuntimeMetadata):
            raise TypeError("timestamp_native requires TimestampRuntimeMetadata")
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if delay_seconds > self.max_delay_seconds:
            raise ValueError(
                "delay_seconds exceeds problem capability: "
                f"{delay_seconds} > {self.max_delay_seconds}"
            )
        return build_timestamp_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=delay_seconds,
            max_candidate_slots=max_candidate_slots,
            requires_post_window_row=self.realization_policy.requires_post_window_row,
        )


def compile_problem(
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    realization_policy: CompiledRealizationPolicyContract,
    chain_runtime: ChainRuntimeSpec | None,
) -> CompiledProblemContract:
    del chain_runtime
    return TimestampNativeCompiledProblemContract(
        compiler_id="timestamp_native",
        problem_id=problem.id,
        feature_set_id=feature_contract.feature_set_id,
        feature_family_id=feature_contract.feature_family_id,
        lookback_seconds=problem.lookback_seconds,
        sample_count=problem.sample_count,
        max_delay_seconds=problem.max_delay_seconds,
        feature_prerequisites=feature_contract.feature_prerequisites,
        realization_policy=realization_policy,
    )
