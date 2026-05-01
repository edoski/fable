"""Full temporal replay evaluator."""

from __future__ import annotations

import numpy as np

from ..temporal.problem_store import CompiledProblemStore
from .config import FullTemporalReplayEvaluatorConfig
from .contracts import CompiledEvaluatorContract, IntVector
from .temporal_replay_runner import (
    TemporalReplaySelection,
    compile_temporal_replay_evaluator_contract,
)


class FullTemporalReplayAdapter:
    def __init__(self, config: FullTemporalReplayEvaluatorConfig) -> None:
        self.config = config

    def selections(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> list[TemporalReplaySelection]:
        del store
        return [
            TemporalReplaySelection(
                selected_positions=np.arange(sample_indices.shape[0], dtype=np.int64),
                metadata={
                    "mode": self.config.id,
                    "sample_count": int(sample_indices.shape[0]),
                },
            )
        ]

    def no_runs_error(self) -> Exception:
        return ValueError("evaluation produced no runs")


def compile_full_temporal_replay_evaluator_contract(
    config: FullTemporalReplayEvaluatorConfig,
) -> CompiledEvaluatorContract:
    return compile_temporal_replay_evaluator_contract(
        evaluation_id=config.id,
        config_payload=config.model_dump(mode="json", exclude_none=True),
        adapter=FullTemporalReplayAdapter(config),
    )
