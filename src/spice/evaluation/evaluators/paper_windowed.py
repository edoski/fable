"""Random-window paper evaluator."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.reporting import NullReporter, Reporter
from ...prediction.contracts import DecodedOffsets
from ...temporal.problem_store import CompiledProblemStore
from ..base import CompiledEvaluatorContract, EvaluationSummary, EvaluatorConfig
from .shared import EVALUATION_METRIC_DESCRIPTORS, summarize_runs, summarize_selected_costs


class PaperWindowedEvaluatorConfig(EvaluatorConfig):
    id: str = "paper_windowed"
    window_seconds: int = Field(default=7200, gt=0)
    repetitions: int = Field(default=50, gt=0)
    seed: int = Field(default=2026, ge=0)


def _run(
    store: CompiledProblemStore,
    realization_policy,
    decoded_offsets: DecodedOffsets,
    sample_indices: np.ndarray,
    reporter: Reporter | None,
    *,
    config: PaperWindowedEvaluatorConfig,
) -> EvaluationSummary:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    sample_timestamps = store.timestamps[store.anchor_rows[sample_indices]]
    first_timestamp = int(sample_timestamps[0])
    last_timestamp = int(sample_timestamps[-1])
    rng = np.random.default_rng(config.seed)
    task_id = reporter.start_task("evaluate random paper windows", total=config.repetitions)
    runs = []
    if last_timestamp - first_timestamp < config.window_seconds:
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                np.arange(sample_indices.shape[0], dtype=np.int64),
                metadata={"mode": "fullset_fallback"},
            )
        )
        reporter.finish_task(task_id, message="fallback=fullset")
        return summarize_runs(runs)
    max_start = last_timestamp - config.window_seconds
    attempts = 0
    while len(runs) < config.repetitions and attempts < config.repetitions * 20:
        attempts += 1
        start_timestamp = int(rng.integers(first_timestamp, max_start + 1))
        end_timestamp = start_timestamp + config.window_seconds
        selected_positions = np.flatnonzero(
            (sample_timestamps >= start_timestamp) & (sample_timestamps < end_timestamp)
        ).astype(np.int64, copy=False)
        if selected_positions.size == 0:
            continue
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                selected_positions,
                metadata={
                    "mode": "windowed",
                    "window_start_timestamp": start_timestamp,
                    "window_end_timestamp": end_timestamp,
                    "repetition": len(runs) + 1,
                },
            )
        )
        reporter.update_task(
            task_id,
            completed=len(runs),
            message=f"runs={len(runs)}",
        )
    if not runs:
        raise ValueError("paper_windowed evaluator produced no non-empty windows")
    reporter.finish_task(task_id, message=f"runs={len(runs)}")
    return summarize_runs(runs)


def compile_evaluator(config: PaperWindowedEvaluatorConfig) -> CompiledEvaluatorContract:
    return CompiledEvaluatorContract(
        evaluator_id="paper_windowed",
        metric_descriptors=EVALUATION_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config_payload=config.model_dump(mode="json", exclude_none=True),
        run_fn=lambda store, realization_policy, decoded_offsets, sample_indices, reporter: _run(
            store,
            realization_policy,
            decoded_offsets,
            sample_indices,
            reporter,
            config=config,
        ),
    )
