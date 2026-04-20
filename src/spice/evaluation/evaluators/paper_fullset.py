"""Deterministic full-set evaluator."""

import numpy as np
from numpy.typing import NDArray

from ...core.reporting import NullReporter, Reporter
from ...prediction.contracts import DecodedOffsets
from ...temporal.problem_store import CompiledProblemStore
from ..base import CompiledEvaluatorContract, EvaluationSummary, EvaluatorConfig
from .shared import EVALUATION_METRIC_DESCRIPTORS, summarize_runs, summarize_selected_costs

IntVector = NDArray[np.int64]


def _run(
    store: CompiledProblemStore,
    realization_policy,
    decoded_offsets: DecodedOffsets,
    sample_indices: IntVector,
    reporter: Reporter | None,
) -> EvaluationSummary:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    task_id = reporter.start_task("evaluate full set")
    run = summarize_selected_costs(
        store,
        realization_policy,
        decoded_offsets,
        sample_indices,
        np.arange(sample_indices.shape[0], dtype=np.int64),
        metadata={"mode": "fullset"},
    )
    reporter.finish_task(task_id, message=f"events={run.n_events}")
    return summarize_runs([run])


class PaperFullsetEvaluatorConfig(EvaluatorConfig):
    id: str = "paper_fullset"


def compile_evaluator(config: PaperFullsetEvaluatorConfig) -> CompiledEvaluatorContract:
    return CompiledEvaluatorContract(
        evaluator_id="paper_fullset",
        metric_descriptors=EVALUATION_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config_payload=config.model_dump(mode="json", exclude_none=True),
        run_fn=_run,
    )
