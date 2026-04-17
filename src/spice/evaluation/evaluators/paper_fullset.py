"""Deterministic paper-style full-set evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...core.reporting import NullReporter, Reporter
from ...temporal.problem_store import CompiledProblemStore
from ..base import EvaluationSummary, EvaluatorConfig
from ..contracts import CompiledEvaluatorContract
from ..registry import EvaluatorSpec, register_evaluator_spec
from .shared import EVALUATION_METRIC_DESCRIPTORS, summarize_runs, summarize_selected_costs

IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PaperFullsetCompiledEvaluator(CompiledEvaluatorContract):
    def run(
        self,
        store: CompiledProblemStore,
        decoded_offsets: object,
        sample_indices: IntVector,
        reporter: Reporter | None,
    ) -> EvaluationSummary:
        reporter = reporter or NullReporter()
        if not isinstance(decoded_offsets, list):
            raise TypeError("paper_fullset decoded_offsets must be a list")
        if sample_indices.size == 0:
            raise ValueError("sample_indices must be non-empty")
        task_id = reporter.start_task("evaluate full set")
        run = summarize_selected_costs(
            store,
            decoded_offsets,
            sample_indices,
            np.arange(sample_indices.shape[0], dtype=np.int64),
            metadata={"mode": "fullset"},
        )
        reporter.finish_task(task_id, message=f"events={run.n_events}")
        return summarize_runs([run])


class PaperFullsetEvaluatorConfig(EvaluatorConfig):
    id: Literal["paper_fullset"] = "paper_fullset"


def _compile(config: PaperFullsetEvaluatorConfig) -> CompiledEvaluatorContract:
    return PaperFullsetCompiledEvaluator(
        evaluator_id="paper_fullset",
        metric_descriptors=EVALUATION_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config_payload=config.model_dump(mode="json", exclude_none=True),
    )


register_evaluator_spec(
    EvaluatorSpec(
        id="paper_fullset",
        config_type=PaperFullsetEvaluatorConfig,
        compile=_compile,
    )
)
