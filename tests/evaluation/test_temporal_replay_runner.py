from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from spice.evaluation import EvaluationRun, EvaluationSummary
from spice.evaluation.temporal_replay_runner import (
    TemporalReplaySelection,
    run_temporal_replay,
)
from spice.prediction import MetricSet
from spice.prediction.decoded_offsets import DecodedOffsets
from spice.temporal import CompiledExecutionPolicyContract
from spice.temporal.problem_store import CompiledProblemStore


class _Adapter:
    def __init__(self, selections: list[TemporalReplaySelection]) -> None:
        self._selections = selections

    def selections(self, store, sample_indices):
        del store, sample_indices
        return self._selections

    def no_runs_error(self) -> Exception:
        return RuntimeError("adapter produced no runs")


def test_temporal_replay_runner_rejects_misaligned_decoded_offsets() -> None:
    with pytest.raises(ValueError, match="decoded_offsets must align"):
        run_temporal_replay(
            cast(CompiledProblemStore, SimpleNamespace()),
            cast(CompiledExecutionPolicyContract, SimpleNamespace()),
            DecodedOffsets(torch.tensor([0], dtype=torch.int64)),
            np.array([0, 1], dtype=np.int64),
            adapter=_Adapter([]),
        )


def test_temporal_replay_runner_rejects_empty_sample_indices() -> None:
    with pytest.raises(ValueError, match="sample_indices must be non-empty"):
        run_temporal_replay(
            cast(CompiledProblemStore, SimpleNamespace()),
            cast(CompiledExecutionPolicyContract, SimpleNamespace()),
            DecodedOffsets(torch.empty(0, dtype=torch.int64)),
            np.empty(0, dtype=np.int64),
            adapter=_Adapter([]),
        )


def test_temporal_replay_runner_raises_adapter_no_runs_error() -> None:
    with pytest.raises(RuntimeError, match="adapter produced no runs"):
        run_temporal_replay(
            cast(CompiledProblemStore, SimpleNamespace()),
            cast(CompiledExecutionPolicyContract, SimpleNamespace()),
            DecodedOffsets(torch.tensor([0], dtype=torch.int64)),
            np.array([0], dtype=np.int64),
            adapter=_Adapter([]),
        )


def test_temporal_replay_runner_passes_selection_metadata_to_accounting(
    monkeypatch,
) -> None:
    seen = {}
    selection = TemporalReplaySelection(
        selected_positions=np.array([0], dtype=np.int64),
        metadata={"mode": "test"},
    )
    summary = EvaluationSummary(
        metrics=MetricSet({"score": 1.0}),
        window_metrics={},
        total_events=1,
        runs=[EvaluationRun(n_events=1, metrics={"score": 1.0}, metadata={"mode": "test"})],
    )

    def fake_summarize_selected_temporal_decisions(
        store,
        execution_policy,
        decoded_offsets,
        sample_indices,
        selected_positions,
        *,
        metadata,
    ):
        seen.update(
            {
                "store": store,
                "execution_policy": execution_policy,
                "decoded_offsets": decoded_offsets,
                "sample_indices": sample_indices,
                "selected_positions": selected_positions,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(run="accounting-run")

    monkeypatch.setattr(
        "spice.evaluation.temporal_replay_runner.summarize_selected_temporal_decisions",
        fake_summarize_selected_temporal_decisions,
    )
    monkeypatch.setattr(
        "spice.evaluation.temporal_replay_runner.summarize_temporal_accounting_runs",
        lambda runs: summary if len(runs) == 1 else None,
    )
    decoded_offsets = DecodedOffsets(torch.tensor([0], dtype=torch.int64))
    sample_indices = np.array([0], dtype=np.int64)
    store = cast(CompiledProblemStore, SimpleNamespace(name="store"))
    execution_policy = cast(CompiledExecutionPolicyContract, SimpleNamespace(name="policy"))

    result = run_temporal_replay(
        store,
        execution_policy,
        decoded_offsets,
        sample_indices,
        adapter=_Adapter([selection]),
    )

    assert result is summary
    assert seen["store"] is store
    assert seen["execution_policy"] is execution_policy
    assert seen["decoded_offsets"] is decoded_offsets
    assert seen["sample_indices"] is sample_indices
    assert seen["selected_positions"] is selection.selected_positions
    assert seen["metadata"] == {"mode": "test"}


@pytest.mark.parametrize(
    ("selected_positions", "match"),
    [
        (np.array([[-1]], dtype=np.int64), "one-dimensional"),
        (np.array([], dtype=np.int64), "non-empty"),
        (np.array([-1], dtype=np.int64), "outside"),
        (np.array([1], dtype=np.int64), "outside"),
        (np.array([0.0], dtype=np.float64), "integer"),
    ],
)
def test_temporal_replay_runner_rejects_invalid_adapter_selections(
    selected_positions,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_temporal_replay(
            cast(CompiledProblemStore, SimpleNamespace()),
            cast(CompiledExecutionPolicyContract, SimpleNamespace()),
            DecodedOffsets(torch.tensor([0], dtype=torch.int64)),
            np.array([0], dtype=np.int64),
            adapter=_Adapter(
                [
                    TemporalReplaySelection(
                        selected_positions=selected_positions,
                        metadata={},
                    )
                ]
            ),
        )


def test_temporal_replay_runner_rejects_non_scalar_metadata() -> None:
    with pytest.raises(ValueError, match="metadata values must be scalar"):
        run_temporal_replay(
            cast(CompiledProblemStore, SimpleNamespace()),
            cast(CompiledExecutionPolicyContract, SimpleNamespace()),
            DecodedOffsets(torch.tensor([0], dtype=torch.int64)),
            np.array([0], dtype=np.int64),
            adapter=_Adapter(
                [
                    TemporalReplaySelection(
                        selected_positions=np.array([0], dtype=np.int64),
                        metadata={"bad": object()},
                    )
                ]
            ),
        )
