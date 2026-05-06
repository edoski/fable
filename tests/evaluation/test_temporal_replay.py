from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from spice.core.errors import SpiceOperatorError
from spice.evaluation.config import (
    FullTemporalReplayEvaluatorConfig,
    PoissonReplayEvaluatorConfig,
)
from spice.evaluation.full_temporal_replay import FullTemporalReplayAdapter
from spice.evaluation.poisson_replay import (
    PoissonReplayAdapter,
    _select_sample_positions_for_arrivals,
)
from spice.evaluation.temporal_replay_runner import (
    TemporalReplaySampleView,
    TemporalReplaySelection,
    run_temporal_replay,
    temporal_replay_sample_view,
)
from spice.prediction.decoded_offsets import DecodedOffsets
from spice.temporal import coerce_execution_policy_config, compile_execution_policy_contract
from spice.temporal.problem_store import CompiledProblemStore


def _store() -> CompiledProblemStore:
    return CompiledProblemStore(
        feature_matrix=np.zeros((10, 1), dtype=np.float32),
        log_base_fees=np.log(
            np.array([100, 90, 80, 70, 60, 50, 45, 40, 35, 30], dtype=np.float32)
        ).astype(np.float32, copy=False),
        timestamps=(np.arange(10, dtype=np.int64) * 60).astype(np.int64, copy=False),
        anchor_rows=np.array([1, 3, 5, 7], dtype=np.int64),
        context_start_rows=np.array([0, 2, 4, 6], dtype=np.int64),
        candidate_start_rows=np.array([2, 4, 6, 8], dtype=np.int64),
        candidate_end_rows=np.array([4, 6, 8, 9], dtype=np.int64),
        max_candidate_slots=2,
    )


def _execution_policy():
    return compile_execution_policy_contract(
        coerce_execution_policy_config({"id": "strict_deadline_miss"})
    )


def test_poisson_arrival_selection_uses_latest_prior_sample() -> None:
    selected = _select_sample_positions_for_arrivals(
        np.array([10, 20, 30], dtype=np.int64),
        np.array([5.0, 10.0, 19.9, 20.0, 25.0, 30.0, 31.0], dtype=np.float64),
    )

    assert selected.tolist() == [0, 0, 1, 1, 2, 2]


def test_poisson_adapter_returns_positions_in_original_sample_order() -> None:
    store = _store()
    sample_indices = np.array([3, 2, 1, 0], dtype=np.int64)
    adapter = PoissonReplayAdapter(
        PoissonReplayEvaluatorConfig(
            id="poisson_replay",
            window_seconds=180,
            arrival_rate_per_second=0.08,
            repetitions=2,
            seed=7,
        )
    )

    samples = temporal_replay_sample_view(store, sample_indices)
    selections = adapter.selections(samples)

    assert selections
    for selection in selections:
        sample_timestamps = samples.sample_timestamps[selection.selected_positions]
        assert sample_timestamps.tolist() == sorted(sample_timestamps.tolist())


def test_full_temporal_replay_adapter_selects_every_sample_once() -> None:
    samples = TemporalReplaySampleView(
        sample_positions=np.array([0, 1, 2], dtype=np.int64),
        sample_timestamps=np.array([180, 60, 120], dtype=np.int64),
        sample_count=3,
    )
    adapter = FullTemporalReplayAdapter(
        FullTemporalReplayEvaluatorConfig(id="full_temporal_replay")
    )

    selections = adapter.selections(samples)

    assert len(selections) == 1
    assert selections[0].selected_positions.tolist() == [0, 1, 2]


@dataclass(frozen=True, slots=True)
class _FakeReplayAdapter:
    replay_selections: tuple[TemporalReplaySelection, ...]

    def selections(self, samples):
        del samples
        return self.replay_selections


class _CapturingReplayAdapter:
    def __init__(self) -> None:
        self.samples: TemporalReplaySampleView | None = None

    def selections(self, samples: TemporalReplaySampleView):
        self.samples = samples
        return (
            TemporalReplaySelection(
                selected_positions=np.array([0], dtype=np.int64),
                metadata={},
            ),
        )


def test_temporal_replay_runner_builds_adapter_sample_view() -> None:
    adapter = _CapturingReplayAdapter()
    run_temporal_replay(
        _store(),
        _execution_policy(),
        DecodedOffsets(torch.tensor([0, 1, 0], dtype=torch.int64)),
        np.array([3, 1, 2], dtype=np.int64),
        adapter=adapter,
    )

    assert adapter.samples is not None
    assert adapter.samples.sample_positions.tolist() == [0, 1, 2]
    assert adapter.samples.sample_timestamps.tolist() == [420, 180, 300]
    assert adapter.samples.sample_count == 3


def test_temporal_replay_runner_composes_adapter_runs() -> None:
    summary = run_temporal_replay(
        _store(),
        _execution_policy(),
        DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
        np.array([0, 1], dtype=np.int64),
        adapter=_FakeReplayAdapter(
            (
                TemporalReplaySelection(
                    selected_positions=np.array([0], dtype=np.int64),
                    metadata={"run": "first"},
                ),
                TemporalReplaySelection(
                    selected_positions=np.array([1], dtype=np.int64),
                    metadata={"run": "second"},
                ),
            )
        ),
    )

    assert summary.total_events == 2
    assert [run.metadata["run"] for run in summary.runs] == ["first", "second"]
    assert all("overflow_count" in run.metadata for run in summary.runs)


def test_temporal_replay_runner_owns_no_run_error() -> None:
    with pytest.raises(SpiceOperatorError, match="fake adapter"):
        run_temporal_replay(
            _store(),
            _execution_policy(),
            DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
            np.array([0, 1], dtype=np.int64),
            adapter=_FakeReplayAdapter(()),
            no_runs_error=SpiceOperatorError("fake adapter produced no runs"),
        )


@pytest.mark.parametrize(
    ("selected_positions", "message"),
    [
        (np.array([], dtype=np.int64), "non-empty"),
        (np.array([[0]], dtype=np.int64), "one-dimensional"),
        (np.array([0.0], dtype=np.float64), "integer"),
        (np.array([2], dtype=np.int64), "outside"),
    ],
)
def test_temporal_replay_runner_validates_selected_positions(
    selected_positions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_temporal_replay(
            _store(),
            _execution_policy(),
            DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
            np.array([0, 1], dtype=np.int64),
            adapter=_FakeReplayAdapter(
                (
                    TemporalReplaySelection(
                        selected_positions=selected_positions,
                        metadata={},
                    ),
                )
            ),
        )


def test_temporal_replay_runner_validates_decoded_sample_alignment() -> None:
    with pytest.raises(ValueError, match="decoded_offsets"):
        run_temporal_replay(
            _store(),
            _execution_policy(),
            DecodedOffsets(torch.tensor([0], dtype=torch.int64)),
            np.array([0, 1], dtype=np.int64),
            adapter=_FakeReplayAdapter(()),
        )


def test_temporal_replay_runner_validates_non_empty_sample_indices() -> None:
    with pytest.raises(ValueError, match="sample_indices"):
        run_temporal_replay(
            _store(),
            _execution_policy(),
            DecodedOffsets(torch.tensor([], dtype=torch.int64)),
            np.array([], dtype=np.int64),
            adapter=_FakeReplayAdapter(()),
        )


def test_temporal_replay_runner_validates_metadata_scalars() -> None:
    with pytest.raises(ValueError, match="metadata values"):
        run_temporal_replay(
            _store(),
            _execution_policy(),
            DecodedOffsets(torch.tensor([0, 1], dtype=torch.int64)),
            np.array([0, 1], dtype=np.int64),
            adapter=_FakeReplayAdapter(
                (
                    TemporalReplaySelection(
                        selected_positions=np.array([0], dtype=np.int64),
                        metadata={"enabled": True},
                    ),
                )
            ),
        )
