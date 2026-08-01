from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import UUID

import polars as pl
import pytest
import torch
from torch import nn

import experiments.inference_benchmark as benchmark
from fable.config import (
    BlockWindow,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainRequest,
)
from fable.min_block_fee import MinBlockFeeOutput, TargetState
from fable.modeling import ArtifactAssociation
from fable.temporal import FeatureState, HistoricalDataset, _HistoricalBacking

_K_STUDY_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_HELD_OUT_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_CORPUS_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_STUDY_ID = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
_METHOD = Method(
    model=LstmDefinition(family="lstm", hidden=1, layers=1, head_hidden=1, dropout=0.0),
    fit=FitMethod(
        learning_rate=0.001,
        weight_decay=0.0,
        accumulation=1,
        gradient_clip_norm=1.0,
        seed=1,
        max_epochs=1,
        validate_every_completed_epoch=1,
        patience=0,
        min_delta=0.0,
    ),
)


def _experiment(horizon: int) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(first_parent_block=10, last_parent_block=19),
        validation_window=BlockWindow(first_parent_block=30, last_parent_block=39),
        context_blocks=2,
        horizon_blocks=horizon,
        ordered_features=("log_base_fee_per_gas",),
    )


def _association(horizon: int, artifact_id: UUID) -> ArtifactAssociation:
    return ArtifactAssociation(
        request=TrainRequest(
            artifact_id=artifact_id,
            source=SelectedStudySource(
                corpus_id=_CORPUS_ID,
                study_id=_STUDY_ID,
                study_result_index=0,
                experiment=_experiment(horizon),
            ),
        ),
        feature_state=FeatureState(means=(0.0,), standard_deviations=(1.0,)),
        target_state=TargetState(mean=0.0, standard_deviation=1.0),
        method=_METHOD,
    )


def _request(index: int, horizon: int) -> EvaluateRequest:
    return EvaluateRequest(
        evaluation_id=UUID(f"20000000-0000-4000-8000-{index:012d}"),
        artifact_id=UUID(f"10000000-0000-4000-8000-{index:012d}"),
        corpus_id=_CORPUS_ID,
        testing_window=BlockWindow(
            first_parent_block=100, last_parent_block=100 + benchmark.ROLLING_HORIZONS[0] - horizon
        ),
    )


class _Dataset:
    def __init__(self, origins: tuple[int, ...]) -> None:
        self.origins = origins
        self.inputs = torch.arange(40, dtype=torch.float32).reshape(20, 2)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[index : index + 2],
            "origin_block": torch.tensor(self.origins[index]),
        }


class _Model(nn.Module):
    def __init__(self, horizon: int, events: list[str]) -> None:
        super().__init__()
        self.horizon = horizon
        self.events = events

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        self.events.append(f"model{self.horizon}:{int(inputs[0, 0, 0])}")
        return MinBlockFeeOutput(
            action_logits=torch.zeros(1, self.horizon), minimum_fee_z=torch.zeros(1)
        )


def _cell(events: list[str]) -> benchmark._Cell:
    return benchmark._Cell(
        name="ethereum.lstm",
        horizons={
            horizon: benchmark._Horizon(
                model=_Model(horizon, events),
                dataset=cast(
                    HistoricalDataset,
                    _Dataset(tuple(range(100, 101 + benchmark.ROLLING_HORIZONS[0] - horizon))),
                ),
            )
            for horizon in reversed(benchmark.ROLLING_HORIZONS)
        },
    )


def _resolved() -> dict[str, dict[int, EvaluateRequest]]:
    groups = ("ethereum.lstm", *(f"chain{index}.family" for index in range(1, 9)))
    return {
        group: {
            horizon: _request(
                group_index * len(benchmark.ROLLING_HORIZONS) + horizon_index, horizon
            )
            for horizon_index, horizon in enumerate(reversed(benchmark.ROLLING_HORIZONS))
        }
        for group_index, group in enumerate(groups)
    }


def _protocol() -> benchmark.Protocol:
    return benchmark._protocol(
        _K_STUDY_ID, _HELD_OUT_ID, _resolved(), warmup_iterations=2, sweeps=1
    )


def test_resolve_derives_complete_groups_and_joins_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k_study = {}
    held_out = {}
    source = _resolved()
    for group, requests in source.items():
        for horizon, request in requests.items():
            label = f"{group}.K{horizon}"
            k_study[label] = request.artifact_id
            held_out[label] = request.evaluation_id
            path = tmp_path / "evaluations" / str(request.evaluation_id) / "evaluation.json"
            path.parent.mkdir(parents=True)
            path.write_text(request.model_dump_json())
    k_study["ethereum.lstm.K10"] = UUID("30000000-0000-4000-8000-000000000010")
    held_out["ethereum.lstm.K10"] = UUID("40000000-0000-4000-8000-000000000010")

    monkeypatch.setattr(
        benchmark,
        "load_experiment_manifest",
        lambda root, kind, experiment_id: (
            k_study if kind == benchmark.ExperimentKind.K_STUDY else held_out
        ),
    )
    monkeypatch.setattr(benchmark, "load_artifact", lambda *_args: pytest.fail("model loaded"))

    resolved = benchmark._resolve(tmp_path, _K_STUDY_ID, _HELD_OUT_ID)

    assert tuple(resolved) == tuple(sorted(source))
    assert all(set(group) == set(benchmark.ROLLING_HORIZONS) for group in resolved.values())
    label = "ethereum.lstm.K5"
    original = k_study[label]
    k_study[label] = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
    with pytest.raises(ValueError, match="does not name"):
        benchmark._resolve(tmp_path, _K_STUDY_ID, _HELD_OUT_ID)

    k_study[label] = original
    held_out.pop(label)
    with pytest.raises(ValueError, match="exactly nine"):
        benchmark._resolve(tmp_path, _K_STUDY_ID, _HELD_OUT_ID)


def test_batch_one_is_a_chronological_view() -> None:
    backing = _HistoricalBacking(
        first_block=100,
        inputs=torch.arange(20, dtype=torch.float32).reshape(10, 2),
        base_fees=torch.arange(100, 110, dtype=torch.int64),
    )
    dataset = HistoricalDataset(
        backing,
        _experiment(2),
        BlockWindow(first_parent_block=102, last_parent_block=104),
        TargetState(mean=0.0, standard_deviation=1.0),
    )
    origin, inputs = benchmark._batch(benchmark._Horizon(nn.Identity(), dataset), 1)
    assert origin == 103
    assert inputs.shape == (1, 2, 2)
    assert inputs.untyped_storage().data_ptr() == backing.inputs.untyped_storage().data_ptr()


def test_cell_load_keeps_four_canonical_models_and_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolved()["ethereum.lstm"]
    loaded_artifacts: list[UUID] = []
    corpus_loads = 0

    def load_artifact(root: Path, artifact_id: UUID) -> tuple[ArtifactAssociation, nn.Module]:
        del root
        loaded_artifacts.append(artifact_id)
        horizon = next(
            horizon for horizon, request in resolved.items() if request.artifact_id == artifact_id
        )
        return _association(horizon, artifact_id), _Model(horizon, [])

    def load_corpus(root: Path, corpus_id: UUID) -> object:
        nonlocal corpus_loads
        del root, corpus_id
        corpus_loads += 1
        return object()

    monkeypatch.setattr(benchmark, "load_artifact", load_artifact)
    monkeypatch.setattr(benchmark, "load_corpus_blocks", load_corpus)
    monkeypatch.setattr(
        benchmark,
        "prepare_historical_window",
        lambda blocks, experiment, window, **_states: cast(
            HistoricalDataset,
            _Dataset(tuple(range(window.first_parent_block, window.last_parent_block + 1))),
        ),
    )

    cell = benchmark._load_cell(Path("/storage"), "ethereum.lstm", resolved)

    assert loaded_artifacts == [
        resolved[horizon].artifact_id for horizon in reversed(benchmark.ROLLING_HORIZONS)
    ]
    assert corpus_loads == 1
    assert set(cell.horizons) == set(benchmark.ROLLING_HORIZONS)
    assert all(not item.model.training for item in cell.horizons.values())

    monkeypatch.setattr(
        benchmark,
        "load_artifact",
        lambda root, artifact_id: (
            _association(benchmark.ROLLING_HORIZONS[0], artifact_id),
            nn.Identity(),
        ),
    )
    with pytest.raises(ValueError, match=r"ethereum\.lstm\.K2"):
        benchmark._load_cell(Path("/storage"), "ethereum.lstm", resolved)


def test_warmup_is_fixed_and_excluded_from_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        benchmark.time,
        "perf_counter_ns",
        lambda: pytest.fail("warmup must not read the measurement clock"),
    )
    benchmark._warm(_cell(events), 2)
    assert len(events) == 2 * len(benchmark.ROLLING_HORIZONS)


def test_timing_uses_outer_clocks_same_origins_and_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    ticks = iter(range(0, 1_000, 10))

    def clock() -> int:
        events.append("clock")
        return next(ticks)

    original_decode = benchmark.decode_action

    def decode(output: MinBlockFeeOutput) -> torch.Tensor:
        events.append("decode")
        return original_decode(output)

    monkeypatch.setattr(benchmark.time, "perf_counter_ns", clock)
    monkeypatch.setattr(benchmark, "decode_action", decode)
    rows = benchmark._time_cell(_cell(events), 1)

    assert rows.columns == ["cell", "sweep", "pass_order", "workload", "origin_block", "elapsed_ns"]
    assert rows.dtypes == [pl.String, pl.Int64, pl.Int64, pl.String, pl.Int64, pl.Int64]
    expected_calls = (
        sum(1 + benchmark.ROLLING_HORIZONS[0] - horizon for horizon in benchmark.ROLLING_HORIZONS)
        + 1
    )
    assert rows["elapsed_ns"].to_list() == [10] * expected_calls
    assert rows.filter(pl.col("workload") == "cascade")["origin_block"].to_list() == [100]
    assert events[-10:] == [
        "clock",
        "model5:0",
        "decode",
        "model4:0",
        "decode",
        "model3:0",
        "decode",
        "model2:0",
        "decode",
        "clock",
    ]


def test_orders_rotate_deterministically() -> None:
    cells = ("alpha.family", "beta.family", "gamma.family")
    assert benchmark._rotate(cells, 1) == ("beta.family", "gamma.family", "alpha.family")
    first = benchmark._pass_order(1)
    assert tuple(workload.horizons for workload in first) == tuple(
        (horizon,) for horizon in reversed(benchmark.ROLLING_HORIZONS)
    ) + (benchmark.ROLLING_HORIZONS,)
    assert benchmark._pass_order(2) == (*first[1:], first[0])


def test_protocol_match_atomic_publication_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    output = tmp_path / "output"
    benchmark._ensure_protocol(output, protocol)
    benchmark._ensure_protocol(output, protocol)
    assert (
        benchmark.Protocol.model_validate_json((output / "protocol.json").read_bytes(), strict=True)
        == protocol
    )
    with pytest.raises(ValueError, match="does not match"):
        benchmark._ensure_protocol(output, protocol.model_copy(update={"warmup_iterations": 3}))

    target = tmp_path / "unit"
    benchmark._publish(target, lambda path: path.write_text("first"))
    with pytest.raises(FileExistsError):
        benchmark._publish(target, lambda path: path.write_text("second"))
    interrupted = tmp_path / "interrupted"

    def interrupt(path: Path) -> None:
        path.write_text("partial")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        benchmark._publish(interrupted, interrupt)
    assert target.read_text() == "first"
    assert not interrupted.exists()
    assert not list(tmp_path.glob(".interrupted.*.tmp"))

    calls = 0

    def load(*args: object) -> benchmark._Cell:
        nonlocal calls
        calls += 1
        return _cell([])

    monkeypatch.setattr(benchmark, "_load_cell", load)
    unit_output = tmp_path / "campaign"
    benchmark._ensure_protocol(unit_output, protocol)
    for _ in range(2):
        benchmark._run_unit(
            tmp_path, unit_output, protocol, "ethereum.lstm", 1, _resolved()["ethereum.lstm"]
        )
    assert calls == 1
    assert pl.read_parquet(
        unit_output / "latency" / "ethereum.lstm" / "sweep-001.parquet"
    ).columns == ["cell", "sweep", "pass_order", "workload", "origin_block", "elapsed_ns"]


def test_protocol_is_only_the_derived_campaign_inputs() -> None:
    protocol = _protocol()
    assert tuple(protocol.model_dump()) == (
        "k_study_experiment_id",
        "held_out_experiment_id",
        "rolling_horizons",
        "roster",
        "warmup_iterations",
        "sweeps",
    )
    assert protocol.rolling_horizons == benchmark.ROLLING_HORIZONS
    assert len(protocol.roster) == 36
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        benchmark._protocol(_K_STUDY_ID, _HELD_OUT_ID, _resolved(), warmup_iterations=1, sweeps=0)
