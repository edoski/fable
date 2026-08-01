from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
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
    CorpusDefinition,
    CorpusRequest,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainRequest,
    TuneRequest,
)
from fable.evaluation import OBSERVATION_SCHEMA
from fable.min_block_fee import MinBlockFeeOutput, TargetState
from fable.modeling import ArtifactAssociation
from fable.study import RetainedResult, Study
from fable.temporal import FeatureState, HistoricalDataset, _HistoricalBacking

_K_STUDY_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_HELD_OUT_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_CORPUS_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_STUDY_ID = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
_FIT = FitMethod(
    learning_rate=0.001,
    weight_decay=0.0,
    accumulation=1,
    gradient_clip_norm=1.0,
    seed=1,
    max_epochs=1,
    validate_every_completed_epoch=1,
    patience=0,
    min_delta=0.0,
)
_METHOD = Method(
    model=LstmDefinition(family="lstm", hidden=1, layers=1, head_hidden=1, dropout=0.0), fit=_FIT
)


def _experiment(horizon: int) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(first_parent_block=10, last_parent_block=19),
        validation_window=BlockWindow(first_parent_block=30, last_parent_block=39),
        context_blocks=2,
        horizon_blocks=horizon,
        ordered_features=("log_base_fee_per_gas",),
    )


def _association(horizon: int, artifact_id: UUID | None = None) -> ArtifactAssociation:
    artifact_id = artifact_id or UUID(f"10000000-0000-4000-8000-{horizon:012d}")
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


def _provenance(
    horizon: int,
    *,
    chain: str = "ethereum",
    family: str = "lstm",
    first: int = 100,
    last_k5: int = 100,
) -> benchmark.ArtifactProvenance:
    artifact_id = UUID(f"10000000-0000-4000-8000-{horizon:012d}")
    return benchmark.ArtifactProvenance(
        cell=f"{chain}.{family}.K{horizon}",
        chain=chain,
        family=family,
        horizon_blocks=horizon,
        artifact_id=artifact_id,
        evaluation_id=UUID(f"20000000-0000-4000-8000-{horizon:012d}"),
        corpus_id=_CORPUS_ID,
        study_id=_STUDY_ID,
        study_result_index=0,
        artifact_sha256=f"{horizon:064x}",
        testing_window=BlockWindow(
            first_parent_block=first, last_parent_block=last_k5 + 5 - horizon
        ),
        context_blocks=2,
        ordered_features=("log_base_fee_per_gas",),
        method=_METHOD,
    )


def _host() -> benchmark.HostMetadata:
    return benchmark.HostMetadata(
        hardware_model="Apple M2 Max",
        hardware_identifier="Mac14,5",
        machine="arm64",
        macos_version="26.5",
        macos_build="25F71",
        power_source="AC Power",
        power_mode="automatic",
        low_power_mode=False,
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
    )


def _protocol(*, sweeps: int = 1) -> benchmark.LatencyProtocol:
    return benchmark.LatencyProtocol(
        campaign_id=UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"),
        purpose="pilot",
        created_at_utc=datetime(2026, 8, 1, 10, tzinfo=UTC),
        repository_commit="f" * 40,
        python_version="3.11.14",
        pytorch_version=torch.__version__,
        host=_host(),
        k_study_experiment_id=_K_STUDY_ID,
        held_out_experiment_id=_HELD_OUT_ID,
        artifacts=tuple(_provenance(horizon) for horizon in benchmark._HORIZONS),
        origin_rule=benchmark._ORIGIN_RULE,
        warmup_rule=benchmark._WARMUP_RULE,
        order_rule=benchmark._ORDER_RULE,
        warmup_iterations=2,
        prediction_atol=1e-6,
        quantile_method="inverted_cdf",
        latency_sweeps=sweeps,
    )


def _observations(path: Path, request: EvaluateRequest, action: int = 0) -> None:
    origins = list(
        range(
            request.testing_window.first_parent_block, request.testing_window.last_parent_block + 1
        )
    )
    count = len(origins)
    pl.DataFrame(
        {
            "origin_block": origins,
            "predicted_action_k": [action] * count,
            "predicted_minimum_log_base_fee": [0.0] * count,
            "minimum_action_k": [0] * count,
            "immediate_base_fee_per_gas": [1] * count,
            "immediate_effective_priority_fee_per_gas_p50": [0] * count,
            "selected_base_fee_per_gas": [1] * count,
            "selected_effective_priority_fee_per_gas_p50": [0] * count,
            "deadline_base_fee_per_gas": [1] * count,
            "deadline_effective_priority_fee_per_gas_p50": [0] * count,
            "minimum_base_fee_per_gas": [1] * count,
        },
        schema=OBSERVATION_SCHEMA,
    ).write_parquet(path)


class _ViewDataset:
    def __init__(self, origins: tuple[int, ...]) -> None:
        self.origins = origins
        self.backing = torch.arange(40, dtype=torch.float32).reshape(20, 2)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.backing[index : index + 2],
            "origin_block": torch.tensor(self.origins[index]),
        }


class _TracingModel(nn.Module):
    def __init__(self, horizon: int, events: list[str]) -> None:
        super().__init__()
        self.horizon = horizon
        self.events = events

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        self.events.append(f"model{self.horizon}:{int(inputs[0, 0, 0])}")
        return MinBlockFeeOutput(
            action_logits=torch.arange(self.horizon, dtype=torch.float32).neg().unsqueeze(0),
            minimum_fee_z=torch.zeros(1),
        )


def _loaded_cell(events: list[str], *, canonical_action: int = 0) -> benchmark._LoadedCell:
    loaded = {}
    for horizon in benchmark._HORIZONS:
        origins = tuple(range(100, 101 + 5 - horizon))
        loaded[horizon] = benchmark._LoadedHorizon(
            provenance=_provenance(horizon),
            association=_association(horizon),
            model=_TracingModel(horizon, events),
            dataset=cast(HistoricalDataset, _ViewDataset(origins)),
            reference=benchmark._Reference(
                origins=origins,
                actions=(canonical_action,) * len(origins),
                minimum_logs=(0.0,) * len(origins),
            ),
        )
    return benchmark._LoadedCell(name="ethereum.lstm", horizons=loaded)


def test_roster_is_exactly_nine_cells_and_thirty_six_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = benchmark._approved_cells()
    k_manifest = {
        cell: UUID(f"10000000-0000-4000-8000-{index:012d}") for index, cell in enumerate(approved)
    }
    held_out = {
        cell: UUID(f"20000000-0000-4000-8000-{index:012d}") for index, cell in enumerate(approved)
    }

    def load_manifest(storage_root: Path, kind: object, experiment_id: UUID) -> dict[str, UUID]:
        del storage_root, experiment_id
        return k_manifest if kind == benchmark.ExperimentKind.K_STUDY else held_out

    monkeypatch.setattr(benchmark, "load_experiment_manifest", load_manifest)
    monkeypatch.setattr(
        benchmark,
        "_validate_artifact",
        lambda storage_root, *, cell, artifact_id, evaluation_id: _provenance(
            int(cell.rsplit("K", maxsplit=1)[1]),
            chain=cell.split(".")[0],
            family=cell.split(".")[1],
        ).model_copy(
            update={"artifact_id": artifact_id, "evaluation_id": evaluation_id, "cell": cell}
        ),
    )

    records = benchmark._resolve_campaign(Path("/unused"), _K_STUDY_ID, _HELD_OUT_ID)

    assert len(records) == 36
    assert len({record.cell.rsplit(".", maxsplit=1)[0] for record in records}) == 9
    assert tuple(record.cell for record in records) == approved

    held_out.pop(approved[-1])
    with pytest.raises(ValueError, match="exact approved K2-K5 roster"):
        benchmark._resolve_campaign(Path("/unused"), _K_STUDY_ID, _HELD_OUT_ID)


def test_validate_artifact_requires_canonical_typed_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    horizon = 2
    association = _association(horizon)
    evaluation_id = UUID("20000000-0000-4000-8000-000000000002")
    request = EvaluateRequest(
        evaluation_id=evaluation_id,
        artifact_id=association.request.artifact_id,
        corpus_id=_CORPUS_ID,
        testing_window=BlockWindow(first_parent_block=100, last_parent_block=103),
    )
    evaluation = tmp_path / "evaluations" / str(evaluation_id)
    evaluation.mkdir(parents=True)
    (evaluation / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")
    _observations(evaluation / "observations.parquet", request)
    checkpoint = tmp_path / "artifacts" / f"{association.request.artifact_id}.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    study = Study(
        request=TuneRequest(
            study_id=_STUDY_ID, corpus_id=_CORPUS_ID, experiment=_experiment(5), methods=(_METHOD,)
        ),
        trials=(RetainedResult(objective=1.0, selected_epoch=1, completed_epochs=1),),
    )
    monkeypatch.setattr(benchmark, "load_artifact", lambda *_: (association, nn.Identity()))
    monkeypatch.setattr(
        benchmark,
        "load_corpus_request",
        lambda *_: CorpusRequest(
            corpus_id=_CORPUS_ID,
            definition=CorpusDefinition(chain_id=1, first_block=0, last_block=200),
        ),
    )
    monkeypatch.setattr(benchmark, "load_study", lambda *_: study)

    record = benchmark._validate_artifact(
        tmp_path,
        cell="ethereum.lstm.K2",
        artifact_id=association.request.artifact_id,
        evaluation_id=evaluation_id,
    )

    assert record.artifact_sha256 == benchmark._sha256(checkpoint)
    assert record.study_id == _STUDY_ID
    assert record.method == _METHOD
    assert record.ordered_features == ("log_base_fee_per_gas",)

    wrong_method = association.model_copy(
        update={"method": _METHOD.model_copy(update={"fit": _FIT.model_copy(update={"seed": 2})})}
    )
    monkeypatch.setattr(benchmark, "load_artifact", lambda *_: (wrong_method, nn.Identity()))
    with pytest.raises(ValueError, match="embedded Method"):
        benchmark._validate_artifact(
            tmp_path,
            cell="ethereum.lstm.K2",
            artifact_id=association.request.artifact_id,
            evaluation_id=evaluation_id,
        )

    monkeypatch.setattr(benchmark, "load_artifact", lambda *_: (association, nn.Identity()))
    with pytest.raises(ValueError, match="family"):
        benchmark._validate_artifact(
            tmp_path,
            cell="ethereum.transformer.K2",
            artifact_id=association.request.artifact_id,
            evaluation_id=evaluation_id,
        )

    wrong_horizon = _association(3, association.request.artifact_id)
    monkeypatch.setattr(benchmark, "load_artifact", lambda *_: (wrong_horizon, nn.Identity()))
    with pytest.raises(ValueError, match="horizon"):
        benchmark._validate_artifact(
            tmp_path,
            cell="ethereum.lstm.K2",
            artifact_id=association.request.artifact_id,
            evaluation_id=evaluation_id,
        )

    wrong_corpus_request = request.model_copy(
        update={"corpus_id": UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")}
    )
    (evaluation / "evaluation.json").write_text(
        wrong_corpus_request.model_dump_json(), encoding="utf-8"
    )
    monkeypatch.setattr(benchmark, "load_artifact", lambda *_: (association, nn.Identity()))
    with pytest.raises(ValueError, match="same Corpus"):
        benchmark._validate_artifact(
            tmp_path,
            cell="ethereum.lstm.K2",
            artifact_id=association.request.artifact_id,
            evaluation_id=evaluation_id,
        )


def test_cell_geometry_requires_shared_origin_and_k_offset() -> None:
    records = [_provenance(horizon) for horizon in benchmark._HORIZONS]
    benchmark._validate_cell_geometry(records)

    records[0] = records[0].model_copy(
        update={"testing_window": BlockWindow(first_parent_block=100, last_parent_block=102)}
    )
    with pytest.raises(ValueError, match="canonical K geometry"):
        benchmark._validate_cell_geometry(records)


def test_batch_one_is_a_view_and_preserves_chronological_origin() -> None:
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

    batch = benchmark._batch_one(dataset, 1, 103)

    assert batch.shape == (1, 2, 2)
    assert batch.untyped_storage().data_ptr() == backing.inputs.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="chronological traversal"):
        benchmark._batch_one(dataset, 1, 104)


def test_one_cell_load_keeps_exactly_four_verified_cpu_models_resident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for horizon in benchmark._HORIZONS:
        record = _provenance(horizon)
        checkpoint = tmp_path / "artifacts" / f"{record.artifact_id}.ckpt"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(f"K{horizon}".encode())
        record = record.model_copy(update={"artifact_sha256": benchmark._sha256(checkpoint)})
        records.append(record)
        request = EvaluateRequest(
            evaluation_id=record.evaluation_id,
            artifact_id=record.artifact_id,
            corpus_id=record.corpus_id,
            testing_window=record.testing_window,
        )
        directory = tmp_path / "evaluations" / str(record.evaluation_id)
        directory.mkdir(parents=True)
        (directory / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")

    loads: list[int] = []
    models: dict[int, nn.Module] = {}

    def load(storage_root: Path, artifact_id: UUID) -> tuple[ArtifactAssociation, nn.Module]:
        del storage_root
        horizon = next(
            record.horizon_blocks for record in records if record.artifact_id == artifact_id
        )
        loads.append(horizon)
        model = _TracingModel(horizon, []).eval()
        models[horizon] = model
        return _association(horizon, artifact_id), model

    monkeypatch.setattr(benchmark, "load_corpus_blocks", lambda *_: object())
    monkeypatch.setattr(benchmark, "load_artifact", load)
    monkeypatch.setattr(
        benchmark,
        "prepare_historical_window",
        lambda blocks, experiment, window, **kwargs: cast(
            HistoricalDataset,
            _ViewDataset(tuple(range(window.first_parent_block, window.last_parent_block + 1))),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_strict_observations",
        lambda path, request, horizon: benchmark._Reference(
            origins=tuple(
                range(
                    request.testing_window.first_parent_block,
                    request.testing_window.last_parent_block + 1,
                )
            ),
            actions=(0,) * (request.testing_window.last_parent_block - 99),
            minimum_logs=(0.0,) * (request.testing_window.last_parent_block - 99),
        ),
    )

    loaded = benchmark._load_cell(tmp_path, records)

    assert loads == [2, 3, 4, 5]
    assert set(loaded.horizons) == {2, 3, 4, 5}
    assert {item.model for item in loaded.horizons.values()} == set(models.values())

    records[0] = records[0].model_copy(update={"artifact_sha256": "0" * 64})
    with pytest.raises(ValueError, match="artifact bytes changed"):
        benchmark._load_cell(tmp_path, records)


def test_parity_and_warmup_are_excluded_and_action_mismatch_rejects() -> None:
    events: list[str] = []
    cell = _loaded_cell(events)
    protocol = _protocol()

    benchmark._warm(cell, protocol.warmup_iterations)
    rows = benchmark._parity_rows(cell, protocol, benchmark._protocol_digest(protocol), 1)

    assert len(rows) == 10
    assert len(events) == 8 + 10
    assert {row["effective_origin_block"] for row in rows} == {100, 101, 102, 103}

    rejected = _loaded_cell([], canonical_action=1)
    with pytest.raises(ValueError, match="action parity failed"):
        benchmark._parity_rows(rejected, protocol, benchmark._protocol_digest(protocol), 1)

    log_rejected = _loaded_cell([])
    loaded_k2 = log_rejected.horizons[2]
    cast(dict[int, benchmark._LoadedHorizon], log_rejected.horizons)[2] = benchmark._LoadedHorizon(
        provenance=loaded_k2.provenance,
        association=loaded_k2.association,
        model=loaded_k2.model,
        dataset=loaded_k2.dataset,
        reference=benchmark._Reference(
            origins=loaded_k2.reference.origins,
            actions=loaded_k2.reference.actions,
            minimum_logs=(1.0,) * len(loaded_k2.reference.origins),
        ),
    )
    with pytest.raises(ValueError, match="minimum-log parity failed"):
        benchmark._parity_rows(log_rejected, protocol, benchmark._protocol_digest(protocol), 1)


def test_standalone_and_cascade_use_separate_clocks_and_same_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cell = _loaded_cell(events)
    protocol = _protocol()
    counter = 0

    def clock() -> int:
        nonlocal counter
        events.append("clock")
        counter += 10
        return counter

    original_decode = benchmark.decode_action

    def traced_decode(output: MinBlockFeeOutput) -> torch.Tensor:
        events.append("decode")
        return original_decode(output)

    monkeypatch.setattr(benchmark, "decode_action", traced_decode)
    monkeypatch.setattr(benchmark, "_assert_runtime", lambda protocol: None)

    rows = benchmark._time_cell(
        cell, protocol, benchmark._protocol_digest(protocol), 1, clock=clock
    )

    assert len(rows) == 11
    assert [row["execution_order"] for row in rows] == list(range(11))
    assert {row["elapsed_ns"] for row in rows} == {10}
    cascade = rows[-1]
    assert cascade["workload"] == "cascade"
    assert cascade["horizon_blocks"] == [5, 4, 3, 2]
    assert cascade["origin_block"] == 100
    cascade_events = events[-10:]
    assert cascade_events == [
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


def test_pass_and_cell_order_rotate_without_changing_membership() -> None:
    assert benchmark._pass_order(1) == benchmark._WORKLOADS
    assert benchmark._pass_order(2) == (*benchmark._WORKLOADS[1:], benchmark._WORKLOADS[0])
    assert benchmark._cell_order(2) == (*benchmark._base_cells()[1:], benchmark._base_cells()[0])


def test_runtime_rejects_thread_or_repository_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _protocol()
    monkeypatch.setattr(benchmark, "_repository_commit", lambda: protocol.repository_commit)
    benchmark._assert_runtime(protocol)

    monkeypatch.setattr(torch, "get_num_threads", lambda: protocol.host.intraop_threads + 1)
    with pytest.raises(RuntimeError, match="intra-op thread"):
        benchmark._assert_runtime(protocol)


def test_protocol_is_immutable_and_main_requires_ten_sweeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "_repository_commit", lambda: "f" * 40)
    records = tuple(_provenance(horizon) for horizon in benchmark._HORIZONS)
    output = tmp_path / "output"

    protocol = benchmark._ensure_protocol(
        output,
        purpose="pilot",
        k_study_experiment_id=_K_STUDY_ID,
        held_out_experiment_id=_HELD_OUT_ID,
        artifacts=records,
        warmup_iterations=2,
        prediction_atol=1e-6,
        latency_sweeps=1,
        host=_host(),
    )
    resumed = benchmark._ensure_protocol(
        output,
        purpose="pilot",
        k_study_experiment_id=_K_STUDY_ID,
        held_out_experiment_id=_HELD_OUT_ID,
        artifacts=records,
        warmup_iterations=2,
        prediction_atol=1e-6,
        latency_sweeps=1,
        host=_host(),
    )
    assert resumed == protocol

    with pytest.raises(ValueError, match="does not match"):
        benchmark._ensure_protocol(
            output,
            purpose="pilot",
            k_study_experiment_id=_K_STUDY_ID,
            held_out_experiment_id=_HELD_OUT_ID,
            artifacts=records,
            warmup_iterations=3,
            prediction_atol=1e-6,
            latency_sweeps=1,
            host=_host(),
        )
    with pytest.raises(ValueError, match="at least ten"):
        benchmark._ensure_protocol(
            tmp_path / "main",
            purpose="main",
            k_study_experiment_id=_K_STUDY_ID,
            held_out_experiment_id=_HELD_OUT_ID,
            artifacts=records,
            warmup_iterations=2,
            prediction_atol=1e-6,
            latency_sweeps=9,
            host=_host(),
        )


def test_atomic_publication_never_overwrites_and_cleans_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "unit"
    benchmark._publish_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        benchmark._publish_bytes(target, b"second")
    assert target.read_bytes() == b"first"

    interrupted = tmp_path / "interrupted"

    def interrupt(source: object, destination: object) -> None:
        del source, destination
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark.os, "link", interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark._publish_bytes(interrupted, b"partial")
    assert not interrupted.exists()
    assert not list(tmp_path.glob(".interrupted.*.tmp"))


def test_completed_unit_is_validated_and_skipped_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    records = tuple(_provenance(horizon) for horizon in benchmark._HORIZONS)
    cell = _loaded_cell([])
    monkeypatch.setattr(benchmark, "_load_cell", lambda *_: cell)
    monkeypatch.setattr(benchmark, "_assert_runtime", lambda protocol: None)
    tick = iter(range(0, 1_000, 10))

    benchmark._run_unit(
        tmp_path,
        tmp_path / "output",
        protocol,
        records,
        "ethereum.lstm",
        1,
        clock=lambda: next(tick),
    )
    parity_path, latency_path = benchmark._unit_paths(tmp_path / "output", "ethereum.lstm", 1)
    assert parity_path.is_file()
    assert latency_path.is_file()
    monkeypatch.setattr(
        benchmark,
        "_load_cell",
        cast(Callable[..., benchmark._LoadedCell], lambda *_: pytest.fail("must resume")),
    )
    benchmark._run_unit(
        tmp_path,
        tmp_path / "output",
        protocol,
        records,
        "ethereum.lstm",
        1,
        clock=lambda: pytest.fail("must not retime"),
    )

    latency = pl.read_parquet(latency_path).with_columns(
        pl.lit(0, dtype=pl.Int64).alias("elapsed_ns")
    )
    latency.write_parquet(latency_path)
    with pytest.raises(ValueError, match="non-positive latency"):
        benchmark._run_unit(
            tmp_path,
            tmp_path / "output",
            protocol,
            records,
            "ethereum.lstm",
            1,
            clock=lambda: pytest.fail("must not retime"),
        )


def test_host_metadata_requires_ac_and_low_power_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "arm64")
    answers = {
        ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"): "Apple M2 Max",
        ("/usr/bin/pmset", "-g", "batt"): "Now drawing from 'Battery Power'",
    }
    monkeypatch.setattr(benchmark, "_command", lambda *args: answers[args])
    with pytest.raises(RuntimeError, match="AC power"):
        benchmark._host_metadata()

    answers[("/usr/bin/pmset", "-g", "batt")] = "Now drawing from 'AC Power'"
    answers[("/usr/bin/pmset", "-g", "custom")] = (
        "Battery Power:\n lowpowermode 0\nAC Power:\n lowpowermode 1"
    )
    with pytest.raises(RuntimeError, match="Low Power Mode"):
        benchmark._host_metadata()


def test_evaluation_json_is_parsed_strictly(tmp_path: Path) -> None:
    request = EvaluateRequest(
        evaluation_id=UUID("20000000-0000-4000-8000-000000000002"),
        artifact_id=UUID("10000000-0000-4000-8000-000000000002"),
        corpus_id=_CORPUS_ID,
        testing_window=BlockWindow(first_parent_block=100, last_parent_block=100),
    )
    path = tmp_path / "evaluation.json"
    document = json.loads(request.model_dump_json())
    document["unexpected"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        EvaluateRequest.model_validate_json(path.read_bytes(), strict=True)
