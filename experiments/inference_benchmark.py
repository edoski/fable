"""Run the post-evaluation inference measurement experiments."""

from __future__ import annotations

import json
import plistlib
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TypeVar, cast
from uuid import UUID, uuid4

import polars as pl
import torch
import typer
from pydantic import UUID4, Field
from torch import nn

from fable.addresses import evaluation_json_path
from fable.config import EvaluateRequest
from fable.corpus import BlockFrame, load_corpus_blocks
from fable.evaluation import ROLLING_HORIZONS
from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.min_block_fee import MinBlockFeeOutput, decode_action
from fable.modeling import load_artifact
from fable.records import StrictFrozenRecord
from fable.temporal import HistoricalDataset, prepare_historical_window

_T = TypeVar("_T")

_POWERMETRICS = (
    "sudo",
    "-n",
    "/usr/bin/powermetrics",
    "--samplers",
    "cpu_power,gpu_power,ane_power,thermal",
    "--sample-rate",
    "1000",
    "--poweravg",
    "0",
    "--format",
    "plist",
)


class Selection(StrictFrozenRecord):
    artifact_id: UUID4
    evaluation_id: UUID4


class Protocol(StrictFrozenRecord):
    k_study_experiment_id: UUID4
    held_out_experiment_id: UUID4
    rolling_horizons: tuple[int, ...]
    roster: dict[str, Selection]
    warmup_iterations: Annotated[int, Field(ge=1)]
    sweeps: Annotated[int, Field(ge=1)]


@dataclass(frozen=True, slots=True)
class _Horizon:
    model: nn.Module
    dataset: HistoricalDataset


@dataclass(frozen=True, slots=True)
class _Cell:
    name: str
    horizons: Mapping[int, _Horizon]


@dataclass(frozen=True, slots=True)
class _Workload:
    name: str
    horizons: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PowerSample:
    end_second: int
    elapsed_ns: int
    thermal_pressure: str
    cpu_mw: float
    gpu_mw: float
    ane_mw: float
    combined_mw: float


@dataclass(frozen=True, slots=True)
class _Phase:
    pair: int
    phase: str
    start_ns: int
    end_ns: int
    active_seconds: float = 0.0
    completed_cascades: int = 0


def _power_samples(raw: bytes) -> tuple[_PowerSample, ...]:
    samples = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        record = cast(dict[str, object], plistlib.loads(item))
        processor = cast(dict[str, object], record["processor"])
        timestamp = cast(datetime, record["timestamp"]).replace(tzinfo=UTC)
        samples.append(
            _PowerSample(
                end_second=int(timestamp.timestamp()),
                elapsed_ns=cast(int, record["elapsed_ns"]),
                thermal_pressure=cast(str, record["thermal_pressure"]),
                cpu_mw=cast(float, processor["cpu_power"]),
                gpu_mw=cast(float, processor["gpu_power"]),
                ane_mw=cast(float, processor["ane_power"]),
                combined_mw=cast(float, processor["combined_power"]),
            )
        )
    return tuple(samples)


def _contained(sample: _PowerSample, start_ns: int, end_ns: int) -> bool:
    return (
        sample.end_second * 1_000_000_000 - sample.elapsed_ns >= start_ns
        and (sample.end_second + 1) * 1_000_000_000 <= end_ns
    )


def _phase_power(samples: Sequence[_PowerSample], phase: _Phase) -> dict[str, float | int | bool]:
    accepted = tuple(
        sample for sample in samples if _contained(sample, phase.start_ns, phase.end_ns)
    )
    duration_ns = sum(sample.elapsed_ns for sample in accepted)

    def mean(field: str) -> float:
        return (
            sum(cast(float, getattr(sample, field)) * sample.elapsed_ns for sample in accepted)
            / duration_ns
        )

    return {
        "samples": len(accepted),
        "sample_seconds": duration_ns / 1_000_000_000,
        "cpu_mw": mean("cpu_mw"),
        "gpu_mw": mean("gpu_mw"),
        "ane_mw": mean("ane_mw"),
        "combined_mw": mean("combined_mw"),
        "thermal_valid": all(sample.thermal_pressure == "Nominal" for sample in accepted),
    }


def _pair_row(
    samples: Sequence[_PowerSample], idle_phase: _Phase, active_phase: _Phase
) -> dict[str, float | int | bool]:
    idle = _phase_power(samples, idle_phase)
    active = _phase_power(samples, active_phase)
    throughput = active_phase.completed_cascades / active_phase.active_seconds
    row: dict[str, float | int | bool] = {
        "pair": idle_phase.pair,
        "idle_samples": idle["samples"],
        "idle_sample_seconds": idle["sample_seconds"],
        "active_samples": active["samples"],
        "active_sample_seconds": active["sample_seconds"],
    }
    for rail in ("cpu", "gpu", "ane", "combined"):
        row[f"idle_{rail}_mw"] = idle[f"{rail}_mw"]
        row[f"active_{rail}_mw"] = active[f"{rail}_mw"]
    row.update(
        {
            "active_seconds": active_phase.active_seconds,
            "completed_cascades": active_phase.completed_cascades,
            "cascades_per_second": throughput,
            "thermal_valid": bool(idle["thermal_valid"] and active["thermal_valid"]),
            "joules_per_cascade": (
                (cast(float, active["combined_mw"]) - cast(float, idle["combined_mw"]))
                / 1000
                / throughput
            ),
        }
    )
    return row


def _capture_power(path: Path, measure: Callable[[Callable[[], None]], _T]) -> _T:
    with path.open("wb") as output:
        process = subprocess.Popen(_POWERMETRICS, stdout=output, stderr=subprocess.DEVNULL)

        def alive() -> None:
            if process.poll() is not None:
                raise RuntimeError("powermetrics failed")

        try:
            result = measure(alive)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            returncode = process.wait()
    if returncode not in (0, -signal.SIGTERM):
        raise RuntimeError("powermetrics failed")
    return result


def _energy_settings(pairs: int, phase_seconds: int, recovery_seconds: int) -> dict[str, int]:
    return {
        "pairs": pairs,
        "phase_seconds": phase_seconds,
        "recovery_seconds": recovery_seconds,
        "sample_rate_ms": 1000,
    }


def _run_phases(
    cell: _Cell, settings: Mapping[str, int], alive: Callable[[], None]
) -> tuple[_Phase, ...]:
    phases = []
    duration_ns = settings["phase_seconds"] * 1_000_000_000
    for pair in range(1, settings["pairs"] + 1):
        if pair > 1 and settings["recovery_seconds"]:
            alive()
            start_ns = time.time_ns()
            time.sleep(settings["recovery_seconds"])
            phases.append(_Phase(pair, "recovery", start_ns, time.time_ns()))
        alive()
        start_ns = time.time_ns()
        time.sleep(settings["phase_seconds"])
        phases.append(_Phase(pair, "idle", start_ns, time.time_ns()))
        alive()
        phases.append(_active_phase(cell, pair, duration_ns, alive))
        alive()
    return tuple(phases)


def _phase_record(phase: _Phase) -> dict[str, float | int | str]:
    record: dict[str, float | int | str] = {
        "pair": phase.pair,
        "phase": phase.phase,
        "start_ns": phase.start_ns,
        "end_ns": phase.end_ns,
    }
    if phase.phase == "active":
        record["active_seconds"] = phase.active_seconds
        record["completed_cascades"] = phase.completed_cascades
    return record


def _write_energy(path: Path, cell: _Cell, settings: Mapping[str, int]) -> None:
    path.mkdir()
    phases = _capture_power(
        path / "powermetrics.plist", lambda alive: _run_phases(cell, settings, alive)
    )
    samples = _power_samples((path / "powermetrics.plist").read_bytes())
    rows = []
    for pair in range(1, settings["pairs"] + 1):
        idle = next(phase for phase in phases if phase.pair == pair and phase.phase == "idle")
        active = next(phase for phase in phases if phase.pair == pair and phase.phase == "active")
        rows.append(_pair_row(samples, idle, active))
    (path / "phases.json").write_text(
        json.dumps(
            {"settings": dict(settings), "phases": [_phase_record(phase) for phase in phases]},
            sort_keys=True,
        )
    )
    pl.DataFrame(rows).write_parquet(path / "pairs.parquet")


def _resolve(
    storage_root: Path, k_study_experiment_id: UUID, held_out_experiment_id: UUID
) -> dict[str, dict[int, EvaluateRequest]]:
    k_study = load_experiment_manifest(storage_root, ExperimentKind.K_STUDY, k_study_experiment_id)
    held_out = load_experiment_manifest(
        storage_root, ExperimentKind.HELD_OUT, held_out_experiment_id
    )
    suffixes = {f"K{horizon}" for horizon in ROLLING_HORIZONS}
    labels = {
        label
        for label in k_study.keys() | held_out.keys()
        if label.rsplit(".", maxsplit=1)[-1] in suffixes
    }
    groups = {label.rsplit(".", maxsplit=1)[0] for label in labels}
    expected = {f"{group}.K{horizon}" for group in groups for horizon in ROLLING_HORIZONS}
    if (
        len(groups) != 9
        or len(labels) != 36
        or labels != expected
        or not expected <= k_study.keys()
        or not expected <= held_out.keys()
    ):
        raise ValueError("completed manifests must contain exactly nine rolling-horizon groups")

    resolved: dict[str, dict[int, EvaluateRequest]] = {group: {} for group in sorted(groups)}
    for label in sorted(expected):
        evaluation_id = held_out[label]
        request = EvaluateRequest.model_validate_json(
            evaluation_json_path(storage_root, evaluation_id).read_bytes(), strict=True
        )
        if request.artifact_id != k_study[label]:
            raise ValueError(f"{label} evaluation does not name its K-study artifact")
        group, horizon_label = label.rsplit(".", maxsplit=1)
        resolved[group][int(horizon_label.removeprefix("K"))] = request
    return resolved


def _protocol(
    k_study_experiment_id: UUID,
    held_out_experiment_id: UUID,
    resolved: Mapping[str, Mapping[int, EvaluateRequest]],
    warmup_iterations: int,
    sweeps: int,
) -> Protocol:
    return Protocol(
        k_study_experiment_id=k_study_experiment_id,
        held_out_experiment_id=held_out_experiment_id,
        rolling_horizons=ROLLING_HORIZONS,
        roster={
            f"{cell}.K{horizon}": Selection(
                artifact_id=request.artifact_id, evaluation_id=request.evaluation_id
            )
            for cell, group in resolved.items()
            for horizon, request in group.items()
        },
        warmup_iterations=warmup_iterations,
        sweeps=sweeps,
    )


def _publish(path: Path, write: Callable[[Path], object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    try:
        write(temporary)
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink(missing_ok=True)


def _ensure_protocol(output: Path, protocol: Protocol) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        if Protocol.model_validate_json(path.read_bytes(), strict=True) != protocol:
            raise ValueError("existing output protocol does not match this invocation")
        return
    if any(output.iterdir()):
        raise ValueError("output without a protocol cannot be resumed")
    _publish(path, lambda temporary: temporary.write_text(protocol.model_dump_json()))


def _load_cell(storage_root: Path, cell: str, resolved: Mapping[int, EvaluateRequest]) -> _Cell:
    blocks: dict[UUID4, BlockFrame] = {}
    horizons = {}
    for horizon in reversed(ROLLING_HORIZONS):
        request = resolved[horizon]
        association, model = load_artifact(storage_root, request.artifact_id)
        model.eval()
        experiment = association.training_definition.experiment
        if experiment.horizon_blocks != horizon:
            raise ValueError(f"{cell}.K{horizon} does not address its artifact horizon")
        corpus = blocks.get(request.corpus_id)
        if corpus is None:
            corpus = load_corpus_blocks(storage_root, request.corpus_id)
            blocks[request.corpus_id] = corpus
        horizons[horizon] = _Horizon(
            model=model,
            dataset=prepare_historical_window(
                corpus,
                experiment,
                request.testing_window,
                feature_state=association.feature_state,
                target_state=association.target_state,
            ),
        )
    return _Cell(name=cell, horizons=horizons)


def _batch(item: _Horizon, index: int) -> tuple[int, torch.Tensor]:
    sample = item.dataset[index]
    return int(sample["origin_block"]), sample["inputs"].unsqueeze(0)


def _infer(model: nn.Module, inputs: torch.Tensor) -> None:
    output = cast(MinBlockFeeOutput, model(inputs))
    decode_action(output)


def _workload_inputs(
    cell: _Cell, horizons: Sequence[int], index: int
) -> tuple[int, tuple[torch.Tensor, ...]]:
    batches = tuple(_batch(cell.horizons[horizon], index) for horizon in horizons)
    origin = batches[0][0]
    if any(batch_origin != origin for batch_origin, _ in batches):
        raise ValueError("cascade horizons do not contain the required same origin")
    return origin, tuple(inputs for _, inputs in batches)


def _run_workload(cell: _Cell, horizons: Sequence[int], inputs: Sequence[torch.Tensor]) -> None:
    for horizon, batch in zip(horizons, inputs, strict=True):
        _infer(cell.horizons[horizon].model, batch)


def _warm(cell: _Cell, iterations: int) -> None:
    for horizon in reversed(ROLLING_HORIZONS):
        item = cell.horizons[horizon]
        _, inputs = _batch(item, 0)
        for _ in range(iterations):
            _infer(item.model, inputs)


def _rotate(values: Sequence[_T], offset: int) -> tuple[_T, ...]:
    split = offset % len(values)
    return tuple((*values[split:], *values[:split]))


def _pass_order(sweep: int) -> tuple[_Workload, ...]:
    standalone = tuple(
        _Workload(f"k{horizon}", (horizon,)) for horizon in reversed(ROLLING_HORIZONS)
    )
    return _rotate((*standalone, _Workload("cascade", ROLLING_HORIZONS)), sweep - 1)


def _time_cell(cell: _Cell, sweep: int) -> pl.DataFrame:
    rows = []
    for pass_order, workload in enumerate(_pass_order(sweep)):
        source = cell.horizons[workload.horizons[0]]
        if any(
            len(cell.horizons[horizon].dataset) < len(source.dataset)
            for horizon in workload.horizons
        ):
            raise ValueError(f"{workload.name} horizons do not contain all required origins")
        for index in range(len(source.dataset)):
            origin, inputs = _workload_inputs(cell, workload.horizons, index)
            start = time.perf_counter_ns()
            _run_workload(cell, workload.horizons, inputs)
            elapsed = time.perf_counter_ns() - start
            rows.append(
                {
                    "cell": cell.name,
                    "sweep": sweep,
                    "pass_order": pass_order,
                    "workload": workload.name,
                    "origin_block": origin,
                    "elapsed_ns": elapsed,
                }
            )
    return pl.DataFrame(rows)


def _active_phase(cell: _Cell, pair: int, duration_ns: int, alive: Callable[[], None]) -> _Phase:
    origin_count = len(cell.horizons[ROLLING_HORIZONS[0]].dataset)
    start_ns = time.time_ns()
    start = time.perf_counter_ns()
    deadline = start + duration_ns
    completed = 0
    while time.perf_counter_ns() < deadline:
        alive()
        index = (pair - 1 + completed) % origin_count
        _, inputs = _workload_inputs(cell, ROLLING_HORIZONS, index)
        _run_workload(cell, ROLLING_HORIZONS, inputs)
        completed += 1
    elapsed = time.perf_counter_ns() - start
    return _Phase(
        pair=pair,
        phase="active",
        start_ns=start_ns,
        end_ns=time.time_ns(),
        active_seconds=elapsed / 1_000_000_000,
        completed_cascades=completed,
    )


def _run_unit(
    storage_root: Path,
    output: Path,
    protocol: Protocol,
    cell_name: str,
    sweep: int,
    resolved: Mapping[int, EvaluateRequest],
) -> None:
    path = output / "latency" / cell_name / f"sweep-{sweep:03d}.parquet"
    if path.exists():
        return
    cell = _load_cell(storage_root, cell_name, resolved)
    with torch.inference_mode():
        _warm(cell, protocol.warmup_iterations)
        rows = _time_cell(cell, sweep)
    _publish(path, rows.write_parquet)


def _run_energy_unit(
    storage_root: Path,
    output: Path,
    cell_name: str,
    resolved: Mapping[int, EvaluateRequest],
    warmup_iterations: int,
    settings: Mapping[str, int],
) -> None:
    path = output / "energy" / cell_name
    if path.exists():
        required = {"powermetrics.plist", "phases.json", "pairs.parquet"}
        if not required <= {child.name for child in path.iterdir()}:
            raise ValueError(f"incomplete energy cell: {cell_name}")
        existing = json.loads((path / "phases.json").read_text())
        if existing["settings"] != dict(settings):
            raise ValueError(f"existing energy settings do not match: {cell_name}")
        return
    cell = _load_cell(storage_root, cell_name, resolved)
    with torch.inference_mode():
        _warm(cell, warmup_iterations)
        subprocess.run(("/usr/bin/sudo", "-v"), check=True)
        _publish(path, lambda temporary: _write_energy(temporary, cell, settings))


def run_cpu(
    storage_root: Path,
    k_study_experiment_id: UUID,
    held_out_experiment_id: UUID,
    output: Path,
    warmup_iterations: int,
    sweeps: int,
) -> None:
    """Validate, resume, and complete one CPU latency campaign."""

    resolved = _resolve(storage_root, k_study_experiment_id, held_out_experiment_id)
    protocol = _protocol(
        k_study_experiment_id, held_out_experiment_id, resolved, warmup_iterations, sweeps
    )
    _ensure_protocol(output, protocol)
    cells = tuple(resolved)
    for sweep in range(1, sweeps + 1):
        for cell in _rotate(cells, sweep - 1):
            _run_unit(storage_root, output, protocol, cell, sweep, resolved[cell])


def run_energy(
    storage_root: Path, output: Path, pairs: int, phase_seconds: int, recovery_seconds: int
) -> None:
    """Resume and complete one powermetrics energy campaign."""

    protocol = Protocol.model_validate_json((output / "protocol.json").read_bytes(), strict=True)
    resolved = _resolve(
        storage_root, protocol.k_study_experiment_id, protocol.held_out_experiment_id
    )
    _ensure_protocol(
        output,
        _protocol(
            protocol.k_study_experiment_id,
            protocol.held_out_experiment_id,
            resolved,
            protocol.warmup_iterations,
            protocol.sweeps,
        ),
    )
    settings = _energy_settings(pairs, phase_seconds, recovery_seconds)
    for cell, requests in resolved.items():
        _run_energy_unit(storage_root, output, cell, requests, protocol.warmup_iterations, settings)


StorageRoot = Annotated[Path, typer.Argument(resolve_path=True, exists=True, file_okay=False)]
Output = Annotated[Path, typer.Argument(resolve_path=True, file_okay=False)]


def cpu(
    storage_root: StorageRoot,
    k_study_experiment_id: UUID,
    held_out_experiment_id: UUID,
    output: Output,
    warmup_iterations: Annotated[int, typer.Option(min=1)],
    sweeps: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    run_cpu(
        storage_root,
        k_study_experiment_id,
        held_out_experiment_id,
        output,
        warmup_iterations,
        sweeps,
    )


def energy(
    storage_root: StorageRoot,
    output: Output,
    recovery_seconds: Annotated[int, typer.Option(min=0)],
    pairs: Annotated[int, typer.Option(min=1)] = 20,
    phase_seconds: Annotated[int, typer.Option(min=1)] = 60,
) -> None:
    run_energy(storage_root, output, pairs, phase_seconds, recovery_seconds)


app = typer.Typer(add_completion=False)
app.command()(cpu)
app.command()(energy)


if __name__ == "__main__":
    app()
