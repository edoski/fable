"""Run the post-evaluation CPU inference-latency experiment."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


def _publish(path: Path, write: Callable[[Path], None]) -> None:
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
            batches = tuple(_batch(cell.horizons[horizon], index) for horizon in workload.horizons)
            origin = batches[0][0]
            if any(batch_origin != origin for batch_origin, _ in batches):
                raise ValueError("cascade horizons do not contain the required same origin")
            start = time.perf_counter_ns()
            for horizon, (_, inputs) in zip(workload.horizons, batches, strict=True):
                _infer(cell.horizons[horizon].model, inputs)
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


app = typer.Typer(add_completion=False)
app.command()(cpu)


if __name__ == "__main__":
    app()
