"""Generic mechanics for temporary experiment cell bundles."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
from uuid import UUID

import polars as pl
import typer

from fable.config import EvaluateRequest, TrainRequest, TuneRequest
from fable.experiments import (
    ExperimentKind,
    ExperimentManifest,
    experiment_directory,
    load_experiment_manifest,
)
from fable.study import load_study, reduce_study_trial

StorageRoot: TypeAlias = Annotated[Path, typer.Argument(resolve_path=True)]
BundleRequest: TypeAlias = TuneRequest | TrainRequest | EvaluateRequest
_RecordColumn: TypeAlias = Literal["study_id", "artifact_id", "evaluation_id"]


def run(*commands: Callable[..., None]) -> None:
    app = typer.Typer(add_completion=False)
    for command in commands:
        app.command()(command)
    app()


def bundle_path(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    canonical = experiment_directory(storage_root, kind, experiment_id)
    return canonical.with_name(f".{canonical.name}")


def open_bundle(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    bundle = bundle_path(storage_root, kind, experiment_id)
    (bundle / "requests").mkdir(parents=True)
    return bundle


def write_tune_cells(bundle: Path, cells: Iterable[tuple[str, TuneRequest]]) -> None:
    _write_tune_cells(bundle, cells, mode="x")


def append_tune_cells(bundle: Path, cells: Iterable[tuple[str, TuneRequest]]) -> None:
    _write_tune_cells(bundle, cells, mode="a")


def _write_tune_cells(
    bundle: Path, cells: Iterable[tuple[str, TuneRequest]], *, mode: Literal["a", "x"]
) -> None:
    cells = tuple(cells)
    existing = read_cells(bundle) if mode == "a" else []
    existing_cells = {row["cell"] for row in existing}
    new_cells = [cell for cell, _ in cells]
    if len(set(new_cells)) != len(new_cells) or existing_cells & set(new_cells):
        raise ValueError("experiment cells must be new and unique")

    request_index = len(dict.fromkeys(row["request"] for row in existing))
    rows: list[tuple[str, Path, int, UUID]] = []
    for index, (cell, request) in enumerate(cells, start=request_index):
        request_path = _write_request(bundle, index, request)
        rows.extend(
            (cell, request_path, method_index, request.study_id)
            for method_index in range(len(request.methods))
        )
    _write_cells(bundle, ("cell", "request", "method_index", "study_id"), rows, mode=mode)


def write_train_cells(bundle: Path, cells: Iterable[tuple[str, TrainRequest]]) -> None:
    rows = (
        (cell, _write_request(bundle, index, request), request.artifact_id)
        for index, (cell, request) in enumerate(cells)
    )
    _write_cells(bundle, ("cell", "request", "artifact_id"), rows)


def write_evaluate_cells(bundle: Path, cells: Iterable[tuple[str, EvaluateRequest]]) -> None:
    rows = (
        (cell, _write_request(bundle, index, request), request.evaluation_id)
        for index, (cell, request) in enumerate(cells)
    )
    _write_cells(bundle, ("cell", "request", "evaluation_id"), rows)


def _write_request(bundle: Path, index: int, request: BundleRequest) -> Path:
    path = bundle / "requests" / f"{index:03d}.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    return path


def _write_cells(
    bundle: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    mode: Literal["a", "x"] = "x",
) -> None:
    with (bundle / "cells.tsv").open(mode, newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        if mode == "x":
            writer.writerow(header)
        writer.writerows(rows)


def read_cells(bundle: Path) -> list[dict[str, str]]:
    with (bundle / "cells.tsv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def load_roster(
    storage_root: Path, kind: ExperimentKind, experiment_id: UUID, column: _RecordColumn
) -> dict[str, UUID]:
    canonical = experiment_directory(storage_root, kind, experiment_id)
    if canonical.exists():
        return load_experiment_manifest(storage_root, kind, experiment_id)
    return {
        row["cell"]: UUID(row[column])
        for row in read_cells(bundle_path(storage_root, kind, experiment_id))
    }


def publish_bundle(
    storage_root: Path, kind: ExperimentKind, experiment_id: UUID, cells: dict[str, UUID]
) -> None:
    manifest = ExperimentManifest(root=cells)
    bundle = bundle_path(storage_root, kind, experiment_id)
    canonical = experiment_directory(storage_root, kind, experiment_id)
    if canonical.exists():
        raise FileExistsError(canonical)

    with (bundle / "manifest.json").open("x", encoding="utf-8") as destination:
        destination.write(manifest.model_dump_json())
    shutil.rmtree(bundle / "requests")
    (bundle / "cells.tsv").unlink()
    (bundle / "jobs.tsv").unlink(missing_ok=True)
    bundle.rename(canonical)


def close_bundle(
    storage_root: Path,
    kind: ExperimentKind,
    experiment_id: UUID,
    column: _RecordColumn,
    verify: Callable[[Path, UUID], object],
) -> None:
    bundle = bundle_path(storage_root, kind, experiment_id)
    rows = read_cells(bundle)

    cells: dict[str, UUID] = {}
    for row in rows:
        record_id = UUID(row[column])
        verify(storage_root, record_id)
        cells[row["cell"]] = record_id

    publish_bundle(storage_root, kind, experiment_id, cells)
    print(experiment_id)


def print_study_metrics(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> None:
    results = []
    for cell, study_id in load_experiment_manifest(storage_root, kind, experiment_id).items():
        study = load_study(storage_root, study_id)
        for method_index in range(len(study.trials)):
            metrics = reduce_study_trial(storage_root, study_id, method_index)
            results.append(
                pl.DataFrame({"cell": [cell], "method_index": [method_index]}).hstack(metrics)
            )
    print(pl.concat(results).write_csv(None, separator="\t"), end="")
