"""Generic mechanics for temporary experiment cell bundles."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, TypeAlias
from uuid import UUID

import typer

from fable.config import EvaluateRequest, TrainRequest, TuneRequest
from fable.experiments import ExperimentKind, ExperimentManifest, experiment_directory
from fable.study import load_study

StorageRoot: TypeAlias = Annotated[Path, typer.Argument(resolve_path=True)]
BundleRequest: TypeAlias = TuneRequest | TrainRequest | EvaluateRequest


def bundle_path(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    canonical = experiment_directory(storage_root, kind, experiment_id)
    return canonical.with_name(f".{canonical.name}")


def open_bundle(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    bundle = bundle_path(storage_root, kind, experiment_id)
    (bundle / "requests").mkdir(parents=True)
    return bundle


def write_tune_cells(bundle: Path, cells: Iterable[tuple[str, TuneRequest]]) -> None:
    rows: list[tuple[str, Path, int, UUID]] = []
    for index, (cell, request) in enumerate(cells):
        request_path = _write_request(bundle, index, request)
        rows.extend(
            (cell, request_path, method_index, request.study_id)
            for method_index in range(len(request.methods))
        )
    _write_cells(bundle, ("cell", "request", "method_index", "study_id"), rows)


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


def _write_cells(bundle: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_cells(bundle: Path) -> list[dict[str, str]]:
    with (bundle / "cells.tsv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


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


def close_study_bundle(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> None:
    bundle = bundle_path(storage_root, kind, experiment_id)
    rows = read_cells(bundle)

    cells: dict[str, UUID] = {}
    for row in rows:
        study_id = UUID(row["study_id"])
        load_study(storage_root, study_id)
        cells[row["cell"]] = study_id

    publish_bundle(storage_root, kind, experiment_id, cells)
    print(experiment_id)
