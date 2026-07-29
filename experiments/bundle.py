"""Generic mechanics for temporary experiment cell bundles."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, TypeAlias
from uuid import UUID

import typer

from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    write_experiment_manifest,
)
from fable.study import load_study

StorageRoot: TypeAlias = Annotated[Path, typer.Argument(resolve_path=True)]


def bundle_path(storage_root: Path, kind: ExperimentKind, experiment_id: UUID) -> Path:
    return storage_root / "experiments" / kind / f".{experiment_id}"


def write_cells(
    bundle: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> None:
    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_cells(bundle: Path) -> list[dict[str, str]]:
    with (bundle / "cells.tsv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def close_study_bundle(
    storage_root: Path,
    kind: ExperimentKind,
    experiment_id: UUID,
) -> None:
    bundle = bundle_path(storage_root, kind, experiment_id)
    rows = read_cells(bundle)

    entries: list[ExperimentEntry] = []
    for row in rows:
        study_id = UUID(row["study_id"])
        load_study(storage_root, study_id)
        entries.append(ExperimentEntry(cell=row["cell"], record_id=study_id))

    write_experiment_manifest(
        storage_root,
        kind,
        ExperimentManifest(experiment_id=experiment_id, entries=tuple(entries)),
    )
    shutil.rmtree(bundle)
    print(experiment_id)
