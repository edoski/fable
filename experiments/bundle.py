"""Generic mechanics for temporary experiment cell bundles."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from uuid import UUID

from fable.experiments import ExperimentKind


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
