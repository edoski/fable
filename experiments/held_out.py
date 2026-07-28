"""Author and reduce the frozen held-out evaluations."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID, uuid4

import polars as pl
import typer
from bundle import bundle_path, read_cells, write_cells

from fable.addresses import evaluation_directory
from fable.config import BlockWindow
from fable.corpus import load_corpus_request
from fable.evaluation import reduce_evaluation, reduce_rolling
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.requests import fresh_evaluate_request
from fable.study import load_study

_MAX_HORIZON = 200
_KIND = ExperimentKind.HELD_OUT


def prepare(
    storage_root: Path,
    hpo_experiment_id: UUID,
    k_experiment_id: UUID,
) -> None:
    experiment_id = uuid4()
    storage_root = storage_root.resolve()
    hpo = load_experiment_manifest(storage_root, ExperimentKind.HPO, hpo_experiment_id)
    k_study = load_experiment_manifest(storage_root, ExperimentKind.K_STUDY, k_experiment_id)
    studies = {
        entry.cell: load_study(storage_root, entry.require_study_id()) for entry in hpo.entries
    }
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    requests = bundle / "requests"
    requests.mkdir(parents=True)

    rows: list[tuple[str, Path, UUID]] = []
    for index, entry in enumerate(k_study.entries):
        artifact_id = entry.require_artifact_id()
        chain, family, horizon_label = entry.cell.split(".")
        horizon = int(horizon_label.removeprefix("K"))
        study = studies[f"{chain}.{family}"]
        validation_end = study.request.experiment.validation_window.last_parent_block
        corpus_request = load_corpus_request(storage_root, study.request.corpus_id)
        first_parent = validation_end + _MAX_HORIZON + 1
        last_parent = corpus_request.definition.last_block - _MAX_HORIZON + max(0, 5 - horizon)
        request = fresh_evaluate_request(
            artifact_id,
            study.request.corpus_id,
            BlockWindow(
                first_parent_block=first_parent,
                last_parent_block=last_parent,
            ),
        )
        request_path = requests / f"{index:02d}.json"
        request_path.write_text(request.model_dump_json(), encoding="utf-8")
        rows.append((entry.cell, request_path, request.evaluation_id))

    write_cells(bundle, ("cell", "request", "evaluation_id"), rows)

    print(experiment_id)


def close(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    entries = tuple(
        ExperimentEntry(cell=row["cell"], evaluation_id=evaluation_id)
        for row in rows
        if evaluation_directory(
            storage_root,
            evaluation_id := UUID(row["evaluation_id"]),
        ).is_dir()
    )
    if len(entries) != len(rows):
        raise FileNotFoundError("every held-out evaluation must exist before closure")
    write_experiment_manifest(
        storage_root,
        _KIND,
        ExperimentManifest(experiment_id=experiment_id, entries=entries),
    )
    shutil.rmtree(bundle)
    print(experiment_id)


def report(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    manifest = load_experiment_manifest(storage_root, _KIND, experiment_id)
    results = [
        pl.DataFrame({"cell": [entry.cell]}).hstack(
            reduce_evaluation(storage_root, entry.require_evaluation_id())
        )
        for entry in manifest.entries
    ]
    print(pl.concat(results).write_csv(None, separator="\t"), end="")


def rolling(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    manifest = load_experiment_manifest(storage_root, _KIND, experiment_id)
    roster: dict[str, dict[int, UUID]] = {}
    for entry in manifest.entries:
        cell, horizon_label = entry.cell.rsplit(".", maxsplit=1)
        horizon = int(horizon_label.removeprefix("K"))
        if horizon in (2, 3, 4, 5):
            roster.setdefault(cell, {})[horizon] = entry.require_evaluation_id()
    print(reduce_rolling(storage_root, roster).write_csv(None, separator="\t"), end="")


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(close)
app.command()(report)
app.command()(rolling)


if __name__ == "__main__":
    app()
