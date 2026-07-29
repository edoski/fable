"""Author and close the frozen horizon-sensitivity experiment."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID, uuid4

import typer
from bundle import bundle_path, read_cells, write_cells

from fable.addresses import artifact_checkpoint_path
from fable.config import SelectedStudySource, TrainRequest
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.study import load_study

_KIND = ExperimentKind.K_STUDY
_HORIZONS = (2, 3, 4, 5, 10, 25, 50, 100, 200)


def prepare(storage_root: Path, hpo_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    storage_root = storage_root.resolve()
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.HPO,
        hpo_experiment_id,
    )
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    requests = bundle / "requests"
    requests.mkdir(parents=True)

    rows: list[tuple[str, Path, UUID]] = []
    for entry in manifest.entries:
        study_id = entry.record_id
        study = load_study(storage_root, study_id)
        selected_index, _ = study.best_result()
        for horizon in _HORIZONS:
            request = TrainRequest(
                source=SelectedStudySource(
                    corpus_id=study.request.corpus_id,
                    study_id=study_id,
                    study_result_index=selected_index,
                    experiment=study.request.experiment.model_copy(
                        update={"horizon_blocks": horizon}
                    ),
                )
            )
            request_path = requests / f"{len(rows):02d}.json"
            request_path.write_text(request.model_dump_json(), encoding="utf-8")
            rows.append((f"{entry.cell}.K{horizon}", request_path, request.artifact_id))

    write_cells(bundle, ("cell", "request", "artifact_id"), rows)

    print(experiment_id)


def close(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    entries = tuple(
        ExperimentEntry(
            cell=row["cell"],
            record_id=artifact_id,
        )
        for row in rows
        if artifact_checkpoint_path(
            storage_root,
            artifact_id := UUID(row["artifact_id"]),
        ).is_file()
    )
    if len(entries) != len(rows):
        raise FileNotFoundError("every K-study artifact must exist before closure")

    write_experiment_manifest(
        storage_root,
        _KIND,
        ExperimentManifest(experiment_id=experiment_id, entries=entries),
    )
    shutil.rmtree(bundle)
    print(experiment_id)


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(close)


if __name__ == "__main__":
    app()
