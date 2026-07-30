"""Author and close the frozen horizon-sensitivity experiment."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import typer
from bundle import (
    StorageRoot,
    bundle_path,
    open_bundle,
    publish_bundle,
    read_cells,
    write_cells,
    write_request,
)

from fable.config import SelectedStudySource, TrainRequest
from fable.experiments import (
    ExperimentKind,
    load_experiment_manifest,
)
from fable.modeling import load_artifact
from fable.study import load_study

_KIND = ExperimentKind.K_STUDY
_HORIZONS = (2, 3, 4, 5, 10, 25, 50, 100, 200)


def prepare(storage_root: StorageRoot, hpo_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.HPO,
        hpo_experiment_id,
    )
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    rows: list[tuple[str, Path, UUID]] = []
    for cell, study_id in manifest.items():
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
            request_path = write_request(bundle, len(rows), request)
            rows.append((f"{cell}.K{horizon}", request_path, request.artifact_id))

    write_cells(bundle, ("cell", "request", "artifact_id"), rows)

    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    cells: dict[str, UUID] = {}
    for row in rows:
        artifact_id = UUID(row["artifact_id"])
        load_artifact(storage_root, artifact_id)
        cells[row["cell"]] = artifact_id

    publish_bundle(storage_root, _KIND, experiment_id, cells)
    print(experiment_id)


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(close)


if __name__ == "__main__":
    app()
