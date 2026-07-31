"""Author and close the frozen horizon-sensitivity experiment."""

from __future__ import annotations

from uuid import UUID, uuid4

from bundle import StorageRoot, close_bundle, open_bundle, run, write_train_cells

from fable.config import SelectedStudySource, TrainRequest
from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.modeling import load_artifact
from fable.study import load_study

_KIND = ExperimentKind.K_STUDY
_HORIZONS = (2, 3, 4, 5, 10, 25, 50, 100, 200)


def prepare(storage_root: StorageRoot, hpo_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    manifest = load_experiment_manifest(storage_root, ExperimentKind.HPO, hpo_experiment_id)
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    cells: list[tuple[str, TrainRequest]] = []
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
            cells.append((f"{cell}.K{horizon}", request))

    write_train_cells(bundle, cells)

    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    close_bundle(storage_root, _KIND, experiment_id, "artifact_id", load_artifact)


if __name__ == "__main__":
    run(prepare, close)
