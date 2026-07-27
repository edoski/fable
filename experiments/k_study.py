"""Author and close the frozen horizon-sensitivity experiment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from bundle import bundle_path, read_cells, write_cells

from fable.addresses import artifact_checkpoint_path
from fable.config import SelectedStudySource
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.requests import fresh_train_request
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
        study_id = entry.require_study_id()
        study = load_study(storage_root, study_id)
        selected_index, _ = min(
            enumerate(study.trials),
            key=lambda item: item[1].objective,
        )
        for horizon in _HORIZONS:
            request = fresh_train_request(
                SelectedStudySource(
                    kind="selected_study",
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
            artifact_id=artifact_id,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("storage_root", type=Path)
    prepare_parser.add_argument("hpo_experiment_id", type=UUID)
    close_parser = commands.add_parser("close")
    close_parser.add_argument("storage_root", type=Path)
    close_parser.add_argument("experiment_id", type=UUID)
    arguments = parser.parse_args()

    if arguments.command == "prepare":
        prepare(
            arguments.storage_root,
            arguments.hpo_experiment_id,
        )
    else:
        close(arguments.storage_root, arguments.experiment_id)


if __name__ == "__main__":
    main()
