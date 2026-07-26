"""Author and close the frozen horizon-sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fable.addresses import artifact_checkpoint_path, study_json_path
from fable.config import SelectedStudySource
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.requests import fresh_train_request
from fable.study import Study

_KIND = ExperimentKind.K_STUDY
_HORIZONS = (2, 3, 4, 5, 10, 25, 50, 100, 200)


def _bundle_path(storage_root: Path, experiment_id: UUID) -> Path:
    return storage_root / "experiments" / _KIND / f".{experiment_id}"


def _load_study(storage_root: Path, study_id: UUID) -> Study:
    return Study.model_validate_json(
        study_json_path(storage_root, study_id).read_bytes(),
        strict=True,
    )


def prepare(storage_root: Path, hpo_experiment_id: UUID, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.HPO,
        hpo_experiment_id,
    )
    bundle = _bundle_path(storage_root, experiment_id)
    requests = bundle / "requests"
    requests.mkdir(parents=True)

    rows: list[tuple[str, Path, UUID]] = []
    index = 0
    for entry in manifest.entries:
        if entry.study_id is None:
            raise ValueError("HPO entry must reference a Study")
        study = _load_study(storage_root, entry.study_id)
        selected_index, _ = min(
            enumerate(study.trials),
            key=lambda item: item[1].objective,
        )
        for horizon in _HORIZONS:
            request = fresh_train_request(
                SelectedStudySource(
                    kind="selected_study",
                    corpus_id=study.request.corpus_id,
                    study_id=entry.study_id,
                    study_result_index=selected_index,
                    experiment=study.request.experiment.model_copy(
                        update={"horizon_blocks": horizon}
                    ),
                )
            )
            request_path = requests / f"{index:02d}.json"
            request_path.write_text(request.model_dump_json(), encoding="utf-8")
            rows.append((f"{entry.cell}.K{horizon}", request_path, request.artifact_id))
            index += 1

    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "request", "artifact_id"))
        writer.writerows(rows)

    print(experiment_id)


def close(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = _bundle_path(storage_root, experiment_id)
    with (bundle / "cells.tsv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))

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
    prepare_parser.add_argument("--experiment-id", type=UUID, default=None)
    close_parser = commands.add_parser("close")
    close_parser.add_argument("storage_root", type=Path)
    close_parser.add_argument("experiment_id", type=UUID)
    arguments = parser.parse_args()

    if arguments.command == "prepare":
        prepare(
            arguments.storage_root,
            arguments.hpo_experiment_id,
            arguments.experiment_id or uuid4(),
        )
    else:
        close(arguments.storage_root, arguments.experiment_id)


if __name__ == "__main__":
    main()
