"""Author and reduce the frozen held-out evaluations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import polars as pl
from bundle import StorageRoot, close_bundle, open_bundle, run, write_evaluate_cells

from fable.config import BlockWindow, EvaluateRequest
from fable.corpus import load_corpus_request
from fable.evaluation import ROLLING_HORIZONS, reduce_baselines, reduce_evaluation, reduce_rolling
from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.study import load_study

_KIND = ExperimentKind.HELD_OUT


def prepare(storage_root: StorageRoot, hpo_experiment_id: UUID, k_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    k_study = load_experiment_manifest(storage_root, ExperimentKind.K_STUDY, k_experiment_id)
    horizons = {cell: int(cell.rsplit(".", maxsplit=1)[1].removeprefix("K")) for cell in k_study}
    maximum_horizon = max(horizons.values())
    hpo = load_experiment_manifest(storage_root, ExperimentKind.HPO, hpo_experiment_id)
    studies = {cell: load_study(storage_root, study_id) for cell, study_id in hpo.items()}
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    cells: list[tuple[str, EvaluateRequest]] = []
    for cell, artifact_id in k_study.items():
        chain, family, _ = cell.split(".")
        horizon = horizons[cell]
        study = studies[f"{chain}.{family}"]
        validation_end = study.request.experiment.validation_window.last_parent_block
        corpus_request = load_corpus_request(storage_root, study.request.corpus_id)
        first_parent = validation_end + maximum_horizon + 1
        last_parent = corpus_request.definition.last_block - maximum_horizon + max(0, 5 - horizon)
        request = EvaluateRequest(
            artifact_id=artifact_id,
            corpus_id=study.request.corpus_id,
            testing_window=BlockWindow(
                first_parent_block=first_parent, last_parent_block=last_parent
            ),
        )
        cells.append((cell, request))

    write_evaluate_cells(bundle, cells)

    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    close_bundle(storage_root, _KIND, experiment_id, "evaluation_id", reduce_evaluation)


def report(storage_root: StorageRoot, experiment_id: UUID) -> None:
    _print_cells(storage_root, experiment_id, reduce_evaluation)


def baselines(storage_root: StorageRoot, experiment_id: UUID) -> None:
    _print_cells(storage_root, experiment_id, reduce_baselines)


def _print_cells(
    storage_root: Path, experiment_id: UUID, reducer: Callable[[Path, UUID], pl.DataFrame]
) -> None:
    manifest = load_experiment_manifest(storage_root, _KIND, experiment_id)
    results = []
    for cell, evaluation_id in manifest.items():
        result = reducer(storage_root, evaluation_id)
        results.append(pl.DataFrame({"cell": [cell] * result.height}).hstack(result))
    print(pl.concat(results).write_csv(None, separator="\t"), end="")


def rolling(storage_root: StorageRoot, experiment_id: UUID) -> None:
    manifest = load_experiment_manifest(storage_root, _KIND, experiment_id)
    roster: dict[str, dict[int, UUID]] = {}
    for experiment_cell, evaluation_id in manifest.items():
        cell, horizon_label = experiment_cell.rsplit(".", maxsplit=1)
        horizon = int(horizon_label.removeprefix("K"))
        if horizon in ROLLING_HORIZONS:
            roster.setdefault(cell, {})[horizon] = evaluation_id
    print(reduce_rolling(storage_root, roster).write_csv(None, separator="\t"), end="")


if __name__ == "__main__":
    run(prepare, close, report, baselines, rolling)
