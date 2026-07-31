"""Author and close the frozen context-sensitivity experiment."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import typer
from bundle import StorageRoot, close_study_bundle, open_bundle, write_tune_cells

from fable.config import TuneRequest
from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.study import Study, load_study

_KIND = ExperimentKind.C_STUDY
_CONTEXTS = (25, 50, 100, 200, 400)
_CHAINS = ("ethereum", "polygon", "avalanche")
_FAMILIES = ("lstm", "transformer", "transformer_lstm")


def _full_feature_studies(storage_root: Path, experiment_id: UUID) -> dict[tuple[str, str], Study]:
    manifest = load_experiment_manifest(
        storage_root, ExperimentKind.FEATURE_ABLATION, experiment_id
    )
    studies: dict[tuple[str, str], Study] = {}
    for cell, study_id in manifest.items():
        chain, family, configuration = cell.split(".")
        if configuration == "full":
            studies[chain, family] = load_study(storage_root, study_id)
    return studies


def prepare(storage_root: StorageRoot, feature_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    selected = _full_feature_studies(storage_root, feature_experiment_id)
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    cells: list[tuple[str, TuneRequest]] = []
    for chain in _CHAINS:
        for family in _FAMILIES:
            source = selected[chain, family]
            method = source.request.methods[0]
            for context in _CONTEXTS:
                request = TuneRequest(
                    corpus_id=source.request.corpus_id,
                    experiment=source.request.experiment.model_copy(
                        update={"context_blocks": context}
                    ),
                    methods=(method,),
                )
                cells.append((f"{chain}.{family}.C{context}", request))

    write_tune_cells(bundle, cells)

    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    close_study_bundle(storage_root, _KIND, experiment_id)


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(close)


if __name__ == "__main__":
    app()
