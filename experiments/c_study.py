"""Author and close the frozen context-sensitivity experiment."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID, uuid4

import typer
from bundle import bundle_path, read_cells, write_cells

from fable.config import TuneRequest
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.study import Study, load_study

_KIND = ExperimentKind.C_STUDY
_CONTEXTS = (25, 50, 100, 200, 400)
_CHAINS = ("ethereum", "polygon", "avalanche")
_FAMILIES = ("lstm", "transformer", "transformer_lstm")


def _full_feature_studies(
    storage_root: Path,
    experiment_id: UUID,
) -> dict[tuple[str, str], Study]:
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.FEATURE_ABLATION,
        experiment_id,
    )
    studies: dict[tuple[str, str], Study] = {}
    for entry in manifest.entries:
        chain, family, configuration = entry.cell.split(".")
        if configuration == "full":
            studies[chain, family] = load_study(storage_root, entry.record_id)
    return studies


def prepare(
    storage_root: Path,
    feature_experiment_id: UUID,
) -> None:
    experiment_id = uuid4()
    storage_root = storage_root.resolve()
    selected = _full_feature_studies(storage_root, feature_experiment_id)
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    requests = bundle / "requests"
    requests.mkdir(parents=True)

    rows: list[tuple[str, Path, int, UUID]] = []
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
                request_path = requests / f"{len(rows):02d}.json"
                request_path.write_text(request.model_dump_json(), encoding="utf-8")
                rows.append(
                    (
                        f"{chain}.{family}.C{context}",
                        request_path,
                        0,
                        request.study_id,
                    )
                )

    write_cells(bundle, ("cell", "request", "method_index", "study_id"), rows)

    print(experiment_id)


def close(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    entries: list[ExperimentEntry] = []
    for row in rows:
        study_id = UUID(row["study_id"])
        load_study(storage_root, study_id)
        entries.append(ExperimentEntry(cell=row["cell"], record_id=study_id))

    write_experiment_manifest(
        storage_root,
        _KIND,
        ExperimentManifest(experiment_id=experiment_id, entries=tuple(entries)),
    )
    shutil.rmtree(bundle)
    print(experiment_id)


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(close)


if __name__ == "__main__":
    app()
