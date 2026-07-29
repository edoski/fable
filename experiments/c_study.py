"""Author and close the frozen context-sensitivity experiment."""

from __future__ import annotations

import shutil
from pathlib import Path
from statistics import fmean
from uuid import UUID, uuid4

import typer
from bundle import bundle_path, read_cells, write_cells

from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.requests import fresh_tune_request
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
            studies[chain, family] = load_study(storage_root, entry.require_study_id())
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
                request = fresh_tune_request(
                    source.request.corpus_id,
                    source.request.experiment.model_copy(update={"context_blocks": context}),
                    (method,),
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


def select(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    objectives: dict[tuple[str, int], list[float]] = {}
    entries: list[ExperimentEntry] = []
    for row in rows:
        chain, _, context_label = row["cell"].split(".")
        context = int(context_label.removeprefix("C"))
        study_id = UUID(row["study_id"])
        study = load_study(storage_root, study_id)
        if len(study.trials) != 1:
            raise ValueError("context Study must contain its one retained result")
        objectives.setdefault((chain, context), []).append(study.trials[0].objective)
        entries.append(ExperimentEntry(cell=row["cell"], study_id=study_id))

    winners: list[tuple[str, int, float]] = []
    for chain in _CHAINS:
        winner = min(
            _CONTEXTS,
            key=lambda context: (fmean(objectives[chain, context]), context),
        )
        winners.append((chain, winner, fmean(objectives[chain, winner])))

    write_experiment_manifest(
        storage_root,
        _KIND,
        ExperimentManifest(experiment_id=experiment_id, entries=tuple(entries)),
    )
    shutil.rmtree(bundle)
    for chain, context, mean in winners:
        print(f"{chain}\t{context}\t{mean:g}")


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(select)


if __name__ == "__main__":
    app()
