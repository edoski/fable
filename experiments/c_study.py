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


def _selected_feature_studies(
    storage_root: Path,
    experiment_id: UUID,
) -> dict[tuple[str, str], Study]:
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.FEATURE_ABLATION,
        experiment_id,
    )
    studies: dict[tuple[str, str, str], Study] = {}
    objectives: dict[tuple[str, str], list[float]] = {}
    for entry in manifest.entries:
        chain, family, feature_set = entry.cell.split(".")
        study = load_study(storage_root, entry.require_study_id())
        studies[chain, family, feature_set] = study
        objectives.setdefault((chain, feature_set), []).append(study.trials[0].objective)

    selected: dict[tuple[str, str], Study] = {}
    for chain in _CHAINS:
        feature_sets = {
            feature_set for candidate_chain, _, feature_set in studies if candidate_chain == chain
        }
        winner = min(
            feature_sets,
            key=lambda feature_set: (
                fmean(objectives[chain, feature_set]),
                len(studies[chain, _FAMILIES[0], feature_set].request.experiment.ordered_features),
            ),
        )
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]
    return selected


def prepare(
    storage_root: Path,
    feature_experiment_id: UUID,
) -> None:
    experiment_id = uuid4()
    storage_root = storage_root.resolve()
    selected = _selected_feature_studies(storage_root, feature_experiment_id)
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    requests = bundle / "requests"
    methods = bundle / "methods"
    requests.mkdir(parents=True)
    methods.mkdir()

    rows: list[tuple[str, Path, Path, UUID]] = []
    for chain in _CHAINS:
        for family in _FAMILIES:
            source = selected[chain, family]
            method = source.trials[0].method
            method_path = methods / f"{chain}-{family}.json"
            method_path.write_text(method.model_dump_json(), encoding="utf-8")
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
                        method_path,
                        request.study_id,
                    )
                )

    write_cells(bundle, ("cell", "request", "method", "study_id"), rows)

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
