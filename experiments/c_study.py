"""Author and close the frozen context-sensitivity experiment."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from uuid import UUID, uuid4

import typer
from bundle import StorageRoot, close_bundle, load_roster, open_bundle, run, write_tune_cells

from fable.config import TuneRequest
from fable.experiments import ExperimentKind
from fable.study import Study, load_study

_KIND = ExperimentKind.C_STUDY
_CONTEXTS = (25, 50, 100, 200, 400)
_CHAINS = ("ethereum", "polygon", "avalanche")
_FAMILIES = ("lstm", "transformer", "transformer_lstm")


def _selected_feature_studies(
    storage_root: Path, experiment_id: UUID
) -> tuple[dict[tuple[str, str], Study], tuple[tuple[str, str, float], ...]]:
    roster = load_roster(storage_root, ExperimentKind.FEATURE_ABLATION, experiment_id, "study_id")
    studies = {
        tuple(cell.split(".")): load_study(storage_root, study_id)
        for cell, study_id in roster.items()
        if not cell.endswith(".base_only")
    }

    selected: dict[tuple[str, str], Study] = {}
    winners: list[tuple[str, str, float]] = []
    for chain in _CHAINS:
        configurations = tuple(
            configuration
            for candidate_chain, family, configuration in studies
            if candidate_chain == chain and family == _FAMILIES[0]
        )
        means = {
            configuration: fmean(
                studies[chain, family, configuration].trials[0].objective
                for family in _FAMILIES
            )
            for configuration in configurations
        }
        winner = min(configurations, key=means.__getitem__)
        winners.append((chain, winner, means[winner]))
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]

    return selected, tuple(winners)


def prepare(storage_root: StorageRoot, feature_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    selected, winners = _selected_feature_studies(storage_root, feature_experiment_id)
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    cells: list[tuple[str, TuneRequest]] = []
    for chain in _CHAINS:
        for family in _FAMILIES:
            source = selected[chain, family]
            method = source.request.methods[0]
            for context in _CONTEXTS:
                request = (
                    source.request
                    if context == source.request.experiment.context_blocks
                    else TuneRequest(
                        corpus_id=source.request.corpus_id,
                        experiment=source.request.experiment.model_copy(
                            update={"context_blocks": context}
                        ),
                        methods=(method,),
                    )
                )
                cells.append((f"{chain}.{family}.C{context}", request))

    write_tune_cells(bundle, cells)

    for chain, configuration, mean in winners:
        typer.echo(f"{chain}\t{configuration}\t{mean:g}", err=True)
    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    close_bundle(storage_root, _KIND, experiment_id, "study_id", load_study)


if __name__ == "__main__":
    run(prepare, close)
