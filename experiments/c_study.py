"""Author and close the frozen context-sensitivity experiment."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from statistics import fmean
from typing import NamedTuple
from uuid import UUID, uuid4

import typer
from bundle import (
    StorageRoot,
    close_bundle,
    load_roster,
    open_bundle,
    print_study_metrics,
    run,
    write_tune_cells,
)

from kairos.config import TuneRequest
from kairos.experiments import ExperimentKind
from kairos.study import Study, load_study

_KIND = ExperimentKind.C_STUDY
_CONTEXTS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200, 400)
_CHAINS = ("ethereum", "polygon", "avalanche")
_FAMILIES = ("lstm", "transformer", "transformer_lstm")
_CONTEXT_TOLERANCE = 0.05


class _ContextSelection(NamedTuple):
    chain: str
    selected_context: int
    selected_mean: float
    best_context: int
    best_mean: float
    threshold: float


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
                studies[chain, family, configuration].trials[0].objective for family in _FAMILIES
            )
            for configuration in configurations
        }
        winner = min(configurations, key=means.__getitem__)
        winners.append((chain, winner, means[winner]))
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]

    return selected, tuple(winners)


def selected_context_studies(
    storage_root: Path, experiment_id: UUID, chains: tuple[str, ...]
) -> tuple[dict[tuple[str, str], Study], tuple[_ContextSelection, ...]]:
    roster = load_roster(storage_root, _KIND, experiment_id, "study_id")
    expected = {
        f"{chain}.{family}.C{context}"
        for chain, family, context in product(chains, _FAMILIES, _CONTEXTS)
    }
    available = {cell for cell in roster if cell.split(".")[0] in chains}
    if available != expected:
        raise ValueError("context-study roster is incomplete")

    studies = {}
    chain_sources = {}
    family_methods = {}
    for cell, study_id in roster.items():
        chain, family, context_label = cell.split(".")
        if chain in chains:
            context = int(context_label.removeprefix("C"))
            study = load_study(storage_root, study_id)
            if len(study.trials) != 1:
                raise ValueError("context-study cells require one-trial Studies")
            if (
                study.request.methods[0].model.family != family
                or study.request.experiment.context_blocks != context
            ):
                raise ValueError("context-study cell does not match Study request")
            source = (
                study.request.corpus_id,
                study.request.experiment.model_copy(update={"context_blocks": 1}),
            )
            if chain_sources.setdefault(chain, source) != source:
                raise ValueError("context-study chain source identity must match")
            method = study.request.methods[0]
            if family_methods.setdefault((chain, family), method) != method:
                raise ValueError("context-study family Method must match")
            studies[chain, family, context] = study

    selected: dict[tuple[str, str], Study] = {}
    selections = []
    for chain in chains:
        means = {
            context: fmean(
                studies[chain, family, context].trials[0].objective for family in _FAMILIES
            )
            for context in _CONTEXTS
        }
        best = min(_CONTEXTS, key=lambda context: (means[context], context))
        threshold = means[best] * (1.0 + _CONTEXT_TOLERANCE)
        winner = min(context for context in _CONTEXTS if means[context] <= threshold)
        selections.append(
            _ContextSelection(chain, winner, means[winner], best, means[best], threshold)
        )
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]
    return selected, tuple(selections)


def report_context_selections(selections: tuple[_ContextSelection, ...]) -> None:
    typer.echo(
        "chain\tselected_context\tselected_mean\tbest_context\tbest_mean\tthreshold", err=True
    )
    for selection in selections:
        typer.echo(
            "\t".join((selection.chain, *(f"{value:g}" for value in selection[1:]))), err=True
        )


def prepare(storage_root: StorageRoot, feature_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    selected, winners = _selected_feature_studies(storage_root, feature_experiment_id)
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    cells: list[tuple[str, TuneRequest]] = []
    for chain in _CHAINS:
        for family in _FAMILIES:
            source = selected[chain, family]
            if source.request.experiment.context_blocks != 25:
                raise ValueError("feature-ablation Studies must use C=25")
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


def report(storage_root: StorageRoot, experiment_id: UUID) -> None:
    print_study_metrics(storage_root, _KIND, experiment_id)
    _, selections = selected_context_studies(storage_root, experiment_id, _CHAINS)
    report_context_selections(selections)


if __name__ == "__main__":
    run(prepare, close, report)
