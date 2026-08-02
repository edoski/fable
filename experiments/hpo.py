"""Author and close the frozen nine-Study HPO experiment."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from statistics import fmean
from typing import Annotated
from uuid import UUID, uuid4

import typer
from bundle import (
    StorageRoot,
    append_tune_cells,
    bundle_path,
    load_roster,
    open_bundle,
    publish_bundle,
    read_cells,
    run,
    write_tune_cells,
)

from fable.config import (
    LstmDefinition,
    Method,
    ModelDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from fable.experiments import ExperimentKind
from fable.study import Study, load_study

_KIND = ExperimentKind.HPO
_CHAINS = ("ethereum", "polygon", "avalanche")
_FAMILIES = ("lstm", "transformer", "transformer_lstm")
_L9 = (
    (0, 0, 0, 0),
    (0, 1, 1, 1),
    (0, 2, 2, 2),
    (1, 0, 1, 2),
    (1, 1, 2, 0),
    (1, 2, 0, 1),
    (2, 0, 2, 1),
    (2, 1, 0, 2),
    (2, 2, 1, 0),
)
_DROPOUT = (0.2, 0.1, 0.3)
_LEARNING_RATE = (3e-4, 1e-4, 1e-3)
_WEIGHT_DECAY = (1e-4, 0.0, 1e-3)
_LSTM_CAPACITIES = ((256, 1, 128), (384, 2, 256))
_TRANSFORMER_CAPACITIES = ((192, 4, 3, 384, 192), (384, 8, 4, 768, 256))


def _model(control: ModelDefinition, capacity: int, dropout: float) -> ModelDefinition:
    if capacity == 0:
        return control.model_copy(update={"dropout": dropout})

    if isinstance(control, LstmDefinition):
        hidden, layers, head_hidden = _LSTM_CAPACITIES[capacity - 1]
        return control.model_copy(
            update={
                "hidden": hidden,
                "layers": layers,
                "head_hidden": head_hidden,
                "dropout": dropout,
            }
        )

    model_width, attention_heads, transformer_layers, feedforward_width, head_hidden = (
        _TRANSFORMER_CAPACITIES[capacity - 1]
    )
    update = {
        "model_width": model_width,
        "attention_heads": attention_heads,
        "transformer_layers": transformer_layers,
        "feedforward_width": feedforward_width,
        "head_hidden": head_hidden,
        "dropout": dropout,
    }
    if isinstance(control, TransformerLstmDefinition):
        update.update({"lstm_hidden": model_width, "lstm_layers": 1})
    return control.model_copy(update=update)


def _methods(control: Method) -> tuple[Method, ...]:
    return tuple(
        Method(
            model=_model(control.model, capacity, _DROPOUT[dropout]),
            fit=control.fit.model_copy(
                update={
                    "learning_rate": _LEARNING_RATE[learning_rate],
                    "weight_decay": _WEIGHT_DECAY[weight_decay],
                }
            ),
        )
        for capacity, dropout, learning_rate, weight_decay in _L9
    )


def _selected_context_studies(
    storage_root: Path, experiment_id: UUID, chains: tuple[str, ...]
) -> tuple[dict[tuple[str, str], Study], tuple[tuple[str, int, float], ...]]:
    roster = load_roster(storage_root, ExperimentKind.C_STUDY, experiment_id, "study_id")
    studies: dict[tuple[str, str, int], Study] = {}
    objectives: dict[tuple[str, int], list[float]] = {}
    for cell, study_id in roster.items():
        chain, family, context_label = cell.split(".")
        if chain not in chains:
            continue
        context = int(context_label.removeprefix("C"))
        study = load_study(storage_root, study_id)
        studies[chain, family, context] = study
        objectives.setdefault((chain, context), []).append(study.trials[0].objective)

    selected: dict[tuple[str, str], Study] = {}
    winners: list[tuple[str, int, float]] = []
    for chain in chains:
        contexts = {context for candidate_chain, _, context in studies if candidate_chain == chain}
        winner = min(contexts, key=lambda context: (fmean(objectives[chain, context]), context))
        winners.append((chain, winner, fmean(objectives[chain, winner])))
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]
    return selected, tuple(winners)


def _chains(values: list[str] | None) -> tuple[str, ...]:
    chains = tuple(values) if values else _CHAINS
    if len(set(chains)) != len(chains) or not set(chains) <= set(_CHAINS):
        raise ValueError(f"chains must be unique members of {_CHAINS}")
    return chains


def _cells(
    selected: dict[tuple[str, str], Study], chains: tuple[str, ...]
) -> list[tuple[str, TuneRequest]]:
    cells: list[tuple[str, TuneRequest]] = []
    for chain, family in product(chains, _FAMILIES):
        source = selected[chain, family]
        request = TuneRequest(
            corpus_id=source.request.corpus_id,
            experiment=source.request.experiment,
            methods=_methods(source.request.methods[0]),
        )
        cells.append((f"{chain}.{family}", request))
    return cells


def _report(context_winners: tuple[tuple[str, int, float], ...]) -> None:
    for chain, context, mean in context_winners:
        typer.echo(f"{chain}\t{context}\t{mean:g}", err=True)


def prepare(
    storage_root: StorageRoot,
    c_experiment_id: UUID,
    chain: Annotated[list[str] | None, typer.Option("--chain")] = None,
) -> None:
    experiment_id = uuid4()
    chains = _chains(chain)
    selected, context_winners = _selected_context_studies(storage_root, c_experiment_id, chains)
    bundle = open_bundle(storage_root, _KIND, experiment_id)
    write_tune_cells(bundle, _cells(selected, chains))

    _report(context_winners)
    print(experiment_id)


def extend(
    storage_root: StorageRoot,
    c_experiment_id: UUID,
    experiment_id: UUID,
    chain: Annotated[list[str], typer.Option("--chain")],
) -> None:
    chains = _chains(chain)
    selected, context_winners = _selected_context_studies(storage_root, c_experiment_id, chains)
    append_tune_cells(bundle_path(storage_root, _KIND, experiment_id), _cells(selected, chains))

    _report(context_winners)
    print(experiment_id)


def select(storage_root: StorageRoot, experiment_id: UUID) -> None:
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    expected_cells = {f"{chain}.{family}" for chain, family in product(_CHAINS, _FAMILIES)}
    if {row["cell"] for row in rows} != expected_cells:
        raise ValueError("HPO roster is incomplete")

    cells = {row["cell"]: UUID(row["study_id"]) for row in rows}
    selections = [
        (cell, *load_study(storage_root, study_id).best_result())
        for cell, study_id in cells.items()
    ]

    publish_bundle(storage_root, _KIND, experiment_id, cells)
    for cell, selected_index, result in selections:
        print(f"{cell}\t{selected_index}\t{result.objective:g}")


if __name__ == "__main__":
    run(prepare, extend, select)
