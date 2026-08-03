"""Author and close the frozen nine-Study HPO experiment."""

from __future__ import annotations

from itertools import product
from typing import Annotated
from uuid import UUID, uuid4

import typer
from bundle import (
    StorageRoot,
    append_tune_cells,
    bundle_path,
    open_bundle,
    print_study_metrics,
    publish_bundle,
    read_cells,
    run,
    write_tune_cells,
)
from c_study import report_context_selections, selected_context_studies

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


def prepare(
    storage_root: StorageRoot,
    c_experiment_id: UUID,
    chain: Annotated[list[str] | None, typer.Option("--chain")] = None,
) -> None:
    experiment_id = uuid4()
    chains = _chains(chain)
    selected, context_winners = selected_context_studies(storage_root, c_experiment_id, chains)
    bundle = open_bundle(storage_root, _KIND, experiment_id)
    write_tune_cells(bundle, _cells(selected, chains))

    report_context_selections(context_winners)
    print(experiment_id)


def extend(
    storage_root: StorageRoot,
    c_experiment_id: UUID,
    experiment_id: UUID,
    chain: Annotated[list[str], typer.Option("--chain")],
) -> None:
    chains = _chains(chain)
    selected, context_winners = selected_context_studies(storage_root, c_experiment_id, chains)
    append_tune_cells(bundle_path(storage_root, _KIND, experiment_id), _cells(selected, chains))

    report_context_selections(context_winners)
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


def report(storage_root: StorageRoot, experiment_id: UUID) -> None:
    print_study_metrics(storage_root, _KIND, experiment_id)


if __name__ == "__main__":
    run(prepare, extend, select, report)
