"""Author and close the frozen nine-Study HPO experiment."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from statistics import fmean
from uuid import UUID, uuid4

import typer
from bundle import (
    StorageRoot,
    bundle_path,
    open_bundle,
    publish_bundle,
    read_cells,
    write_tune_cells,
)

from fable.config import (
    FitMethod,
    LstmDefinition,
    Method,
    ModelDefinition,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from fable.experiments import (
    ExperimentKind,
    load_experiment_manifest,
)
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
_FIT = FitMethod(
    learning_rate=3e-4,
    weight_decay=1e-4,
    accumulation=1,
    gradient_clip_norm=1.0,
    seed=2026,
    max_epochs=36,
    validate_every_completed_epoch=1,
    patience=8,
    min_delta=0.0,
)


def _model(family: str, capacity: int, dropout: float) -> ModelDefinition:
    if family == "lstm":
        hidden, layers, head_hidden = (
            (256, 2, 256),
            (256, 1, 128),
            (384, 2, 256),
        )[capacity]
        return LstmDefinition(
            family="lstm",
            hidden=hidden,
            layers=layers,
            head_hidden=head_hidden,
            dropout=dropout,
        )
    if family == "transformer":
        model_width, attention_heads, transformer_layers, feedforward_width, head_hidden = (
            (256, 4, 4, 512, 256),
            (192, 4, 3, 384, 192),
            (384, 8, 4, 768, 256),
        )[capacity]
        return TransformerDefinition(
            family="transformer",
            model_width=model_width,
            attention_heads=attention_heads,
            transformer_layers=transformer_layers,
            feedforward_width=feedforward_width,
            head_hidden=head_hidden,
            dropout=dropout,
        )
    (
        model_width,
        attention_heads,
        transformer_layers,
        feedforward_width,
        lstm_hidden,
        lstm_layers,
        head_hidden,
    ) = (
        (256, 4, 4, 512, 256, 1, 256),
        (192, 4, 3, 384, 192, 1, 192),
        (384, 8, 4, 768, 384, 1, 256),
    )[capacity]
    return TransformerLstmDefinition(
        family="transformer_lstm",
        model_width=model_width,
        attention_heads=attention_heads,
        transformer_layers=transformer_layers,
        feedforward_width=feedforward_width,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        head_hidden=head_hidden,
        dropout=dropout,
    )


def _methods(family: str) -> tuple[Method, ...]:
    return tuple(
        Method(
            model=_model(family, capacity, _DROPOUT[dropout]),
            fit=_FIT.model_copy(
                update={
                    "learning_rate": _LEARNING_RATE[learning_rate],
                    "weight_decay": _WEIGHT_DECAY[weight_decay],
                }
            ),
        )
        for capacity, dropout, learning_rate, weight_decay in _L9
    )


def _selected_context_studies(
    storage_root: Path,
    experiment_id: UUID,
) -> tuple[
    dict[tuple[str, str], Study],
    tuple[tuple[str, int, float], ...],
]:
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.C_STUDY,
        experiment_id,
    )
    studies: dict[tuple[str, str, int], Study] = {}
    objectives: dict[tuple[str, int], list[float]] = {}
    for cell, study_id in manifest.items():
        chain, family, context_label = cell.split(".")
        context = int(context_label.removeprefix("C"))
        study = load_study(storage_root, study_id)
        studies[chain, family, context] = study
        objectives.setdefault((chain, context), []).append(study.trials[0].objective)

    selected: dict[tuple[str, str], Study] = {}
    winners: list[tuple[str, int, float]] = []
    for chain in _CHAINS:
        contexts = {context for candidate_chain, _, context in studies if candidate_chain == chain}
        winner = min(
            contexts,
            key=lambda context: (fmean(objectives[chain, context]), context),
        )
        winners.append((chain, winner, fmean(objectives[chain, winner])))
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]
    return selected, tuple(winners)


def prepare(storage_root: StorageRoot, c_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    selected, context_winners = _selected_context_studies(storage_root, c_experiment_id)
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    methods_by_family = {family: _methods(family) for family in _FAMILIES}

    cells: list[tuple[str, TuneRequest]] = []
    for chain, family in product(_CHAINS, _FAMILIES):
        source = selected[chain, family]
        request = TuneRequest(
            corpus_id=source.request.corpus_id,
            experiment=source.request.experiment,
            methods=methods_by_family[family],
        )
        cells.append((f"{chain}.{family}", request))

    write_tune_cells(bundle, cells)

    for chain, context, mean in context_winners:
        typer.echo(f"{chain}\t{context}\t{mean:g}", err=True)
    print(experiment_id)


def select(storage_root: StorageRoot, experiment_id: UUID) -> None:
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    cells: dict[str, UUID] = {}
    studies: dict[UUID, Study] = {}
    selections: list[tuple[str, int, float]] = []
    for row in rows:
        study_id = UUID(row["study_id"])
        if study_id not in studies:
            studies[study_id] = load_study(storage_root, study_id)
        cell = row["cell"]
        if cell in cells:
            if cells[cell] != study_id:
                raise ValueError("one HPO cell cannot reference multiple Studies")
            continue
        study = studies[study_id]
        selected_index, result = study.best_result()
        cells[cell] = study_id
        selections.append((cell, selected_index, result.objective))

    publish_bundle(storage_root, _KIND, experiment_id, cells)
    for cell, selected_index, objective in selections:
        print(f"{cell}\t{selected_index}\t{objective:g}")


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(select)


if __name__ == "__main__":
    app()
