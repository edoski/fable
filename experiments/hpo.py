"""Author and close the frozen nine-Study HPO experiment."""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path
from statistics import fmean
from uuid import UUID, uuid4

from bundle import bundle_path, read_cells, write_cells

from fable.config import (
    FitMethod,
    LstmDefinition,
    Method,
    ModelDefinition,
    TransformerDefinition,
    TransformerLstmDefinition,
)
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    load_experiment_manifest,
    write_experiment_manifest,
)
from fable.requests import fresh_tune_request
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
_DROPOUT = (0.1, 0.2, 0.3)
_LEARNING_RATE = (1e-4, 3e-4, 1e-3)
_WEIGHT_DECAY = (0.0, 1e-4, 1e-3)
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
            (256, 1, 128),
            (256, 2, 256),
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
            (192, 4, 3, 384, 192),
            (256, 4, 4, 512, 256),
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
        (192, 4, 3, 384, 192, 1, 192),
        (256, 4, 4, 512, 256, 1, 256),
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
) -> dict[tuple[str, str], Study]:
    manifest = load_experiment_manifest(
        storage_root,
        ExperimentKind.C_STUDY,
        experiment_id,
    )
    studies: dict[tuple[str, str, int], Study] = {}
    objectives: dict[tuple[str, int], list[float]] = {}
    for entry in manifest.entries:
        chain, family, context_label = entry.cell.split(".")
        context = int(context_label.removeprefix("C"))
        study = load_study(storage_root, entry.require_study_id())
        studies[chain, family, context] = study
        objectives.setdefault((chain, context), []).append(study.trials[0].objective)

    selected: dict[tuple[str, str], Study] = {}
    for chain in _CHAINS:
        contexts = {context for candidate_chain, _, context in studies if candidate_chain == chain}
        winner = min(
            contexts,
            key=lambda context: (fmean(objectives[chain, context]), context),
        )
        for family in _FAMILIES:
            selected[chain, family] = studies[chain, family, winner]
    return selected


def prepare(storage_root: Path, c_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    storage_root = storage_root.resolve()
    selected = _selected_context_studies(storage_root, c_experiment_id)
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    requests = bundle / "requests"
    methods_directory = bundle / "methods"
    requests.mkdir(parents=True)
    methods_directory.mkdir()

    methods_by_family = {family: _methods(family) for family in _FAMILIES}
    method_paths: dict[tuple[str, int], Path] = {}
    for family, methods in methods_by_family.items():
        family_directory = methods_directory / family
        family_directory.mkdir()
        for index, method in enumerate(methods):
            path = family_directory / f"{index}.json"
            path.write_text(method.model_dump_json(), encoding="utf-8")
            method_paths[family, index] = path

    rows: list[tuple[str, Path, Path, UUID, int]] = []
    for index, (chain, family) in enumerate(product(_CHAINS, _FAMILIES)):
        source = selected[chain, family]
        request = fresh_tune_request(
            source.request.corpus_id,
            source.request.experiment,
            methods_by_family[family],
        )
        request_path = requests / f"{index}.json"
        request_path.write_text(request.model_dump_json(), encoding="utf-8")
        cell = f"{chain}.{family}"
        rows.extend(
            (
                cell,
                request_path,
                method_paths[family, method_index],
                request.study_id,
                method_index,
            )
            for method_index in range(len(request.methods))
        )

    write_cells(
        bundle,
        ("cell", "request", "method", "study_id", "method_index"),
        rows,
    )

    print(experiment_id)


def select(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    rows = read_cells(bundle)

    entries: list[ExperimentEntry] = []
    selections: list[tuple[str, int, float]] = []
    seen: set[UUID] = set()
    for row in rows:
        study_id = UUID(row["study_id"])
        if study_id in seen:
            continue
        seen.add(study_id)
        study = load_study(storage_root, study_id)
        retained_methods = tuple(result.method for result in study.trials)
        if retained_methods != study.request.methods:
            raise ValueError("HPO Study must retain every frozen Method")
        selected_index, result = min(
            enumerate(study.trials),
            key=lambda item: item[1].objective,
        )
        entries.append(ExperimentEntry(cell=row["cell"], study_id=study_id))
        selections.append((row["cell"], selected_index, result.objective))

    write_experiment_manifest(
        storage_root,
        _KIND,
        ExperimentManifest(experiment_id=experiment_id, entries=tuple(entries)),
    )
    shutil.rmtree(bundle)
    for cell, selected_index, objective in selections:
        print(f"{cell}\t{selected_index}\t{objective:g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("storage_root", type=Path)
    prepare_parser.add_argument("c_experiment_id", type=UUID)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("storage_root", type=Path)
    select_parser.add_argument("experiment_id", type=UUID)
    arguments = parser.parse_args()

    if arguments.command == "prepare":
        prepare(
            arguments.storage_root,
            arguments.c_experiment_id,
        )
    else:
        select(arguments.storage_root, arguments.experiment_id)


if __name__ == "__main__":
    main()
