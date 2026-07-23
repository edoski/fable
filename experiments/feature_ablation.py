"""Author and close the frozen feature-ablation experiment."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from statistics import fmean
from uuid import UUID, uuid4

from fable.addresses import study_json_path
from fable.config import (
    BlockWindow,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TransformerDefinition,
    TransformerLstmDefinition,
)
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    write_experiment_manifest,
)
from fable.requests import fresh_tune_request
from fable.study import Study

_KIND = ExperimentKind.FEATURE_ABLATION
_CHAINS = (
    (
        "ethereum",
        UUID("e7b17cfb-320d-4e6c-93c1-06dbc24f312d"),
        BlockWindow(first_parent_block=23_936_094, last_parent_block=25_118_158),
        BlockWindow(first_parent_block=25_118_359, last_parent_block=25_268_763),
    ),
    (
        "polygon",
        UUID("2933f8c1-85ce-407b-987a-014128b284e2"),
        BlockWindow(first_parent_block=83_756_900, last_parent_block=86_218_706),
        BlockWindow(first_parent_block=86_218_907, last_parent_block=87_218_399),
    ),
    (
        "avalanche",
        UUID("a06ae6b3-6c3c-445e-8dd8-f5933f9ce0a5"),
        BlockWindow(first_parent_block=72_241_049, last_parent_block=79_663_626),
        BlockWindow(first_parent_block=79_663_827, last_parent_block=81_367_328),
    ),
)
_FEATURE_SETS = ("B", "B+S+T+P", "B+T+P", "B+S+P", "B+S+T")
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
_METHODS = (
    Method(
        model=LstmDefinition(
            family="lstm",
            hidden=256,
            layers=2,
            head_hidden=256,
            dropout=0.2,
        ),
        fit=_FIT,
    ),
    Method(
        model=TransformerDefinition(
            family="transformer",
            model_width=256,
            attention_heads=4,
            transformer_layers=4,
            feedforward_width=512,
            head_hidden=256,
            dropout=0.2,
        ),
        fit=_FIT,
    ),
    Method(
        model=TransformerLstmDefinition(
            family="transformer_lstm",
            model_width=256,
            attention_heads=4,
            transformer_layers=4,
            feedforward_width=512,
            lstm_hidden=256,
            lstm_layers=1,
            head_hidden=256,
            dropout=0.2,
        ),
        fit=_FIT,
    ),
)


def _ordered_features(chain: str, feature_set: str) -> tuple[str, ...]:
    state = (
        (
            "gas_utilization",
            "log_exact_forming_base_fee_per_gas",
            "log_gas_limit",
            "log1p_tx_count",
        )
        if chain == "ethereum"
        else ("gas_utilization", "log_gas_limit", "log1p_tx_count")
    )
    groups = {
        "B": ("log_base_fee_per_gas",),
        "S": state,
        "T": ("block_interval_seconds", "hour_sin", "hour_cos"),
        "P": ("log1p_effective_priority_fee_per_gas_p50",),
    }
    return tuple(feature for group in feature_set.split("+") for feature in groups[group])


def _bundle_path(storage_root: Path, experiment_id: UUID) -> Path:
    return storage_root / "experiments" / _KIND / f".{experiment_id}"


def prepare(storage_root: Path, experiment_id: UUID) -> None:
    if experiment_id.version != 4:
        raise ValueError("experiment_id must be a UUIDv4")

    storage_root = storage_root.resolve()
    bundle = _bundle_path(storage_root, experiment_id)
    requests = bundle / "requests"
    methods = bundle / "methods"
    requests.mkdir(parents=True)
    methods.mkdir()

    method_paths: dict[str, Path] = {}
    for method in _METHODS:
        family = method.model.family
        path = methods / f"{family}.json"
        path.write_text(method.model_dump_json(), encoding="utf-8")
        method_paths[family] = path

    rows: list[tuple[str, Path, Path, UUID]] = []
    index = 0
    for chain, corpus_id, training_window, validation_window in _CHAINS:
        for method in _METHODS:
            family = method.model.family
            for feature_set in _FEATURE_SETS:
                cell = f"{chain}.{family}.{feature_set}"
                request = fresh_tune_request(
                    corpus_id,
                    ExperimentSemantics(
                        training_window=training_window,
                        validation_window=validation_window,
                        context_blocks=100,
                        horizon_blocks=5,
                        ordered_features=_ordered_features(chain, feature_set),
                    ),
                    (method,),
                )
                path = requests / f"{index:02d}.json"
                path.write_text(request.model_dump_json(), encoding="utf-8")
                rows.append((cell, path, method_paths[family], request.study_id))
                index += 1

    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "request", "method", "study_id"))
        writer.writerows(rows)

    print(experiment_id)


def select(storage_root: Path, experiment_id: UUID) -> None:
    storage_root = storage_root.resolve()
    bundle = _bundle_path(storage_root, experiment_id)
    with (bundle / "cells.tsv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))

    objectives: dict[tuple[str, str], list[float]] = {}
    entries: list[ExperimentEntry] = []
    for row in rows:
        chain, _, feature_set = row["cell"].split(".")
        study_id = UUID(row["study_id"])
        study = Study.model_validate_json(
            study_json_path(storage_root, study_id).read_bytes(),
            strict=True,
        )
        if study.request.study_id != study_id or len(study.trials) != 1:
            raise ValueError("feature-ablation Study must contain its one retained result")
        objectives.setdefault((chain, feature_set), []).append(study.trials[0].objective)
        entries.append(ExperimentEntry(cell=row["cell"], study_id=study_id))

    winners: list[tuple[str, str, float]] = []
    for chain, *_ in _CHAINS:
        winner = min(
            _FEATURE_SETS,
            key=lambda feature_set: (
                fmean(objectives[chain, feature_set]),
                len(_ordered_features(chain, feature_set)),
            ),
        )
        mean = fmean(objectives[chain, winner])
        winners.append((chain, winner, mean))

    write_experiment_manifest(
        storage_root,
        _KIND,
        ExperimentManifest(experiment_id=experiment_id, entries=tuple(entries)),
    )
    shutil.rmtree(bundle)
    for chain, feature_set, mean in winners:
        print(f"{chain}\t{feature_set}\t{mean:g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("storage_root", type=Path)
    prepare_parser.add_argument("--experiment-id", type=UUID, default=None)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("storage_root", type=Path)
    select_parser.add_argument("experiment_id", type=UUID)
    arguments = parser.parse_args()

    if arguments.command == "prepare":
        prepare(arguments.storage_root, arguments.experiment_id or uuid4())
    else:
        select(arguments.storage_root, arguments.experiment_id)


if __name__ == "__main__":
    main()
