"""Author and close the frozen feature-ablation experiment."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from uuid import UUID, uuid4

import typer
from bundle import StorageRoot, bundle_path, close_study_bundle, write_cells

from fable.config import (
    BlockWindow,
    ExperimentSemantics,
    FeatureName,
    FitMethod,
    LstmDefinition,
    Method,
    TransformerDefinition,
    TransformerLstmDefinition,
    TuneRequest,
)
from fable.experiments import (
    ExperimentKind,
    load_experiment_manifest,
)
from fable.study import load_study

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
        BlockWindow(first_parent_block=75_191_113, last_parent_block=79_663_626),
        BlockWindow(first_parent_block=79_663_827, last_parent_block=81_367_328),
    ),
)
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


def _feature_units(chain: str) -> tuple[tuple[str, tuple[FeatureName, ...]], ...]:
    units: list[tuple[str, tuple[FeatureName, ...]]] = [
        ("base_fee", ("log_base_fee_per_gas",)),
        ("gas_utilization", ("gas_utilization",)),
    ]
    if chain == "ethereum":
        units.append(("exact_forming_base_fee", ("log_exact_forming_base_fee_per_gas",)))
    units.extend(
        (
            ("gas_limit", ("log_gas_limit",)),
            ("transaction_count", ("log1p_tx_count",)),
            ("block_interval", ("block_interval_seconds",)),
            ("hour", ("hour_sin", "hour_cos")),
            ("day_of_week", ("dow_sin", "dow_cos")),
            (
                "priority_fee_p50",
                ("log1p_effective_priority_fee_per_gas_p50",),
            ),
            (
                "priority_fee_p90",
                ("log1p_effective_priority_fee_per_gas_p90",),
            ),
        )
    )
    return tuple(units)


def _feature_configurations(
    chain: str,
) -> tuple[tuple[str, tuple[FeatureName, ...]], ...]:
    units = _feature_units(chain)
    full = _flatten_units(units)
    leave_one_out = tuple(
        (
            f"without_{omitted_name}",
            _flatten_units(units, excluding=omitted_name),
        )
        for omitted_name, _ in units
    )
    return (
        ("full", full),
        *leave_one_out,
        ("base_only", ("log_base_fee_per_gas",)),
    )


def _flatten_units(
    units: tuple[tuple[str, tuple[FeatureName, ...]], ...],
    *,
    excluding: str | None = None,
) -> tuple[FeatureName, ...]:
    features: list[FeatureName] = []
    for name, unit in units:
        if name != excluding:
            features.extend(unit)
    return tuple(features)


def prepare(storage_root: StorageRoot) -> None:
    experiment_id = uuid4()
    bundle = bundle_path(storage_root, _KIND, experiment_id)
    requests = bundle / "requests"
    requests.mkdir(parents=True)

    rows: list[tuple[str, Path, int, UUID]] = []
    for chain, corpus_id, training_window, validation_window in _CHAINS:
        for method in _METHODS:
            family = method.model.family
            for configuration, ordered_features in _feature_configurations(chain):
                request = TuneRequest(
                    corpus_id=corpus_id,
                    experiment=ExperimentSemantics(
                        training_window=training_window,
                        validation_window=validation_window,
                        context_blocks=100,
                        horizon_blocks=5,
                        ordered_features=ordered_features,
                    ),
                    methods=(method,),
                )
                path = requests / f"{len(rows):03d}.json"
                path.write_text(request.model_dump_json(), encoding="utf-8")
                rows.append(
                    (
                        f"{chain}.{family}.{configuration}",
                        path,
                        0,
                        request.study_id,
                    )
                )

    write_cells(bundle, ("cell", "request", "method_index", "study_id"), rows)

    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    close_study_bundle(storage_root, _KIND, experiment_id)


def report(storage_root: StorageRoot, experiment_id: UUID) -> None:
    manifest = load_experiment_manifest(storage_root, _KIND, experiment_id)
    objectives: dict[tuple[str, str], list[float]] = {}
    for entry in manifest.entries:
        chain, _, configuration = entry.cell.split(".")
        study = load_study(storage_root, entry.record_id)
        objectives.setdefault((chain, configuration), []).append(study.trials[0].objective)

    for chain, *_ in _CHAINS:
        for configuration, _ in _feature_configurations(chain):
            print(f"{chain}\t{configuration}\t{fmean(objectives[chain, configuration]):g}")


app = typer.Typer(add_completion=False)
app.command()(prepare)
app.command()(close)
app.command()(report)


if __name__ == "__main__":
    app()
