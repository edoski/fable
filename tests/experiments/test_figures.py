from __future__ import annotations

import math
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import polars as pl
import pytest

from fable.addresses import evaluation_directory
from fable.config import (
    BlockWindow,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TuneRequest,
)
from fable.experiments import ExperimentKind, ExperimentManifest, experiment_directory
from fable.observations import OBSERVATION_SCHEMA
from fable.study import RetainedResult, Study
from tests.experiments.helpers import publish_test_study
from tests.helpers import run_script

_ROOT = Path(__file__).parents[2]
_HPO_SCRIPT = _ROOT / "experiments" / "hpo.py"
_CONTEXTS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200, 400)
_FIGURE_SCRIPTS = {
    ExperimentKind.FEATURE_ABLATION: _ROOT / "experiments" / "figure_feature_ablation.py",
    ExperimentKind.C_STUDY: _ROOT / "experiments" / "figure_context_study.py",
    ExperimentKind.HPO: _ROOT / "experiments" / "figure_hpo.py",
    ExperimentKind.HELD_OUT: _ROOT / "experiments" / "figure_held_out.py",
}
_FIT = FitMethod(
    learning_rate=3e-4,
    weight_decay=1e-4,
    accumulation=1,
    gradient_clip_norm=1.0,
    seed=2026,
    max_epochs=10,
    validate_every_completed_epoch=1,
    patience=2,
    min_delta=0.0,
)
_METHOD = Method(
    model=LstmDefinition(family="lstm", hidden=32, layers=1, head_hidden=16, dropout=0.2), fit=_FIT
)
_EXPERIMENT = ExperimentSemantics(
    training_window=BlockWindow(first_parent_block=100, last_parent_block=199),
    validation_window=BlockWindow(first_parent_block=205, last_parent_block=249),
    context_blocks=100,
    horizon_blocks=5,
    ordered_features=("log_base_fee_per_gas",),
)


def _publish_study(storage_root: Path, objectives: tuple[float, ...]) -> UUID:
    study_id = uuid4()
    request = TuneRequest(
        study_id=study_id,
        corpus_id=uuid4(),
        experiment=_EXPERIMENT,
        methods=tuple(
            _METHOD.model_copy(
                update={"fit": _FIT.model_copy(update={"learning_rate": 3e-4 + index * 1e-5})}
            )
            for index, _ in enumerate(objectives)
        ),
    )
    study = Study(
        request=request,
        trials=tuple(
            RetainedResult(objective=value, selected_epoch=1, completed_epochs=1)
            for value in objectives
        ),
    )
    publish_test_study(storage_root, study)
    return study_id


def _publish_manifest(storage_root: Path, kind: ExperimentKind, cells: dict[str, UUID]) -> UUID:
    experiment_id = uuid4()
    directory = experiment_directory(storage_root, kind, experiment_id)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        ExperimentManifest(root=cells).model_dump_json(), encoding="utf-8"
    )
    return experiment_id


def _assert_pdf(path: Path) -> None:
    assert path.read_bytes().startswith(b"%PDF-")


def test_feature_ablation_figure_uses_canonical_studies_and_is_reproducible(tmp_path: Path) -> None:
    cells = {
        "ethereum.lstm.full": _publish_study(tmp_path, (0.04,)),
        "ethereum.lstm.without_gas_utilization": _publish_study(tmp_path, (0.05,)),
        "ethereum.lstm.base_only": _publish_study(tmp_path, (0.06,)),
    }
    experiment_id = _publish_manifest(tmp_path, ExperimentKind.FEATURE_ABLATION, cells)
    output = tmp_path / "figures"

    result = run_script(
        _FIGURE_SCRIPTS[ExperimentKind.FEATURE_ABLATION],
        tmp_path,
        experiment_id,
        "--output-directory",
        output,
    )
    figure = output / "feature-ablation.pdf"
    first = figure.read_bytes()
    _assert_pdf(figure)
    assert result.stdout.strip() == str(figure)

    run_script(
        _FIGURE_SCRIPTS[ExperimentKind.FEATURE_ABLATION],
        tmp_path,
        experiment_id,
        "--output-directory",
        output,
    )
    assert figure.read_bytes() == first


def test_feature_ablation_figure_rejects_incomplete_family_roster(tmp_path: Path) -> None:
    cells = {
        "ethereum.lstm.full": _publish_study(tmp_path, (0.04,)),
        "ethereum.lstm.without_gas_utilization": _publish_study(tmp_path, (0.05,)),
        "ethereum.transformer.full": _publish_study(tmp_path, (0.04,)),
    }
    experiment_id = _publish_manifest(tmp_path, ExperimentKind.FEATURE_ABLATION, cells)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_script(
            _FIGURE_SCRIPTS[ExperimentKind.FEATURE_ABLATION],
            tmp_path,
            experiment_id,
            "--output-directory",
            tmp_path / "figures",
        )

    assert "ethereum.transformer lacks configurations ['without_gas_utilization']" in (
        error.value.stderr
    )


def test_context_and_hpo_figures_use_canonical_study_objectives(tmp_path: Path) -> None:
    context_id = _publish_manifest(
        tmp_path,
        ExperimentKind.C_STUDY,
        {
            f"ethereum.{family}.C{context}": _publish_study(
                tmp_path, (0.04 if context == 25 else 0.0419 if context == 1 else 0.08,)
            )
            for family in ("lstm", "transformer", "transformer_lstm")
            for context in _CONTEXTS
        },
    )
    hpo_id = _publish_manifest(
        tmp_path,
        ExperimentKind.HPO,
        {"ethereum.lstm": _publish_study(tmp_path, (0.04, 0.035, 0.045))},
    )
    output = tmp_path / "figures"

    context = run_script(
        _FIGURE_SCRIPTS[ExperimentKind.C_STUDY], tmp_path, context_id, "--output-directory", output
    )
    hpo = run_script(
        _FIGURE_SCRIPTS[ExperimentKind.HPO], tmp_path, hpo_id, "--output-directory", output
    )

    context_figure = output / "context-study.pdf"
    hpo_figure = output / "hpo.pdf"
    _assert_pdf(context_figure)
    _assert_pdf(hpo_figure)
    assert context.stdout.strip() == str(context_figure)
    assert hpo.stdout.strip() == str(hpo_figure)

    selection = run_script(_HPO_SCRIPT, "prepare", tmp_path, context_id, "--chain", "ethereum")
    assert selection.stderr.splitlines() == [
        "chain\tselected_context\tselected_mean\tbest_context\tbest_mean\tthreshold",
        "ethereum\t1\t0.0419\t25\t0.04\t0.042",
    ]


def _publish_evaluation(
    storage_root: Path, evaluation_id: UUID, horizon: int, actions: list[int]
) -> None:
    base_fees = {101: 100, 102: 80, 103: 60, 104: 40, 105: 20}
    priority_fees = {block: fee // 10 for block, fee in base_fees.items()}
    rows = []
    for origin, action in zip(range(100, 100 + len(actions)), actions, strict=True):
        outcome_blocks = range(origin + 1, origin + horizon + 1)
        outcome_fees = [base_fees[block] for block in outcome_blocks]
        minimum_action = outcome_fees.index(min(outcome_fees))
        selected_block = origin + 1 + action
        deadline_block = origin + horizon
        rows.append(
            {
                "origin_block": origin,
                "predicted_action_k": action,
                "predicted_minimum_log_base_fee": math.log(outcome_fees[minimum_action]),
                "minimum_action_k": minimum_action,
                "immediate_base_fee_per_gas": base_fees[origin + 1],
                "immediate_effective_priority_fee_per_gas_p50": priority_fees[origin + 1],
                "selected_base_fee_per_gas": base_fees[selected_block],
                "selected_effective_priority_fee_per_gas_p50": priority_fees[selected_block],
                "deadline_base_fee_per_gas": base_fees[deadline_block],
                "deadline_effective_priority_fee_per_gas_p50": priority_fees[deadline_block],
                "minimum_base_fee_per_gas": outcome_fees[minimum_action],
            }
        )
    directory = evaluation_directory(storage_root, evaluation_id)
    directory.mkdir(parents=True)
    pl.DataFrame(rows, schema=OBSERVATION_SCHEMA).write_parquet(directory / "observations.parquet")


def test_held_out_figure_uses_canonical_reducers_for_horizon_and_rolling_plots(
    tmp_path: Path,
) -> None:
    actions = {5: [4], 4: [0, 3], 3: [0, 0, 2], 2: [0, 0, 0, 1]}
    cells = {}
    for horizon, horizon_actions in actions.items():
        evaluation_id = uuid4()
        _publish_evaluation(tmp_path, evaluation_id, horizon, horizon_actions)
        cells[f"ethereum.lstm.K{horizon}"] = evaluation_id
    experiment_id = _publish_manifest(tmp_path, ExperimentKind.HELD_OUT, cells)
    output = tmp_path / "figures"

    result = run_script(
        _FIGURE_SCRIPTS[ExperimentKind.HELD_OUT],
        tmp_path,
        experiment_id,
        "--output-directory",
        output,
    )

    horizon = output / "horizon-study.pdf"
    rolling = output / "rolling-comparison.pdf"
    _assert_pdf(horizon)
    _assert_pdf(rolling)
    assert result.stdout.splitlines() == [str(horizon), str(rolling)]
