from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from uuid import UUID

from fable.config import TuneRequest
from fable.experiments import ExperimentManifest
from fable.study import RetainedResult, Study

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"
_EXPERIMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *(str(argument) for argument in arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


def _rows(storage_root: Path) -> list[dict[str, str]]:
    path = storage_root / "experiments" / "feature_ablation" / f".{_EXPERIMENT_ID}" / "cells.tsv"
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def test_prepare_authors_the_exact_feature_ablation_matrix(tmp_path: Path) -> None:
    result = _run("prepare", tmp_path, "--experiment-id", _EXPERIMENT_ID)

    rows = _rows(tmp_path)
    requests = [
        TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    ]

    assert result.stdout.strip() == str(_EXPERIMENT_ID)
    assert len(rows) == 45
    assert [row["cell"] for row in rows[:5]] == [
        "ethereum.lstm.B",
        "ethereum.lstm.B+S+T+P",
        "ethereum.lstm.B+T+P",
        "ethereum.lstm.B+S+P",
        "ethereum.lstm.B+S+T",
    ]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm.B+S+T"
    assert len({request.study_id for request in requests}) == 45
    assert {len(request.methods) for request in requests} == {1}
    assert requests[0].experiment.model_dump() == {
        "training_window": {
            "first_parent_block": 23_936_094,
            "last_parent_block": 25_118_158,
        },
        "validation_window": {
            "first_parent_block": 25_118_359,
            "last_parent_block": 25_268_763,
        },
        "context_blocks": 100,
        "horizon_blocks": 5,
        "ordered_features": ("log_base_fee_per_gas",),
    }
    assert requests[1].experiment.ordered_features == (
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_exact_forming_base_fee_per_gas",
        "log_gas_limit",
        "log1p_tx_count",
        "block_interval_seconds",
        "hour_sin",
        "hour_cos",
        "log1p_effective_priority_fee_per_gas_p50",
    )
    assert requests[-2].experiment.ordered_features == (
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_gas_limit",
        "log1p_tx_count",
        "log1p_effective_priority_fee_per_gas_p50",
    )


def test_select_publishes_all_studies_and_reports_chain_winners(tmp_path: Path) -> None:
    _run("prepare", tmp_path, "--experiment-id", _EXPERIMENT_ID)
    objectives = {
        "ethereum": {"B": 1.0, "B+S+T+P": 1.0},
        "polygon": {"B+T+P": 0.5},
        "avalanche": {"B+S+P": 0.25},
    }
    for row in _rows(tmp_path):
        chain, _, feature_set = row["cell"].split(".")
        request = TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        objective = objectives.get(chain, {}).get(feature_set, 2.0)
        study = Study(
            request=request,
            trials=(
                RetainedResult(
                    method=request.methods[0],
                    objective=objective,
                    selected_epoch=1,
                    completed_epochs=1,
                ),
            ),
        )
        path = tmp_path / "studies" / f"{request.study_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(study.model_dump_json(), encoding="utf-8")

    result = _run("select", tmp_path, _EXPERIMENT_ID)

    manifest_path = tmp_path / "experiments" / "feature_ablation" / f"{_EXPERIMENT_ID}.json"
    manifest = ExperimentManifest.model_validate_json(
        manifest_path.read_bytes(),
        strict=True,
    )
    assert result.stdout.splitlines() == [
        "ethereum\tB\t1",
        "polygon\tB+T+P\t0.5",
        "avalanche\tB+S+P\t0.25",
    ]
    assert manifest.experiment_id == _EXPERIMENT_ID
    assert len(manifest.entries) == 45
    assert not manifest_path.with_name(f".{_EXPERIMENT_ID}").exists()
