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
_FEATURE_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"
_C_SCRIPT = _ROOT / "experiments" / "c_study.py"
_FEATURE_EXPERIMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _run(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def _publish_studies(
    storage_root: Path,
    rows: list[dict[str, str]],
    objectives: dict[tuple[str, str], float],
) -> None:
    for row in rows:
        request = TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        parts = row["cell"].split(".")
        study = Study(
            request=request,
            trials=(
                RetainedResult(
                    method=request.methods[0],
                    objective=objectives.get((parts[0], parts[-1]), 2.0),
                    selected_epoch=1,
                    completed_epochs=1,
                ),
            ),
        )
        path = storage_root / "studies" / f"{request.study_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(study.model_dump_json(), encoding="utf-8")


def test_context_study_uses_selected_features_and_reports_chain_winners(
    tmp_path: Path,
) -> None:
    _run(
        _FEATURE_SCRIPT,
        "prepare",
        tmp_path,
        "--experiment-id",
        _FEATURE_EXPERIMENT_ID,
    )
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{_FEATURE_EXPERIMENT_ID}"
    feature_rows = _rows(feature_bundle / "cells.tsv")
    _publish_studies(
        tmp_path,
        feature_rows,
        {
            ("ethereum", "B+S+T+P"): 0.5,
            ("polygon", "B+T+P"): 0.5,
            ("avalanche", "B+S+P"): 0.5,
        },
    )
    _run(_FEATURE_SCRIPT, "select", tmp_path, _FEATURE_EXPERIMENT_ID)

    result = _run(
        _C_SCRIPT,
        "prepare",
        tmp_path,
        _FEATURE_EXPERIMENT_ID,
    )
    experiment_id = UUID(result.stdout.strip())
    bundle = tmp_path / "experiments" / "c_study" / f".{experiment_id}"
    rows = _rows(bundle / "cells.tsv")
    requests = [
        TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    ]

    assert experiment_id.version == 4
    assert len(rows) == 45
    assert [row["cell"] for row in rows[:5]] == [
        "ethereum.lstm.C25",
        "ethereum.lstm.C50",
        "ethereum.lstm.C100",
        "ethereum.lstm.C200",
        "ethereum.lstm.C400",
    ]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm.C400"
    assert [request.experiment.context_blocks for request in requests[:5]] == [
        25,
        50,
        100,
        200,
        400,
    ]
    assert requests[0].experiment.ordered_features[-1] == (
        "log1p_effective_priority_fee_per_gas_p50"
    )
    assert requests[15].experiment.ordered_features == (
        "log_base_fee_per_gas",
        "block_interval_seconds",
        "hour_sin",
        "hour_cos",
        "log1p_effective_priority_fee_per_gas_p50",
    )

    _publish_studies(
        tmp_path,
        rows,
        {
            ("ethereum", "C50"): 0.25,
            ("polygon", "C100"): 0.25,
            ("avalanche", "C200"): 0.25,
        },
    )
    result = _run(_C_SCRIPT, "select", tmp_path, experiment_id)

    assert result.stdout.splitlines() == [
        "ethereum\t50\t0.25",
        "polygon\t100\t0.25",
        "avalanche\t200\t0.25",
    ]
    manifest = ExperimentManifest.model_validate_json(
        (tmp_path / "experiments" / "c_study" / f"{experiment_id}.json").read_bytes(),
        strict=True,
    )
    assert len(manifest.entries) == 45
    assert not bundle.exists()
