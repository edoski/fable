from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from fable.config import TuneRequest
from fable.experiments import ExperimentManifest
from fable.study import RetainedResult, Study
from tests.helpers import read_tsv_rows, run_script

_ROOT = Path(__file__).parents[2]
_FEATURE_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"
_C_SCRIPT = _ROOT / "experiments" / "c_study.py"


def _publish_studies(
    storage_root: Path,
    rows: list[dict[str, str]],
    objectives: dict[tuple[str, str], float],
    *,
    tie_polygon_feature_counts: bool = False,
) -> None:
    for row in rows:
        request = TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        parts = row["cell"].split(".")
        if tie_polygon_feature_counts and parts[0] == "polygon" and parts[-1] == "B+S+P":
            experiment = request.experiment
            tied_features = (*experiment.ordered_features, "hour_sin", "hour_cos")
            request = request.model_copy(
                update={
                    "experiment": experiment.model_copy(update={"ordered_features": tied_features})
                }
            )
        study = Study(
            request=request,
            trials=(
                RetainedResult(
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "3")
    feature_experiment_id = UUID(
        run_script(
            _FEATURE_SCRIPT,
            "prepare",
            tmp_path,
        ).stdout.strip()
    )
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{feature_experiment_id}"
    feature_rows = read_tsv_rows(feature_bundle / "cells.tsv")
    _publish_studies(
        tmp_path,
        feature_rows,
        {
            ("ethereum", "B+S+T+P"): 0.5,
            ("polygon", "B+T+P"): 0.5,
            ("polygon", "B+S+P"): 0.5,
            ("avalanche", "B+S+P"): 0.5,
        },
        tie_polygon_feature_counts=True,
    )
    run_script(_FEATURE_SCRIPT, "select", tmp_path, feature_experiment_id)

    result = run_script(
        _C_SCRIPT,
        "prepare",
        tmp_path,
        feature_experiment_id,
    )
    experiment_id = UUID(result.stdout.strip())
    bundle = tmp_path / "experiments" / "c_study" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
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
    assert {row["method_index"] for row in rows} == {"0"}
    assert [request.experiment.context_blocks for request in requests[:5]] == [
        25,
        50,
        100,
        200,
        400,
    ]
    assert requests[0].experiment.ordered_features[-1] == (
        "log1p_effective_priority_fee_per_gas_p90"
    )
    assert requests[15].experiment.ordered_features == (
        "log_base_fee_per_gas",
        "block_interval_seconds",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "log1p_effective_priority_fee_per_gas_p50",
        "log1p_effective_priority_fee_per_gas_p90",
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
    result = run_script(_C_SCRIPT, "select", tmp_path, experiment_id)

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
