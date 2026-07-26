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
_HPO_SCRIPT = _ROOT / "experiments" / "hpo.py"
_FEATURE_EXPERIMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_C_EXPERIMENT_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


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
    objective: float,
) -> None:
    seen: set[UUID] = set()
    for row in rows:
        request = TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        if request.study_id in seen:
            continue
        seen.add(request.study_id)
        study = Study(
            request=request,
            trials=tuple(
                RetainedResult(
                    method=method,
                    objective=objective + index,
                    selected_epoch=1,
                    completed_epochs=1,
                )
                for index, method in enumerate(request.methods)
            ),
        )
        path = storage_root / "studies" / f"{request.study_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(study.model_dump_json(), encoding="utf-8")


def test_hpo_authors_nine_ordered_l9_studies_and_selects_each_winner(
    tmp_path: Path,
) -> None:
    _run(_FEATURE_SCRIPT, "prepare", tmp_path, "--experiment-id", _FEATURE_EXPERIMENT_ID)
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{_FEATURE_EXPERIMENT_ID}"
    _publish_studies(tmp_path, _rows(feature_bundle / "cells.tsv"), 1.0)
    _run(_FEATURE_SCRIPT, "select", tmp_path, _FEATURE_EXPERIMENT_ID)

    _run(
        _C_SCRIPT,
        "prepare",
        tmp_path,
        _FEATURE_EXPERIMENT_ID,
        "--experiment-id",
        _C_EXPERIMENT_ID,
    )
    c_bundle = tmp_path / "experiments" / "c_study" / f".{_C_EXPERIMENT_ID}"
    _publish_studies(tmp_path, _rows(c_bundle / "cells.tsv"), 1.0)
    _run(_C_SCRIPT, "select", tmp_path, _C_EXPERIMENT_ID)

    result = _run(
        _HPO_SCRIPT,
        "prepare",
        tmp_path,
        _C_EXPERIMENT_ID,
    )
    experiment_id = UUID(result.stdout.strip())
    bundle = tmp_path / "experiments" / "hpo" / f".{experiment_id}"
    rows = _rows(bundle / "candidates.tsv")
    requests = {
        row["cell"]: TuneRequest.model_validate_json(
            Path(row["request"]).read_bytes(),
            strict=True,
        )
        for row in rows
    }

    assert experiment_id.version == 4
    assert len(rows) == 81
    assert len(requests) == 9
    assert [row["cell"] for row in rows[:9]] == ["ethereum.lstm"] * 9
    assert [row["method_index"] for row in rows[:9]] == [str(index) for index in range(9)]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm"
    assert {len(request.methods) for request in requests.values()} == {9}
    assert {request.experiment.context_blocks for request in requests.values()} == {25}
    assert requests["ethereum.lstm"].methods[0].model.model_dump() == {
        "family": "lstm",
        "hidden": 256,
        "layers": 1,
        "head_hidden": 128,
        "dropout": 0.1,
    }
    assert requests["ethereum.lstm"].methods[-1].fit.model_dump() == {
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
        "accumulation": 1,
        "gradient_clip_norm": 1.0,
        "seed": 2026,
        "max_epochs": 36,
        "validate_every_completed_epoch": 1,
        "patience": 8,
        "min_delta": 0.0,
    }

    _publish_studies(tmp_path, rows, 0.5)
    result = _run(_HPO_SCRIPT, "select", tmp_path, experiment_id)

    assert result.stdout.splitlines() == [
        "ethereum.lstm\t0\t0.5",
        "ethereum.transformer\t0\t0.5",
        "ethereum.transformer_lstm\t0\t0.5",
        "polygon.lstm\t0\t0.5",
        "polygon.transformer\t0\t0.5",
        "polygon.transformer_lstm\t0\t0.5",
        "avalanche.lstm\t0\t0.5",
        "avalanche.transformer\t0\t0.5",
        "avalanche.transformer_lstm\t0\t0.5",
    ]
    manifest = ExperimentManifest.model_validate_json(
        (tmp_path / "experiments" / "hpo" / f"{experiment_id}.json").read_bytes(),
        strict=True,
    )
    assert len(manifest.entries) == 9
    assert not bundle.exists()
