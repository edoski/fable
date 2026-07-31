from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from fable.config import TuneRequest
from fable.experiments import ExperimentManifest
from tests.experiments.helpers import publish_generated_studies
from tests.helpers import read_tsv_rows, run_script

_ROOT = Path(__file__).parents[2]
_FEATURE_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"
_C_SCRIPT = _ROOT / "experiments" / "c_study.py"
_HPO_SCRIPT = _ROOT / "experiments" / "hpo.py"


def test_hpo_pipeline_authors_context_and_search_studies_then_selects_each_winner(
    tmp_path: Path,
) -> None:
    feature_experiment_id = UUID(run_script(_FEATURE_SCRIPT, "prepare", tmp_path).stdout.strip())
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{feature_experiment_id}"
    publish_generated_studies(
        tmp_path, read_tsv_rows(feature_bundle / "cells.tsv"), default_objective=1.0
    )
    run_script(_FEATURE_SCRIPT, "close", tmp_path, feature_experiment_id)

    c_experiment_id = UUID(
        run_script(_C_SCRIPT, "prepare", tmp_path, feature_experiment_id).stdout.strip()
    )
    c_bundle = tmp_path / "experiments" / "c_study" / f".{c_experiment_id}"
    c_rows = read_tsv_rows(c_bundle / "cells.tsv")
    c_requests = [
        TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in c_rows
    ]

    assert c_experiment_id.version == 4
    assert len(c_rows) == 45
    assert [row["cell"] for row in c_rows[:5]] == [
        "ethereum.lstm.C25",
        "ethereum.lstm.C50",
        "ethereum.lstm.C100",
        "ethereum.lstm.C200",
        "ethereum.lstm.C400",
    ]
    assert c_rows[-1]["cell"] == "avalanche.transformer_lstm.C400"
    assert len({request.study_id for request in c_requests}) == 45
    assert {request.study_id.version for request in c_requests} == {4}
    assert {row["method_index"] for row in c_rows} == {"0"}
    assert [request.experiment.context_blocks for request in c_requests[:5]] == [
        25,
        50,
        100,
        200,
        400,
    ]
    assert c_requests[0].experiment.ordered_features[-1] == (
        "log1p_effective_priority_fee_per_gas_p90"
    )
    assert c_requests[15].experiment.ordered_features == (
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_gas_limit",
        "log1p_tx_count",
        "block_interval_seconds",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "log1p_effective_priority_fee_per_gas_p50",
        "log1p_effective_priority_fee_per_gas_p90",
    )
    context_objectives = {
        f"{chain}.{family}.{context}": objective
        for chain, context, objective in (
            ("ethereum", "C50", 0.25),
            ("polygon", "C100", 0.5),
            ("avalanche", "C200", 0.75),
        )
        for family in ("lstm", "transformer", "transformer_lstm")
    }
    publish_generated_studies(
        tmp_path, c_rows, default_objective=1.0, objectives=context_objectives
    )
    c_result = run_script(_C_SCRIPT, "close", tmp_path, c_experiment_id)
    c_canonical = tmp_path / "experiments" / "c_study" / str(c_experiment_id)
    c_manifest = ExperimentManifest.model_validate_json(
        (c_canonical / "manifest.json").read_bytes(), strict=True
    )

    assert c_result.stdout.strip() == str(c_experiment_id)
    assert len(c_manifest.root) == 45
    assert [str(record_id) for record_id in c_manifest.root.values()] == [
        row["study_id"] for row in c_rows
    ]
    assert {path.name for path in c_canonical.iterdir()} == {"manifest.json"}
    assert not c_bundle.exists()

    result = run_script(_HPO_SCRIPT, "prepare", tmp_path, c_experiment_id)
    experiment_id = UUID(result.stdout.strip())
    assert result.stderr.splitlines() == [
        "ethereum\t50\t0.25",
        "polygon\t100\t0.5",
        "avalanche\t200\t0.75",
    ]
    bundle = tmp_path / "experiments" / "hpo" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
    requests = {
        row["cell"]: TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    }

    assert experiment_id.version == 4
    assert len(rows) == 81
    assert len(requests) == 9
    assert [row["cell"] for row in rows[:9]] == ["ethereum.lstm"] * 9
    assert [row["method_index"] for row in rows[:9]] == [str(index) for index in range(9)]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm"
    assert len({request.study_id for request in requests.values()}) == 9
    assert {request.study_id.version for request in requests.values()} == {4}
    assert {len(request.methods) for request in requests.values()} == {9}
    assert {
        chain: {
            request.experiment.context_blocks
            for cell, request in requests.items()
            if cell.startswith(f"{chain}.")
        }
        for chain in ("ethereum", "polygon", "avalanche")
    } == {"ethereum": {50}, "polygon": {100}, "avalanche": {200}}
    assert requests["ethereum.lstm"].methods[0].model.model_dump() == {
        "family": "lstm",
        "hidden": 256,
        "layers": 2,
        "head_hidden": 256,
        "dropout": 0.2,
    }
    assert requests["ethereum.lstm"].methods[0].fit.model_dump() == {
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "accumulation": 1,
        "gradient_clip_norm": 1.0,
        "seed": 2026,
        "max_epochs": 36,
        "validate_every_completed_epoch": 1,
        "patience": 8,
        "min_delta": 0.0,
    }
    assert requests["ethereum.lstm"].methods[-1].fit.model_dump() == {
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "accumulation": 1,
        "gradient_clip_norm": 1.0,
        "seed": 2026,
        "max_epochs": 36,
        "validate_every_completed_epoch": 1,
        "patience": 8,
        "min_delta": 0.0,
    }

    publish_generated_studies(tmp_path, rows, default_objective=0.5)
    result = run_script(_HPO_SCRIPT, "select", tmp_path, experiment_id)

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
    canonical = tmp_path / "experiments" / "hpo" / str(experiment_id)
    manifest = ExperimentManifest.model_validate_json(
        (canonical / "manifest.json").read_bytes(), strict=True
    )
    assert len(manifest.root) == 9
    assert [str(record_id) for record_id in manifest.root.values()] == list(
        dict.fromkeys(row["study_id"] for row in rows)
    )
    assert {path.name for path in canonical.iterdir()} == {"manifest.json"}
    assert not bundle.exists()


@pytest.mark.parametrize(
    ("second_study", "message"),
    (("valid", "one HPO cell cannot reference multiple Studies"), ("missing", "FileNotFoundError")),
)
def test_hpo_select_validates_every_distinct_repeated_cell_reference(
    tmp_path: Path, second_study: str, message: str
) -> None:
    source_id = UUID(run_script(_FEATURE_SCRIPT, "prepare", tmp_path).stdout.strip())
    source = tmp_path / "experiments" / "feature_ablation" / f".{source_id}"
    source_rows = read_tsv_rows(source / "cells.tsv")[:2]
    publish_generated_studies(tmp_path, source_rows, default_objective=1.0)

    experiment_id = uuid4()
    bundle = tmp_path / "experiments" / "hpo" / f".{experiment_id}"
    bundle.mkdir(parents=True)
    conflicting_study_id = (
        source_rows[1]["study_id"]
        if second_study == "valid"
        else "90000000-0000-4000-8000-000000000001"
    )
    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "request", "method_index", "study_id"))
        writer.writerow(("same.cell", source_rows[0]["request"], 0, source_rows[0]["study_id"]))
        writer.writerow(("same.cell", source_rows[1]["request"], 0, conflicting_study_id))

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_script(_HPO_SCRIPT, "select", tmp_path, experiment_id)

    assert message in error.value.stderr
    assert bundle.is_dir()
