from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import UUID

import pytest

from fable.config import TuneRequest
from fable.experiments import ExperimentManifest
from tests.experiments.helpers import publish_generated_studies
from tests.helpers import read_tsv_rows, run_script

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"


def test_prepare_authors_full_leave_one_out_and_base_only_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path.parent)
    result = run_script(_SCRIPT, "prepare", tmp_path.name)
    experiment_id = UUID(result.stdout.strip())

    bundle = tmp_path / "experiments" / "feature_ablation" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
    requests = [
        TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    ]

    assert experiment_id.version == 4
    assert len(rows) == 102
    assert [row["cell"] for row in rows[:12]] == [
        "ethereum.lstm.full",
        "ethereum.lstm.without_base_fee",
        "ethereum.lstm.without_gas_utilization",
        "ethereum.lstm.without_exact_forming_base_fee",
        "ethereum.lstm.without_gas_limit",
        "ethereum.lstm.without_transaction_count",
        "ethereum.lstm.without_block_interval",
        "ethereum.lstm.without_hour",
        "ethereum.lstm.without_day_of_week",
        "ethereum.lstm.without_priority_fee_p50",
        "ethereum.lstm.without_priority_fee_p90",
        "ethereum.lstm.base_only",
    ]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm.base_only"
    assert len({request.study_id for request in requests}) == 102
    assert {request.study_id.version for request in requests} == {4}
    assert {len(request.methods) for request in requests} == {1}
    assert {row["method_index"] for row in rows} == {"0"}
    assert requests[0].experiment.model_dump() == {
        "training_window": {"first_parent_block": 23_936_094, "last_parent_block": 25_118_158},
        "validation_window": {"first_parent_block": 25_118_359, "last_parent_block": 25_268_763},
        "context_blocks": 100,
        "horizon_blocks": 5,
        "ordered_features": (
            "log_base_fee_per_gas",
            "gas_utilization",
            "log_exact_forming_base_fee_per_gas",
            "log_gas_limit",
            "log1p_tx_count",
            "block_interval_seconds",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "log1p_effective_priority_fee_per_gas_p50",
            "log1p_effective_priority_fee_per_gas_p90",
        ),
    }
    assert requests[7].experiment.ordered_features == (
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_exact_forming_base_fee_per_gas",
        "log_gas_limit",
        "log1p_tx_count",
        "block_interval_seconds",
        "dow_sin",
        "dow_cos",
        "log1p_effective_priority_fee_per_gas_p50",
        "log1p_effective_priority_fee_per_gas_p90",
    )
    assert requests[9].experiment.ordered_features[-1] == (
        "log1p_effective_priority_fee_per_gas_p90"
    )
    assert requests[11].experiment.ordered_features == ("log_base_fee_per_gas",)
    assert requests[-2].experiment.ordered_features == (
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
    )


def test_close_publishes_all_studies_and_report_averages_each_configuration(tmp_path: Path) -> None:
    experiment_id = UUID(run_script(_SCRIPT, "prepare", tmp_path).stdout.strip())
    bundle = tmp_path / "experiments" / "feature_ablation" / f".{experiment_id}"
    objectives = {
        f"{chain}.{family}.{configuration}": objective
        for chain, configuration, objective in (
            ("ethereum", "full", 1.0),
            ("ethereum", "without_hour", 0.75),
            ("polygon", "without_priority_fee_p90", 0.5),
            ("avalanche", "base_only", 0.25),
        )
        for family in ("lstm", "transformer", "transformer_lstm")
    }
    rows = read_tsv_rows(bundle / "cells.tsv")
    jobs = "job_id\tslot\trow\tcell\n42\t0\t0\tethereum.lstm.full\n"
    (bundle / "jobs.tsv").write_text(jobs, encoding="utf-8")
    publish_generated_studies(tmp_path, rows, default_objective=2.0, objectives=objectives)

    result = run_script(_SCRIPT, "close", tmp_path, experiment_id)

    canonical = tmp_path / "experiments" / "feature_ablation" / str(experiment_id)
    manifest_path = canonical / "manifest.json"
    manifest = ExperimentManifest.model_validate_json(manifest_path.read_bytes(), strict=True)
    assert result.stdout.strip() == str(experiment_id)
    assert len(manifest.root) == 102
    assert [str(record_id) for record_id in manifest.root.values()] == [
        row["study_id"] for row in rows
    ]
    assert {path.name for path in canonical.iterdir()} == {"manifest.json"}
    assert not bundle.exists()

    report = run_script(_SCRIPT, "report", tmp_path, experiment_id)
    lines = report.stdout.splitlines()
    assert len(lines) == 34
    assert lines[0] == "ethereum\tfull\t1"
    assert lines[7] == "ethereum\twithout_hour\t0.75"
    assert lines[21] == "polygon\twithout_priority_fee_p90\t0.5"
    assert lines[-1] == "avalanche\tbase_only\t0.25"


def test_close_rejects_existing_canonical_bundle_without_changing_scratch(tmp_path: Path) -> None:
    experiment_id = UUID(run_script(_SCRIPT, "prepare", tmp_path).stdout.strip())
    bundle = tmp_path / "experiments" / "feature_ablation" / f".{experiment_id}"
    publish_generated_studies(tmp_path, read_tsv_rows(bundle / "cells.tsv"), default_objective=1.0)
    canonical = bundle.with_name(str(experiment_id))
    canonical.mkdir()

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_script(_SCRIPT, "close", tmp_path, experiment_id)

    assert "FileExistsError" in error.value.stderr
    assert list(canonical.iterdir()) == []
    assert (bundle / "cells.tsv").is_file()
    assert not (bundle / "manifest.json").exists()
