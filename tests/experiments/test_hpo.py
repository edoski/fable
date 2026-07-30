from __future__ import annotations

from pathlib import Path
from uuid import UUID

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
        tmp_path,
        read_tsv_rows(feature_bundle / "cells.tsv"),
        default_objective=1.0,
    )
    run_script(_FEATURE_SCRIPT, "close", tmp_path, feature_experiment_id)

    c_experiment_id = UUID(
        run_script(
            _C_SCRIPT,
            "prepare",
            tmp_path,
            feature_experiment_id,
        ).stdout.strip()
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
        tmp_path,
        c_rows,
        default_objective=1.0,
        objectives=context_objectives,
    )
    c_result = run_script(_C_SCRIPT, "close", tmp_path, c_experiment_id)
    c_manifest = ExperimentManifest.model_validate_json(
        (
            tmp_path / "experiments" / "c_study" / str(c_experiment_id) / "manifest.json"
        ).read_bytes(),
        strict=True,
    )

    assert c_result.stdout.strip() == str(c_experiment_id)
    assert len(c_manifest.root) == 45
    assert [str(record_id) for record_id in c_manifest.root.values()] == [
        row["study_id"] for row in c_rows
    ]
    assert not c_bundle.exists()

    result = run_script(
        _HPO_SCRIPT,
        "prepare",
        tmp_path,
        c_experiment_id,
    )
    experiment_id = UUID(result.stdout.strip())
    assert result.stderr.splitlines() == [
        "ethereum\t50\t0.25",
        "polygon\t100\t0.5",
        "avalanche\t200\t0.75",
    ]
    bundle = tmp_path / "experiments" / "hpo" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
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
    } == {
        "ethereum": {50},
        "polygon": {100},
        "avalanche": {200},
    }
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
    manifest = ExperimentManifest.model_validate_json(
        (tmp_path / "experiments" / "hpo" / str(experiment_id) / "manifest.json").read_bytes(),
        strict=True,
    )
    assert len(manifest.root) == 9
    assert [str(record_id) for record_id in manifest.root.values()] == list(
        dict.fromkeys(row["study_id"] for row in rows)
    )
    assert not bundle.exists()
