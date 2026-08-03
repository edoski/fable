from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import UUID

import pytest

from fable.addresses import study_json_path
from fable.config import TuneRequest
from fable.experiments import ExperimentManifest
from fable.study import Study
from tests.experiments.helpers import publish_generated_studies
from tests.helpers import read_tsv_rows, run_script

_ROOT = Path(__file__).parents[2]
_FEATURE_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"
_C_SCRIPT = _ROOT / "experiments" / "c_study.py"
_HPO_SCRIPT = _ROOT / "experiments" / "hpo.py"


def test_context_study_selects_chain_mean_winners_and_reuses_reference_studies(
    tmp_path: Path,
) -> None:
    feature_experiment_id = UUID(run_script(_FEATURE_SCRIPT, "prepare", tmp_path).stdout.strip())
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{feature_experiment_id}"
    feature_rows = read_tsv_rows(feature_bundle / "cells.tsv")
    objectives = {
        f"{chain}.{family}.{configuration}": objective
        for chain, configuration, objective in (
            ("ethereum", "without_day_of_week", 0.4),
            ("polygon", "without_block_interval", 0.5),
        )
        for family in ("lstm", "transformer", "transformer_lstm")
    }
    objectives.update(
        {
            "avalanche.lstm.without_base_fee": 1.2,
            "avalanche.transformer.without_base_fee": 0.2,
            "avalanche.transformer_lstm.without_base_fee": 0.2,
        }
    )
    publish_generated_studies(tmp_path, feature_rows, default_objective=1.0, objectives=objectives)

    result = run_script(_C_SCRIPT, "prepare", tmp_path, feature_experiment_id)

    experiment_id = UUID(result.stdout.strip())
    assert result.stderr.splitlines() == [
        "ethereum\twithout_day_of_week\t0.4",
        "polygon\twithout_block_interval\t0.5",
        "avalanche\twithout_base_fee\t0.533333",
    ]
    bundle = tmp_path / "experiments" / "c_study" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
    assert len(rows) == 45

    feature_studies = {row["cell"]: row["study_id"] for row in feature_rows}
    selected_configurations = {
        "ethereum": "without_day_of_week",
        "polygon": "without_block_interval",
        "avalanche": "without_base_fee",
    }
    selected = {
        f"{chain}.{family}": feature_studies[f"{chain}.{family}.{configuration}"]
        for chain, configuration in selected_configurations.items()
        for family in ("lstm", "transformer", "transformer_lstm")
    }
    reference_rows = [row for row in rows if row["cell"].endswith(".C100")]
    assert {row["cell"].removesuffix(".C100"): row["study_id"] for row in reference_rows} == (
        selected
    )

    requests = {
        row["cell"]: TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    }
    assert "dow_sin" not in requests["ethereum.lstm.C25"].experiment.ordered_features
    assert (
        "block_interval_seconds"
        not in requests["polygon.transformer.C25"].experiment.ordered_features
    )
    assert (
        "log_base_fee_per_gas"
        not in requests["avalanche.transformer_lstm.C25"].experiment.ordered_features
    )


def test_context_study_requires_every_feature_candidate_study(tmp_path: Path) -> None:
    feature_experiment_id = UUID(run_script(_FEATURE_SCRIPT, "prepare", tmp_path).stdout.strip())
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{feature_experiment_id}"
    candidate_rows = [
        row
        for row in read_tsv_rows(feature_bundle / "cells.tsv")
        if not row["cell"].endswith(".base_only")
    ]
    publish_generated_studies(tmp_path, candidate_rows[:-1], default_objective=1.0)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_script(_C_SCRIPT, "prepare", tmp_path, feature_experiment_id)

    assert "FileNotFoundError" in error.value.stderr


def test_hpo_pipeline_authors_context_and_search_studies_then_selects_each_winner(
    tmp_path: Path,
) -> None:
    feature_experiment_id = UUID(run_script(_FEATURE_SCRIPT, "prepare", tmp_path).stdout.strip())
    feature_bundle = tmp_path / "experiments" / "feature_ablation" / f".{feature_experiment_id}"
    feature_objectives = {
        f"polygon.{family}.full": 0.5 for family in ("lstm", "transformer", "transformer_lstm")
    }
    publish_generated_studies(
        tmp_path,
        read_tsv_rows(feature_bundle / "cells.tsv"),
        default_objective=1.0,
        objectives=feature_objectives,
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
    ready_rows = [row for row in c_rows if not row["cell"].startswith("avalanche.")]
    publish_generated_studies(
        tmp_path, ready_rows, default_objective=1.0, objectives=context_objectives
    )
    control_row = next(row for row in c_rows if row["cell"] == "ethereum.lstm.C50")
    control_path = study_json_path(tmp_path, UUID(control_row["study_id"]))
    control_study = Study.model_validate_json(control_path.read_bytes(), strict=True)
    control = control_study.request.methods[0]
    control = control.model_copy(
        update={
            "model": control.model.model_copy(update={"hidden": 320, "head_hidden": 320}),
            "fit": control.fit.model_copy(
                update={
                    "accumulation": 2,
                    "gradient_clip_norm": 0.5,
                    "max_epochs": 40,
                    "validate_every_completed_epoch": 2,
                    "patience": 10,
                    "min_delta": 0.01,
                }
            ),
        }
    )
    control_path.write_text(
        Study(
            request=control_study.request.model_copy(update={"methods": (control,)}),
            trials=control_study.trials,
        ).model_dump_json(),
        encoding="utf-8",
    )

    result = run_script(
        _HPO_SCRIPT,
        "prepare",
        tmp_path,
        c_experiment_id,
        "--chain",
        "ethereum",
        "--chain",
        "polygon",
    )
    experiment_id = UUID(result.stdout.strip())
    assert result.stderr.splitlines() == ["ethereum\t50\t0.25", "polygon\t100\t0.5"]
    bundle = tmp_path / "experiments" / "hpo" / f".{experiment_id}"
    assert len(read_tsv_rows(bundle / "cells.tsv")) == 54

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_script(_HPO_SCRIPT, "select", tmp_path, experiment_id)
    assert "HPO roster is incomplete" in error.value.stderr

    avalanche_rows = [row for row in c_rows if row["cell"].startswith("avalanche.")]
    publish_generated_studies(
        tmp_path, avalanche_rows, default_objective=1.0, objectives=context_objectives
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

    result = run_script(
        _HPO_SCRIPT, "extend", tmp_path, c_experiment_id, experiment_id, "--chain", "avalanche"
    )
    assert result.stdout.strip() == str(experiment_id)
    assert result.stderr.splitlines() == ["avalanche\t200\t0.75"]
    rows = read_tsv_rows(bundle / "cells.tsv")
    requests = {
        row["cell"]: TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    }

    assert len(rows) == 81
    assert len(requests) == 9
    assert [row["cell"] for row in rows[:9]] == ["ethereum.lstm"] * 9
    assert [row["method_index"] for row in rows[:9]] == [str(index) for index in range(9)]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm"
    assert len({request.study_id for request in requests.values()}) == 9
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
        "hidden": 320,
        "layers": 2,
        "head_hidden": 320,
        "dropout": 0.2,
    }
    assert requests["ethereum.lstm"].methods[0].fit.model_dump() == {
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "accumulation": 2,
        "gradient_clip_norm": 0.5,
        "seed": 2026,
        "max_epochs": 40,
        "validate_every_completed_epoch": 2,
        "patience": 10,
        "min_delta": 0.01,
    }
    assert requests["polygon.lstm"].methods[-1].fit.model_dump() == {
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
    transformer = requests["polygon.transformer"].methods
    hybrid = requests["polygon.transformer_lstm"].methods
    attention_dimensions = (
        "model_width",
        "attention_heads",
        "transformer_layers",
        "feedforward_width",
        "head_hidden",
    )
    transformer_dimensions = [
        tuple(method.model.model_dump()[name] for name in attention_dimensions)
        for method in transformer
    ]
    assert transformer_dimensions == [
        (256, 4, 4, 512, 256),
        (256, 4, 4, 512, 256),
        (256, 4, 4, 512, 256),
        (192, 4, 3, 384, 192),
        (192, 4, 3, 384, 192),
        (192, 4, 3, 384, 192),
        (384, 8, 4, 768, 256),
        (384, 8, 4, 768, 256),
        (384, 8, 4, 768, 256),
    ]
    assert [
        tuple(method.model.model_dump()[name] for name in attention_dimensions) for method in hybrid
    ] == transformer_dimensions
    assert [
        (method.model.model_dump()["lstm_hidden"], method.model.model_dump()["lstm_layers"])
        for method in hybrid
    ] == [(width, 1) for width, *_ in transformer_dimensions]
    assert [
        (method.model.dropout, method.fit.learning_rate, method.fit.weight_decay)
        for method in transformer
    ] == [
        (0.2, 0.0003, 0.0001),
        (0.1, 0.0001, 0.0),
        (0.3, 0.001, 0.001),
        (0.2, 0.0001, 0.001),
        (0.1, 0.001, 0.0001),
        (0.3, 0.0003, 0.0),
        (0.2, 0.001, 0.0),
        (0.1, 0.0003, 0.001),
        (0.3, 0.0001, 0.0001),
    ]

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
    assert not bundle.exists()
