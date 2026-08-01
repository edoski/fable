from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import UUID

import lightning
import polars as pl
import pytest
import torch

import fable.modeling as modeling
from fable.addresses import artifact_checkpoint_path, evaluation_directory
from fable.config import (
    BlockWindow,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TrainRequest,
    TuneRequest,
)
from fable.evaluation import OBSERVATION_SCHEMA
from fable.experiments import ExperimentKind, ExperimentManifest, experiment_manifest_path
from fable.min_block_fee import TargetState
from fable.modeling import ArtifactAssociation
from fable.study import RetainedResult, Study
from fable.temporal import FeatureState
from tests.helpers import read_tsv_rows, run_script

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "experiments" / "k_study.py"
_HELD_OUT_SCRIPT = _ROOT / "experiments" / "held_out.py"
_HPO_EXPERIMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_CORPUS_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_METHOD = Method(
    model=LstmDefinition(family="lstm", hidden=256, layers=2, head_hidden=256, dropout=0.2),
    fit=FitMethod(
        learning_rate=3e-4,
        weight_decay=1e-4,
        accumulation=1,
        gradient_clip_norm=1.0,
        seed=2026,
        max_epochs=36,
        validate_every_completed_epoch=1,
        patience=8,
        min_delta=0.0,
    ),
)
_ARTIFACT_METHOD = Method(
    model=LstmDefinition(family="lstm", hidden=1, layers=1, head_hidden=1, dropout=0.0),
    fit=_METHOD.fit,
)


def _publish_hpo(storage_root: Path) -> None:
    cells: dict[str, UUID] = {}
    for index, cell in enumerate(
        f"{chain}.{family}"
        for chain in ("ethereum", "polygon", "avalanche")
        for family in ("lstm", "transformer", "transformer_lstm")
    ):
        study_id = UUID(f"10000000-0000-4000-8000-{index:012d}")
        method = _METHOD.model_copy(
            update={"fit": _METHOD.fit.model_copy(update={"seed": 3_000 + index})}
        )
        request = TuneRequest(
            workflow="tune",
            study_id=study_id,
            corpus_id=_CORPUS_ID,
            experiment=ExperimentSemantics(
                training_window=BlockWindow(first_parent_block=100, last_parent_block=200),
                validation_window=BlockWindow(first_parent_block=401, last_parent_block=500),
                context_blocks=100,
                horizon_blocks=5,
                ordered_features=("log_base_fee_per_gas",),
            ),
            methods=(method, _METHOD),
        )
        study = Study(
            request=request,
            trials=(
                RetainedResult(objective=2.0, selected_epoch=1, completed_epochs=1),
                RetainedResult(objective=1.0, selected_epoch=1, completed_epochs=1),
            ),
        )
        path = storage_root / "studies" / f"{study_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(study.model_dump_json(), encoding="utf-8")
        cells[cell] = study_id
    manifest_path = experiment_manifest_path(storage_root, ExperimentKind.HPO, _HPO_EXPERIMENT_ID)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(ExperimentManifest(root=cells).model_dump_json(), encoding="utf-8")


def _publish_artifacts(storage_root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        request = TrainRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        feature_count = len(request.source.experiment.ordered_features)
        association = ArtifactAssociation(
            request=request,
            feature_state=FeatureState(
                means=(0.0,) * feature_count, standard_deviations=(1.0,) * feature_count
            ),
            target_state=TargetState(mean=0.0, standard_deviation=1.0),
            method=_ARTIFACT_METHOD,
        )
        encoded = association.model_dump(mode="json")
        module = modeling._FitModule(encoded)
        path = artifact_checkpoint_path(storage_root, request.artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": {"association": encoded},
                "pytorch-lightning_version": lightning.__version__,
            },
            path,
        )


def _publish_evaluations(storage_root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        request = EvaluateRequest.model_validate_json(
            Path(row["request"]).read_bytes(), strict=True
        )
        directory = evaluation_directory(storage_root, request.evaluation_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evaluation.json").write_text(request.model_dump_json(), encoding="utf-8")
        origins = list(
            range(
                request.testing_window.first_parent_block,
                request.testing_window.last_parent_block + 1,
            )
        )
        count = len(origins)
        pl.DataFrame(
            {
                "origin_block": origins,
                "predicted_action_k": [0] * count,
                "predicted_minimum_log_base_fee": [0.0] * count,
                "minimum_action_k": [0] * count,
                "immediate_base_fee_per_gas": [1] * count,
                "immediate_effective_priority_fee_per_gas_p50": [0] * count,
                "selected_base_fee_per_gas": [1] * count,
                "selected_effective_priority_fee_per_gas_p50": [0] * count,
                "deadline_base_fee_per_gas": [1] * count,
                "deadline_effective_priority_fee_per_gas_p50": [0] * count,
                "minimum_base_fee_per_gas": [1] * count,
            },
            schema=OBSERVATION_SCHEMA,
        ).write_parquet(directory / "observations.parquet")


def test_k_study_authors_and_closes_eighty_one_selected_study_artifacts(tmp_path: Path) -> None:
    _publish_hpo(tmp_path)

    result = run_script(_SCRIPT, "prepare", tmp_path, _HPO_EXPERIMENT_ID)
    experiment_id = UUID(result.stdout.strip())
    bundle = tmp_path / "experiments" / "k_study" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
    requests = [
        TrainRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    ]
    sources = [request.source for request in requests]

    assert len(rows) == 81
    assert [row["cell"] for row in rows[:9]] == [
        "ethereum.lstm.K2",
        "ethereum.lstm.K3",
        "ethereum.lstm.K4",
        "ethereum.lstm.K5",
        "ethereum.lstm.K10",
        "ethereum.lstm.K25",
        "ethereum.lstm.K50",
        "ethereum.lstm.K100",
        "ethereum.lstm.K200",
    ]
    assert rows[-1]["cell"] == "avalanche.transformer_lstm.K200"
    assert [source.experiment.horizon_blocks for source in sources[:9]] == [
        2,
        3,
        4,
        5,
        10,
        25,
        50,
        100,
        200,
    ]
    assert {source.study_result_index for source in sources} == {1}
    assert len({request.artifact_id for request in requests}) == 81

    for row in rows:
        checkpoint = tmp_path / "artifacts" / f"{row['artifact_id']}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
    with pytest.raises(subprocess.CalledProcessError):
        run_script(_SCRIPT, "close", tmp_path, experiment_id)
    assert bundle.is_dir()

    _publish_artifacts(tmp_path, rows)
    run_script(_SCRIPT, "close", tmp_path, experiment_id)

    canonical = tmp_path / "experiments" / "k_study" / str(experiment_id)
    manifest = ExperimentManifest.model_validate_json(
        (canonical / "manifest.json").read_bytes(), strict=True
    )
    assert len(manifest.root) == 81
    assert [str(record_id) for record_id in manifest.root.values()] == [
        row["artifact_id"] for row in rows
    ]
    assert not bundle.exists()

    corpus = {
        "request": {
            "corpus_id": str(_CORPUS_ID),
            "definition": {"chain_id": 1, "first_block": 0, "last_block": 1_000},
        },
        "finalized_anchor": {"block_number": 1_000, "block_hash": "0" * 64},
    }
    corpus_path = tmp_path / "corpora" / str(_CORPUS_ID) / "corpus.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    held_out_result = run_script(
        _HELD_OUT_SCRIPT, "prepare", tmp_path, _HPO_EXPERIMENT_ID, experiment_id
    )
    held_out_experiment_id = UUID(held_out_result.stdout.strip())

    held_out = tmp_path / "experiments" / "held_out" / f".{held_out_experiment_id}"
    evaluation_rows = read_tsv_rows(held_out / "cells.tsv")
    evaluation_requests = [
        EvaluateRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in evaluation_rows
    ]
    assert len(evaluation_rows) == 81
    assert len({request.evaluation_id for request in evaluation_requests}) == 81
    assert [request.testing_window for request in evaluation_requests[:4]] == [
        BlockWindow(first_parent_block=701, last_parent_block=803),
        BlockWindow(first_parent_block=701, last_parent_block=802),
        BlockWindow(first_parent_block=701, last_parent_block=801),
        BlockWindow(first_parent_block=701, last_parent_block=800),
    ]
    for row in evaluation_rows:
        (tmp_path / "evaluations" / row["evaluation_id"]).mkdir(parents=True)
    with pytest.raises(subprocess.CalledProcessError):
        run_script(_HELD_OUT_SCRIPT, "close", tmp_path, held_out_experiment_id)
    assert held_out.is_dir()

    _publish_evaluations(tmp_path, evaluation_rows)
    run_script(_HELD_OUT_SCRIPT, "close", tmp_path, held_out_experiment_id)
    held_out_canonical = tmp_path / "experiments" / "held_out" / str(held_out_experiment_id)
    held_out_manifest = ExperimentManifest.model_validate_json(
        (held_out_canonical / "manifest.json").read_bytes(), strict=True
    )
    assert len(held_out_manifest.root) == 81
    assert [str(record_id) for record_id in held_out_manifest.root.values()] == [
        row["evaluation_id"] for row in evaluation_rows
    ]
    assert not held_out.exists()

    k_manifest_path = canonical / "manifest.json"
    k_manifest_path.write_text(
        ExperimentManifest(
            root={
                cell: artifact_id
                for cell, artifact_id in manifest.root.items()
                if not cell.endswith(".K200")
            }
        ).model_dump_json(),
        encoding="utf-8",
    )
    derived_result = run_script(
        _HELD_OUT_SCRIPT, "prepare", tmp_path, _HPO_EXPERIMENT_ID, experiment_id
    )
    derived_bundle = tmp_path / "experiments" / "held_out" / f".{derived_result.stdout.strip()}"
    derived_rows = read_tsv_rows(derived_bundle / "cells.tsv")
    derived_requests = [
        EvaluateRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in derived_rows
    ]
    assert len(derived_rows) == 72
    assert [request.testing_window for request in derived_requests[:4]] == [
        BlockWindow(first_parent_block=601, last_parent_block=903),
        BlockWindow(first_parent_block=601, last_parent_block=902),
        BlockWindow(first_parent_block=601, last_parent_block=901),
        BlockWindow(first_parent_block=601, last_parent_block=900),
    ]
