from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fable.config import (
    BlockWindow,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainRequest,
    TuneRequest,
)
from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    write_experiment_manifest,
)
from fable.study import RetainedResult, Study
from tests.helpers import read_tsv_rows, run_script

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "experiments" / "k_study.py"
_HELD_OUT_SCRIPT = _ROOT / "experiments" / "held_out.py"
_HPO_EXPERIMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_CORPUS_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_METHOD = Method(
    model=LstmDefinition(
        family="lstm",
        hidden=256,
        layers=2,
        head_hidden=256,
        dropout=0.2,
    ),
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


def _publish_hpo(storage_root: Path) -> None:
    entries: list[ExperimentEntry] = []
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
                RetainedResult(
                    objective=2.0,
                    selected_epoch=1,
                    completed_epochs=1,
                ),
                RetainedResult(
                    objective=1.0,
                    selected_epoch=1,
                    completed_epochs=1,
                ),
            ),
        )
        path = storage_root / "studies" / f"{study_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(study.model_dump_json(), encoding="utf-8")
        entries.append(ExperimentEntry(cell=cell, record_id=study_id))
    write_experiment_manifest(
        storage_root,
        ExperimentKind.HPO,
        ExperimentManifest(
            experiment_id=_HPO_EXPERIMENT_ID,
            entries=tuple(entries),
        ),
    )


def test_k_study_authors_and_closes_eighty_one_selected_study_artifacts(
    tmp_path: Path,
) -> None:
    _publish_hpo(tmp_path)

    result = run_script(
        _SCRIPT,
        "prepare",
        tmp_path,
        _HPO_EXPERIMENT_ID,
    )
    experiment_id = UUID(result.stdout.strip())
    bundle = tmp_path / "experiments" / "k_study" / f".{experiment_id}"
    rows = read_tsv_rows(bundle / "cells.tsv")
    requests = [
        TrainRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    ]
    sources = [
        request.source for request in requests if isinstance(request.source, SelectedStudySource)
    ]

    assert experiment_id.version == 4
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
    assert len(sources) == 81
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
    run_script(_SCRIPT, "close", tmp_path, experiment_id)

    manifest = ExperimentManifest.model_validate_json(
        (tmp_path / "experiments" / "k_study" / f"{experiment_id}.json").read_bytes(),
        strict=True,
    )
    assert len(manifest.entries) == 81
    assert [str(entry.record_id) for entry in manifest.entries] == [
        row["artifact_id"] for row in rows
    ]
    assert not bundle.exists()

    corpus = {
        "request": {
            "corpus_id": str(_CORPUS_ID),
            "definition": {
                "chain_id": 1,
                "first_block": 0,
                "last_block": 1_000,
            },
        },
        "finalized_anchor": {
            "block_number": 1_000,
            "block_hash": "0" * 64,
        },
    }
    corpus_path = tmp_path / "corpora" / str(_CORPUS_ID) / "corpus.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    held_out_result = run_script(
        _HELD_OUT_SCRIPT,
        "prepare",
        tmp_path,
        _HPO_EXPERIMENT_ID,
        experiment_id,
    )
    held_out_experiment_id = UUID(held_out_result.stdout.strip())

    held_out = tmp_path / "experiments" / "held_out" / f".{held_out_experiment_id}"
    evaluation_rows = read_tsv_rows(held_out / "cells.tsv")
    evaluation_requests = [
        EvaluateRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in evaluation_rows
    ]
    assert held_out_experiment_id.version == 4
    assert len(evaluation_rows) == 81
    assert [request.testing_window for request in evaluation_requests[:4]] == [
        BlockWindow(first_parent_block=701, last_parent_block=803),
        BlockWindow(first_parent_block=701, last_parent_block=802),
        BlockWindow(first_parent_block=701, last_parent_block=801),
        BlockWindow(first_parent_block=701, last_parent_block=800),
    ]
    for row in evaluation_rows:
        (tmp_path / "evaluations" / row["evaluation_id"]).mkdir(parents=True)
    run_script(_HELD_OUT_SCRIPT, "close", tmp_path, held_out_experiment_id)
    held_out_manifest = ExperimentManifest.model_validate_json(
        (tmp_path / "experiments" / "held_out" / f"{held_out_experiment_id}.json").read_bytes(),
        strict=True,
    )
    assert len(held_out_manifest.entries) == 81
    assert [str(entry.record_id) for entry in held_out_manifest.entries] == [
        row["evaluation_id"] for row in evaluation_rows
    ]
    assert not held_out.exists()
