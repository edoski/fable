from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pytest import MonkeyPatch

from fable import tuning
from fable.config import (
    AdamWMethod,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmCapacity,
    LstmMethod,
    LstmMethodSpace,
    OriginWindow,
    StudyDefinition,
    TuneRequest,
)
from fable.corpus import Corpus
from fable.modeling import FitDeployment
from fable.study import RetainedResult
from fable.temporal.history import HistoricalPreparation

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")

FIT = FitMethod(
    accumulation=1,
    gradient_clip_norm=0.75,
    scheduler="none",
    seed=17,
    max_epochs=12,
    validate_every_completed_epoch=1,
    patience=4,
    min_delta=0.01,
    improvement="strict_lower",
    restore="earliest_best",
)
METHOD = LstmMethod(
    family="lstm",
    capacity=LstmCapacity(hidden=16, layers=1, head_hidden=8),
    dropout=0.2,
    optimizer=AdamWMethod(learning_rate=3e-4, weight_decay=1e-4),
    training_batch=8,
    fit=FIT,
)
EXPERIMENT = ExperimentSemantics(
    training_window=OriginWindow(
        role="training",
        first_parent_block=10,
        last_parent_block=20,
    ),
    validation_window=OriginWindow(
        role="validation",
        first_parent_block=30,
        last_parent_block=35,
    ),
    context_blocks=3,
    horizon_blocks=2,
    ordered_features=("log_base_fee_per_gas",),
    loss=LossDefinition(
        classification_algorithm="cross_entropy",
        classification_weighting="unweighted",
        regression_algorithm="smooth_l1",
        regression_threshold=0.75,
        classification_scale=1.0,
        regression_scale=1.0,
    ),
)
REQUEST = TuneRequest(
    workflow="tune",
    study_id=STUDY_ID,
    corpus_id=CORPUS_ID,
    study_definition=StudyDefinition(
        experiment=EXPERIMENT,
        method_space=LstmMethodSpace(family="lstm", methods=(METHOD,)),
    ),
)
DEPLOYMENT = FitDeployment(
    deterministic=True,
    benchmark=False,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=False,
    float32_matmul_precision="highest",
    cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=False,
)
RESULT = RetainedResult(
    method=METHOD,
    objective=0.4,
    selected_epoch=3,
    completed_epochs=8,
)


def test_run_candidate_prepares_fits_and_retains_one_result(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    corpus = cast(Corpus, object())
    prepared = cast(HistoricalPreparation, object())
    calls: list[tuple[str, tuple[object, ...]]] = []

    def load_corpus(storage_root: Path, corpus_id: UUID) -> Corpus:
        calls.append(("load", (storage_root, corpus_id)))
        return corpus

    def prepare_fit_history(
        loaded_corpus: Corpus,
        experiment: ExperimentSemantics,
    ) -> HistoricalPreparation:
        calls.append(("prepare", (loaded_corpus, experiment)))
        return prepared

    def run_fit(
        request: TuneRequest,
        method: LstmMethod,
        preparation: HistoricalPreparation,
        scratch: Path,
        deployment: FitDeployment,
    ) -> RetainedResult:
        assert scratch.is_dir()
        assert deployment is DEPLOYMENT
        calls.append(("fit", (request, method, preparation, scratch, deployment)))
        return RESULT

    def retain_result(
        storage_root: Path,
        request: TuneRequest,
        result: RetainedResult,
    ) -> None:
        calls.append(("retain", (storage_root, request, result)))

    monkeypatch.setattr(tuning, "load_corpus", load_corpus)
    monkeypatch.setattr(tuning, "prepare_fit_history", prepare_fit_history)
    monkeypatch.setattr(tuning, "_run_candidate", run_fit)
    monkeypatch.setattr(tuning, "retain_result", retain_result)

    returned = tuning.run_candidate(tmp_path, REQUEST, METHOD, DEPLOYMENT)

    scratch = tmp_path / "studies" / f".{STUDY_ID}"
    assert calls == [
        ("load", (tmp_path, CORPUS_ID)),
        ("prepare", (corpus, EXPERIMENT)),
        ("fit", (REQUEST, METHOD, prepared, scratch, DEPLOYMENT)),
        ("retain", (tmp_path, REQUEST, RESULT)),
    ]
    assert returned is None


@pytest.mark.parametrize(
    ("failure_stage", "expected_calls"),
    [
        ("load", ["load"]),
        ("prepare", ["load", "prepare"]),
        ("fit", ["load", "prepare", "fit"]),
        ("retain", ["load", "prepare", "fit", "retain"]),
    ],
)
def test_run_candidate_preserves_failure_boundaries(
    failure_stage: str,
    expected_calls: list[str],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    corpus = cast(Corpus, object())
    prepared = cast(HistoricalPreparation, object())
    failure = RuntimeError(failure_stage)
    calls: list[str] = []

    def load_corpus(storage_root: Path, corpus_id: UUID) -> Corpus:
        del storage_root, corpus_id
        calls.append("load")
        if failure_stage == "load":
            raise failure
        return corpus

    def prepare_fit_history(
        loaded_corpus: Corpus,
        experiment: ExperimentSemantics,
    ) -> HistoricalPreparation:
        del loaded_corpus, experiment
        calls.append("prepare")
        if failure_stage == "prepare":
            raise failure
        return prepared

    def run_fit(
        request: TuneRequest,
        method: LstmMethod,
        preparation: HistoricalPreparation,
        scratch: Path,
        deployment: FitDeployment,
    ) -> RetainedResult:
        del request, method, preparation, deployment
        calls.append("fit")
        (scratch / "last.ckpt").touch()
        if failure_stage == "fit":
            raise failure
        return RESULT

    def retain_result(
        storage_root: Path,
        request: TuneRequest,
        result: RetainedResult,
    ) -> None:
        del storage_root, request, result
        calls.append("retain")
        if failure_stage == "retain":
            raise failure

    monkeypatch.setattr(tuning, "load_corpus", load_corpus)
    monkeypatch.setattr(tuning, "prepare_fit_history", prepare_fit_history)
    monkeypatch.setattr(tuning, "_run_candidate", run_fit)
    monkeypatch.setattr(tuning, "retain_result", retain_result)

    with pytest.raises(RuntimeError) as caught:
        tuning.run_candidate(tmp_path, REQUEST, METHOD, DEPLOYMENT)

    scratch = tmp_path / "studies" / f".{STUDY_ID}"
    assert caught.value is failure
    assert calls == expected_calls
    assert scratch.exists() is (failure_stage in {"fit", "retain"})
    assert (scratch / "last.ckpt").exists() is (failure_stage in {"fit", "retain"})
