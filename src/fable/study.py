"""Immutable Study publication and selected-Method loading."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated, Self, TypeAlias
from uuid import UUID

import polars as pl
import torch
from pydantic import UUID4, Field, model_validator

from .addresses import (
    study_directory,
    study_json_path,
    study_trial_checkpoint_path,
    study_trial_observations_path,
)
from .config import Method, SelectedStudySource, TrainingDefinition, TuneRequest
from .observations import reduce_observations, validate_observations
from .records import StrictFrozenRecord

_Epoch: TypeAlias = Annotated[int, Field(ge=1)]


class RetainedResult(StrictFrozenRecord):
    objective: Annotated[float, Field(allow_inf_nan=False)]
    selected_epoch: _Epoch
    completed_epochs: _Epoch

    @model_validator(mode="after")
    def validate_epochs(self) -> Self:
        if self.selected_epoch > self.completed_epochs:
            raise ValueError("selected_epoch must not exceed completed_epochs")
        return self


class Study(StrictFrozenRecord):
    request: TuneRequest
    trials: tuple[RetainedResult, ...]

    @model_validator(mode="after")
    def validate_trials(self) -> Self:
        if len(self.trials) != len(self.request.methods):
            raise ValueError("trials must align with request methods")
        for method, result in zip(self.request.methods, self.trials, strict=True):
            if result.completed_epochs > method.fit.max_epochs:
                raise ValueError("completed_epochs must not exceed method.fit.max_epochs")
        return self

    def best_result(self) -> tuple[int, RetainedResult]:
        return min(enumerate(self.trials), key=lambda indexed: indexed[1].objective)


class _CandidateResult(StrictFrozenRecord):
    request: TuneRequest
    result: RetainedResult


def retain_result(
    storage_root: Path,
    request: TuneRequest,
    method_index: int,
    result: RetainedResult,
    selected_checkpoint: Path,
    observations: pl.DataFrame,
) -> None:
    """Atomically retain one completed candidate inside Study scratch."""

    retained = _retained_trial_directory(storage_root, request.study_id, method_index)
    temporary = retained.with_name(f".{retained.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    os.link(selected_checkpoint, temporary / "selected.ckpt")
    observations.write_parquet(temporary / "validation.parquet")
    (temporary / "result.json").write_text(
        _CandidateResult(request=request, result=result).model_dump_json(), encoding="utf-8"
    )
    _validate_trial(temporary, request, method_index, result)
    temporary.rename(retained)


def publish_study(storage_root: Path, study_id: UUID4) -> None:
    scratch = _study_scratch(storage_root, study_id)
    first = _load_candidate_result_path(_retained_trial_directory(storage_root, study_id, 0))
    request = first.request
    if request.study_id != study_id:
        raise ValueError("result Study ID does not match requested Study ID")

    expected_trials = tuple(
        _retained_trial_directory(storage_root, study_id, index)
        for index in range(len(request.methods))
    )
    if set(scratch.glob("trial-*")) != set(expected_trials):
        raise ValueError("retained trials do not match TuneRequest methods")

    trials = []
    completed = scratch.with_name(f".{study_id}.completed")
    shutil.rmtree(completed, ignore_errors=True)
    (completed / "trials").mkdir(parents=True)
    for index, retained in enumerate(expected_trials):
        candidate = first if index == 0 else _load_candidate_result_path(retained)
        if candidate.request != request:
            raise ValueError("result requests must be identical")
        _validate_trial(retained, request, index, candidate.result)
        trial = completed / "trials" / str(index)
        trial.mkdir()
        os.link(retained / "selected.ckpt", trial / "selected.ckpt")
        os.link(retained / "validation.parquet", trial / "validation.parquet")
        trials.append(candidate.result)

    (completed / "study.json").write_text(
        Study(request=request, trials=tuple(trials)).model_dump_json(), encoding="utf-8"
    )
    canonical = study_directory(storage_root, study_id)
    if canonical.exists():
        raise FileExistsError(canonical)
    try:
        completed.rename(canonical)
    except OSError as error:
        if canonical.exists():
            raise FileExistsError(canonical) from error
        raise
    shutil.rmtree(scratch)


def load_study(storage_root: Path, study_id: UUID) -> Study:
    study = Study.model_validate_json(study_json_path(storage_root, study_id).read_bytes())
    if study.request.study_id != study_id:
        raise ValueError("Study ID does not match requested Study ID")
    for index in range(len(study.trials)):
        if not study_trial_checkpoint_path(storage_root, study_id, index).is_file():
            raise FileNotFoundError("Study trial selected checkpoint is missing")
        validate_observations(study_trial_observations_path(storage_root, study_id, index))
    return study


def reduce_study_trial(storage_root: Path, study_id: UUID, method_index: int) -> pl.DataFrame:
    study = load_study(storage_root, study_id)
    study.request.method_at(method_index)
    result = study.trials[method_index]
    observations_path = study_trial_observations_path(storage_root, study_id, method_index)
    metrics = reduce_observations(observations_path)
    if result.objective != metrics["base_fee_optimality_gap"][0]:
        raise ValueError("Study objective must equal validation observations")
    return metrics


def load_selected_method(storage_root: Path, source: SelectedStudySource) -> Method:
    study = load_study(storage_root, source.study_id)
    if study.request.corpus_id != source.corpus_id:
        raise ValueError("selected source Corpus ID does not match canonical Study")
    return study.request.method_at(source.study_result_index)


def _validate_trial(
    retained: Path, request: TuneRequest, method_index: int, result: RetainedResult
) -> None:
    expected = TrainingDefinition(
        experiment=request.experiment, method=request.method_at(method_index)
    ).model_dump(mode="json")
    checkpoint = torch.load(retained / "selected.ckpt", map_location="cpu", weights_only=True)
    if checkpoint["hyper_parameters"]["association"] != expected:
        raise ValueError("selected checkpoint association must match the request Method")
    metrics = reduce_observations(retained / "validation.parquet")
    if result.objective != metrics["base_fee_optimality_gap"][0]:
        raise ValueError("result objective must equal validation observations")


def _study_scratch(storage_root: Path, study_id: UUID) -> Path:
    return storage_root / "studies" / f".{study_id}"


def candidate_scratch_directory(storage_root: Path, study_id: UUID4, method_index: int) -> Path:
    return _study_scratch(storage_root, study_id) / f"candidate-{method_index}"


def _retained_trial_directory(storage_root: Path, study_id: UUID, method_index: int) -> Path:
    return _study_scratch(storage_root, study_id) / f"trial-{method_index}"


def _load_candidate_result_path(path: Path) -> _CandidateResult:
    return _CandidateResult.model_validate_json((path / "result.json").read_bytes())
