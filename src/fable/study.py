"""Immutable Study publication and selected-Method loading."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated, Self, TypeAlias
from uuid import UUID

from pydantic import UUID4, Field, model_validator

from .addresses import study_json_path
from .config import (
    Method,
    SelectedStudySource,
    TuneRequest,
)
from .records import StrictFrozenRecord

_Epoch: TypeAlias = Annotated[int, Field(ge=1)]
_MethodIndex: TypeAlias = Annotated[int, Field(ge=0)]


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
    trials: Annotated[tuple[RetainedResult, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_trials(self) -> Self:
        if len(self.trials) != len(self.request.methods):
            raise ValueError("trials must align with request methods")
        for method, result in zip(self.request.methods, self.trials, strict=True):
            if result.completed_epochs > method.fit.max_epochs:
                raise ValueError("completed_epochs must not exceed method.fit.max_epochs")
        return self

    def best_result(self) -> tuple[int, RetainedResult]:
        return min(
            enumerate(self.trials),
            key=lambda indexed: indexed[1].objective,
        )


class _CandidateResult(StrictFrozenRecord):
    request: TuneRequest
    method_index: _MethodIndex
    result: RetainedResult


def retain_result(
    storage_root: Path,
    request: TuneRequest,
    method_index: int,
    result: RetainedResult,
) -> None:
    result_path = _result_path(storage_root, request.study_id, method_index)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(
        _CandidateResult(
            request=request,
            method_index=method_index,
            result=result,
        ).model_dump_json(),
        encoding="utf-8",
    )
    os.replace(temporary, result_path)


def publish_study(storage_root: Path, study_id: UUID4) -> None:
    scratch = _study_scratch(storage_root, study_id)
    first_path = _result_path(storage_root, study_id, 0)
    first = _load_candidate_result_path(first_path)
    request = first.request
    if request.study_id != study_id:
        raise ValueError("result Study ID does not match requested Study ID")

    canonical = study_json_path(storage_root, study_id)
    if canonical.exists():
        raise FileExistsError(canonical)

    expected_paths = tuple(
        _result_path(storage_root, study_id, index) for index in range(len(request.methods))
    )
    if set(scratch.glob("result-*.json")) != set(expected_paths):
        raise ValueError("result files do not match TuneRequest methods")

    trials: list[RetainedResult] = []
    for method_index, result_path in enumerate(expected_paths):
        candidate = first if method_index == 0 else _load_candidate_result_path(result_path)
        if candidate.request != request:
            raise ValueError("result requests must be identical")
        if candidate.method_index != method_index:
            raise ValueError("result method index does not match file index")
        trials.append(candidate.result)

    hidden = canonical.with_name(f".{canonical.name}")
    hidden.write_text(
        Study(request=request, trials=tuple(trials)).model_dump_json(),
        encoding="utf-8",
    )
    shutil.rmtree(scratch)
    os.link(hidden, canonical)
    try:
        hidden.unlink()
    except OSError:
        pass


def load_study(storage_root: Path, study_id: UUID) -> Study:
    study = _load_study_path(study_json_path(storage_root, study_id))
    if study.request.study_id != study_id:
        raise ValueError("Study ID does not match requested Study ID")
    return study


def load_selected_method(
    storage_root: Path,
    source: SelectedStudySource,
) -> Method:
    study = load_study(storage_root, source.study_id)
    if study.request.corpus_id != source.corpus_id:
        raise ValueError("selected source Corpus ID does not match canonical Study")

    return study.request.method_at(source.study_result_index)


def _study_scratch(storage_root: Path, study_id: UUID4) -> Path:
    return storage_root / "studies" / f".{study_id}"


def candidate_scratch_directory(
    storage_root: Path,
    study_id: UUID4,
    method_index: int,
) -> Path:
    return _study_scratch(storage_root, study_id) / f"candidate-{method_index}"


def _result_path(storage_root: Path, study_id: UUID4, method_index: int) -> Path:
    return _study_scratch(storage_root, study_id) / f"result-{method_index}.json"


def _load_study_path(path: Path) -> Study:
    return Study.model_validate_json(path.read_bytes(), strict=True)


def _load_candidate_result_path(path: Path) -> _CandidateResult:
    return _CandidateResult.model_validate_json(path.read_bytes(), strict=True)
