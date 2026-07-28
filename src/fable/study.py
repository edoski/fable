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


class RetainedResult(StrictFrozenRecord):
    method: Method
    objective: Annotated[float, Field(allow_inf_nan=False)]
    selected_epoch: _Epoch
    completed_epochs: _Epoch

    @model_validator(mode="after")
    def validate_epochs(self) -> Self:
        if self.selected_epoch > self.completed_epochs:
            raise ValueError("selected_epoch must not exceed completed_epochs")
        if self.completed_epochs > self.method.fit.max_epochs:
            raise ValueError("completed_epochs must not exceed method.fit.max_epochs")
        return self


class Study(StrictFrozenRecord):
    request: TuneRequest
    trials: Annotated[tuple[RetainedResult, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_methods(self) -> Self:
        for result in self.trials:
            if result.method not in self.request.methods:
                raise ValueError("Method is outside the TuneRequest")
        return self


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
        Study(request=request, trials=(result,)).model_dump_json(),
        encoding="utf-8",
    )
    os.replace(temporary, result_path)


def publish_study(storage_root: Path, study_id: UUID4) -> None:
    scratch = _study_scratch(storage_root, study_id)
    result_paths = set(scratch.glob("result-*.json"))
    if not result_paths:
        raise FileNotFoundError(scratch / "result-*.json")
    first_path = _result_path(storage_root, study_id, 0)
    if first_path not in result_paths:
        raise ValueError("result files do not match TuneRequest methods")
    first = _load_study_path(first_path)
    request = first.request
    if request.study_id != study_id:
        raise ValueError("result Study ID does not match requested Study ID")

    canonical = study_json_path(storage_root, study_id)
    if canonical.exists():
        raise FileExistsError(canonical)

    expected_paths = tuple(
        _result_path(storage_root, study_id, index) for index in range(len(request.methods))
    )
    if result_paths != set(expected_paths):
        raise ValueError("result files do not match TuneRequest methods")

    trials: list[RetainedResult] = []
    for method_index, result_path in enumerate(expected_paths):
        result_study = first if method_index == 0 else _load_study_path(result_path)
        if result_study.request != request:
            raise ValueError("result requests must be identical")
        if len(result_study.trials) != 1:
            raise ValueError("each result file must contain exactly one trial")
        result = result_study.trials[0]
        if result.method != request.methods[method_index]:
            raise ValueError("result method does not match request index")
        trials.append(result)

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

    return study.trials[source.study_result_index].method


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
