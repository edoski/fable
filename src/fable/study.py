"""Immutable Study publication and selected-Method loading."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated, Self, TypeAlias

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from .addresses import study_json_path
from .config import (
    Method,
    SelectedStudySource,
    TrainingDefinition,
    TuneRequest,
)

__all__ = [
    "RetainedResult",
    "Study",
    "apply_method",
    "load_selected_method",
    "publish_study",
    "retain_result",
]

_Objective: TypeAlias = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]
_Epoch: TypeAlias = Annotated[int, Field(strict=True, ge=1)]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class RetainedResult(_FrozenRecord):
    method: Method
    objective: _Objective
    selected_epoch: _Epoch
    completed_epochs: _Epoch

    @model_validator(mode="after")
    def validate_epochs(self) -> Self:
        if self.selected_epoch > self.completed_epochs:
            raise ValueError("selected_epoch must not exceed completed_epochs")
        if self.completed_epochs > self.method.fit.max_epochs:
            raise ValueError("completed_epochs must not exceed method.fit.max_epochs")
        return self


class Study(_FrozenRecord):
    request: TuneRequest
    trials: Annotated[tuple[RetainedResult, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_methods(self) -> Self:
        for result in self.trials:
            _require_method(self.request, result.method)
        return self


def apply_method(request: TuneRequest, method: Method) -> TrainingDefinition:
    """Approve a Method against its TuneRequest and compose its definition."""

    _require_method(request, method)
    return TrainingDefinition(experiment=request.experiment, method=method)


def retain_result(
    storage_root: Path,
    request: TuneRequest,
    method_index: int,
    result: RetainedResult,
) -> None:
    if result.method != request.methods[method_index]:
        raise ValueError("result method does not match request index")

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
    first = _load_study(min(result_paths))
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
        result_study = _load_study(result_path)
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
    hidden.rename(canonical)
    shutil.rmtree(scratch)


def load_selected_method(
    storage_root: Path,
    source: SelectedStudySource,
) -> Method:
    study = _load_study(study_json_path(storage_root, source.study_id))
    if study.request.study_id != source.study_id:
        raise ValueError("selected source Study ID does not match canonical Study")
    if study.request.corpus_id != source.corpus_id:
        raise ValueError("selected source Corpus ID does not match canonical Study")

    return study.trials[source.study_result_index].method


def _require_method(request: TuneRequest, method: Method) -> None:
    if method not in request.methods:
        raise ValueError("Method is outside the TuneRequest")


def _study_scratch(storage_root: Path, study_id: UUID4) -> Path:
    return storage_root / "studies" / f".{study_id}"


def _result_path(storage_root: Path, study_id: UUID4, method_index: int) -> Path:
    return _study_scratch(storage_root, study_id) / f"result-{method_index}.json"


def _load_study(path: Path) -> Study:
    return Study.model_validate_json(path.read_bytes(), strict=True)
