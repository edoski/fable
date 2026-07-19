"""Run one Study candidate and retain its successful result."""

from __future__ import annotations

from pathlib import Path

from .config import Method, TuneRequest
from .corpus import load_corpus
from .modeling import FitDeployment
from .modeling.artifacts import _run_candidate
from .study import retain_result
from .temporal.history import prepare_fit_history

__all__ = ["run_candidate"]


def run_candidate(
    storage_root: Path,
    request: TuneRequest,
    method: Method,
    deployment: FitDeployment,
) -> None:
    corpus = load_corpus(storage_root, request.corpus_id)
    prepared = prepare_fit_history(corpus, request.study_definition.experiment)
    study_scratch = storage_root / "studies" / f".{request.study_id}"
    study_scratch.mkdir(parents=True, exist_ok=True)
    result = _run_candidate(request, method, prepared, study_scratch, deployment)
    retain_result(storage_root, request, result)
