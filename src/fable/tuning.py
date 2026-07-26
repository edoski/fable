"""Run one Study candidate and retain its successful result."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Deployment, Method, TuneRequest
from .corpus import load_corpus
from .modeling import _run_candidate
from .study import retain_result
from .temporal.history import prepare_fit_history

__all__ = ["run_candidate"]


def run_candidate(
    storage_root: Path,
    request: TuneRequest,
    method: Method,
    deployment: Deployment,
) -> None:
    corpus = load_corpus(storage_root, request.corpus_id)
    prepared = prepare_fit_history(corpus, request.experiment)
    method_index = request.methods.index(method)
    study_scratch = storage_root / "studies" / f".{request.study_id}"
    candidate_scratch = study_scratch / f"candidate-{method_index}"
    candidate_scratch.mkdir(parents=True, exist_ok=True)
    result = _run_candidate(request, method, prepared, candidate_scratch, deployment)
    retain_result(storage_root, request, method_index, result)
    shutil.rmtree(candidate_scratch)
