"""Run one Study candidate and retain its successful result."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Method, TuneRequest
from .corpus import load_corpus
from .modeling import fit_candidate
from .study import candidate_scratch_directory, retain_result
from .temporal.history import prepare_fit_history


def run_candidate(
    storage_root: Path,
    request: TuneRequest,
    method: Method,
) -> None:
    corpus = load_corpus(storage_root, request.corpus_id)
    prepared = prepare_fit_history(corpus, request.experiment)
    method_index = request.methods.index(method)
    candidate_scratch = candidate_scratch_directory(
        storage_root,
        request.study_id,
        method_index,
    )
    candidate_scratch.mkdir(parents=True, exist_ok=True)
    result = fit_candidate(request, method, prepared, candidate_scratch)
    retain_result(storage_root, request, method_index, result)
    shutil.rmtree(candidate_scratch)
