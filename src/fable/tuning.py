"""Run one Study candidate and retain its successful result."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Method, TuneRequest
from .modeling import fit_candidate
from .study import candidate_scratch_directory, retain_result


def run_candidate(
    storage_root: Path,
    request: TuneRequest,
    method: Method,
) -> None:
    method_index = request.methods.index(method)
    candidate_scratch = candidate_scratch_directory(
        storage_root,
        request.study_id,
        method_index,
    )
    result = fit_candidate(request, method, storage_root, candidate_scratch)
    retain_result(storage_root, request, method_index, result)
    shutil.rmtree(candidate_scratch)
