from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

from fable.config import TuneRequest
from fable.study import RetainedResult, Study


def publish_generated_studies(
    storage_root: Path,
    rows: list[dict[str, str]],
    *,
    default_objective: float,
    objectives: Mapping[str, float] | None = None,
) -> None:
    objectives = objectives or {}
    seen: set[UUID] = set()
    for row in rows:
        request = TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        if request.study_id in seen:
            continue
        seen.add(request.study_id)
        objective = objectives.get(row["cell"], default_objective)
        study = Study(
            request=request,
            trials=tuple(
                RetainedResult(
                    objective=objective + method_index, selected_epoch=1, completed_epochs=1
                )
                for method_index, _ in enumerate(request.methods)
            ),
        )
        path = storage_root / "studies" / f"{request.study_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(study.model_dump_json(), encoding="utf-8")
