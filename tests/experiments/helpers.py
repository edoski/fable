from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

import polars as pl

from fable.addresses import (
    study_json_path,
    study_trial_checkpoint_path,
    study_trial_observations_path,
)
from fable.config import TuneRequest
from fable.observations import OBSERVATION_SCHEMA
from fable.study import RetainedResult, Study


def publish_test_study(storage_root: Path, study: Study) -> None:
    path = study_json_path(storage_root, study.request.study_id)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(study.model_dump_json(), encoding="utf-8")
    for index, result in enumerate(study.trials):
        checkpoint = study_trial_checkpoint_path(storage_root, study.request.study_id, index)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        minimum = 100
        selected = minimum + round(minimum * result.objective)
        pl.DataFrame(
            [
                {
                    "origin_block": 1,
                    "predicted_action_k": 1,
                    "predicted_minimum_log_base_fee": 1.0,
                    "minimum_action_k": 0,
                    "immediate_base_fee_per_gas": 200,
                    "immediate_effective_priority_fee_per_gas_p50": 2,
                    "selected_base_fee_per_gas": selected,
                    "selected_effective_priority_fee_per_gas_p50": 1,
                    "deadline_base_fee_per_gas": 150,
                    "deadline_effective_priority_fee_per_gas_p50": 1,
                    "minimum_base_fee_per_gas": minimum,
                }
            ],
            schema=OBSERVATION_SCHEMA,
        ).write_parquet(study_trial_observations_path(storage_root, study.request.study_id, index))


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
        publish_test_study(storage_root, study)
