# pyright: strict

"""Benchmark selection ledger models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import Field

from ..config.models import ArtifactVariant, ProblemSpec
from ..config.selections import WorkflowSelection
from ..core.config_model import ConfigModel

_ROOT_FIELDS = frozenset({"dataset_id", "study_id", "artifact_id"})


class BenchmarkSelectionLedger(ConfigModel):
    surface: str | None = None
    chain: str | None = None
    problem: str | None = None
    features: str | None = None
    model: str | None = None
    tuning_space: str | None = None
    objective: str | None = None
    evaluation: str | None = None
    training: str | None = None
    split: str | None = None
    tuning: str | None = None
    study: str | None = None
    variant: ArtifactVariant | None = None
    trial_count: int | None = Field(default=None, gt=0)
    delay_seconds: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    storage_root: Path | None = None


def materialize_selection_ledger(
    *,
    source_row: Mapping[str, object],
    workflow_selection: WorkflowSelection,
) -> BenchmarkSelectionLedger:
    fields = set(BenchmarkSelectionLedger.model_fields)
    payload: dict[str, object] = {}
    for key, value in source_row.items():
        if key in _ROOT_FIELDS or key not in fields:
            continue
        if isinstance(value, ProblemSpec):
            payload[key] = value.id
            continue
        if value is not None:
            payload[key] = value
    for key in fields:
        if key in payload or not hasattr(workflow_selection, key):
            continue
        value = getattr(workflow_selection, key)
        if value is not None:
            payload[key] = value
    return BenchmarkSelectionLedger.model_validate(payload)
