# pyright: strict

"""Benchmark selection ledger models."""

from __future__ import annotations

from collections.abc import Mapping

from ...config.selections import (
    WorkflowSelection,
    benchmark_selection_coordinate_fields,
    benchmark_selection_root_fields,
)
from ._models import BenchmarkSelectionLedger


def materialize_selection_ledger(
    *,
    selection_row: Mapping[str, object],
    workflow_selection: WorkflowSelection,
) -> BenchmarkSelectionLedger:
    fields = benchmark_selection_coordinate_fields() & set(BenchmarkSelectionLedger.model_fields)
    root_fields = benchmark_selection_root_fields()
    payload: dict[str, object] = {}
    for key, value in selection_row.items():
        if key in root_fields or key not in fields:
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
