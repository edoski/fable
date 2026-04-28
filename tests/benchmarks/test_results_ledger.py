from __future__ import annotations

import csv
from pathlib import Path

EXPECTED_HEADER = [
    "recorded_at_utc",
    "git_commit",
    "execution_ref",
    "artifact_id",
    "evaluation_storage_id",
    "chain",
    "dataset",
    "surface",
    "features",
    "model",
    "problem",
    "prediction",
    "objective",
    "evaluation",
    "delay_seconds",
    "variant",
    "study",
    "sample_count",
    "total_events",
    "profit_over_baseline",
    "cost_over_optimum",
    "baseline_cost_over_optimum",
    "total_loss",
    "offset_accuracy",
    "macro_f1",
    "classification_loss",
    "regression_loss",
    "log_fee_mae",
    "log_fee_mse",
    "exact_optimum_hit_rate",
    "notes",
]


def test_results_ledger_template_header() -> None:
    path = Path("benchmarks/results.csv")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows == [EXPECTED_HEADER]
