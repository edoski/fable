# pyright: strict

"""CSV export adapter for benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

from .result_store import BENCHMARK_RESULT_INDEX_PATH, IndexedBenchmarkResult, list_indexed_results

LEDGER_COLUMNS = (
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
)


def export_results_csv(
    *,
    output_path: Path,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
    benchmark: str | None = None,
    chain: str | None = None,
    model: str | None = None,
    evaluation: str | None = None,
) -> list[dict[str, str]]:
    rows = [
        result_record_csv_row(row)
        for row in list_indexed_results(
            index_path,
            benchmark=benchmark,
            chain=chain,
            model=model,
            evaluation=evaluation,
        )
    ]
    write_results_csv(output_path, rows)
    return rows


def write_results_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in LEDGER_COLUMNS})


def result_record_csv_row(record: IndexedBenchmarkResult) -> dict[str, str]:
    row = {
        "recorded_at_utc": record.recorded_at_utc,
        "git_commit": record.git_commit,
        "execution_ref": record.execution_ref,
        "artifact_id": record.artifact_id,
        "evaluation_storage_id": record.evaluation_storage_id,
        "chain": record.chain_name,
        "dataset": record.artifact_dataset_name,
        "surface": record.surface,
        "features": record.features_id,
        "model": record.model_id,
        "problem": record.problem_id,
        "prediction": record.prediction_id,
        "objective": record.objective_id,
        "evaluation": record.evaluator_id,
        "delay_seconds": str(record.delay_seconds),
        "variant": record.variant,
        "study": record.study_name or "",
        "sample_count": str(record.sample_count),
        "total_events": str(record.total_events),
        "notes": "",
    }
    for column in LEDGER_COLUMNS:
        if column in row:
            continue
        row[column] = _metric_cell(record.metrics.get(column))
    return {column: row[column] for column in LEDGER_COLUMNS}


def _metric_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)
