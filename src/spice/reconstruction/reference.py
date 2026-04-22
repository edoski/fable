"""Reference data loaders for ICDCS temporal reconstruction."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import polars as pl

from .models import ReferenceMetricRow, ReferenceRawDataset

_CHAIN_DATASET_FILES = {
    "ethereum": "eth_block_data.csv",
    "polygon": "polygon_block_data.csv",
    "avalanche": "avalanche_block_data.csv",
}

_NOTEBOOK_METRIC_BLOCK = re.compile(
    r"Number of unique minBlock values in training set:\s*(?P<classes>\d+)",
)
_METRIC_LINE_PATTERNS = {
    "mae_block": re.compile(r"MAE minBlock \(raw\):\s*(?P<value>[-+0-9.eE]+)"),
    "mae_block_rounded": re.compile(
        r"MAE minBlock \(rounded\):\s*(?P<value>[-+0-9.eE]+)"
    ),
    "acc_block_rounded": re.compile(
        r"Accuracy minBlock \(round\):\s*(?P<value>[-+0-9.eE]+)"
    ),
    "mae_min_fee": re.compile(r"MAE minBaseFee:\s*(?P<value>[-+0-9.eE]+)"),
}


def canonical_reference_root(reference_root: Path) -> Path:
    return reference_root.expanduser().resolve()


def load_reference_predictions(reference_root: Path) -> list[ReferenceMetricRow]:
    resolved_root = canonical_reference_root(reference_root)
    predictions_path = resolved_root / "predictions.csv"
    frame = pl.read_csv(predictions_path)
    deduped_rows = frame.unique(maintain_order=True)
    rows = [
        ReferenceMetricRow(
            chain=str(row["chain"]),
            delay_seconds=int(row["seconds"]),
            model_type=str(row["type_model"]),
            mae_block=float(row["mae_block"]),
            mae_block_rounded=float(row["mae_block_rounded"]),
            acc_block_rounded=float(row["acc_block_rounded"]),
            mae_min_fee=float(row["mae_min_fee"]),
        )
        for row in deduped_rows.iter_rows(named=True)
    ]
    unique_class_counts: list[dict[str, float]] = []
    for notebook_name in ("plot.ipynb", "test_chain_model.ipynb", "testchain2.ipynb"):
        unique_class_counts.extend(_load_unique_class_counts(resolved_root / notebook_name))
    if unique_class_counts:
        return _attach_unique_class_counts(rows, unique_class_counts)
    return rows


def load_reference_raw_blocks(reference_root: Path, chain: str) -> pl.DataFrame:
    resolved_root = canonical_reference_root(reference_root)
    filename = _CHAIN_DATASET_FILES.get(chain)
    if filename is None:
        known = ", ".join(sorted(_CHAIN_DATASET_FILES))
        raise ValueError(f"Unsupported chain: {chain}. Known values: {known}")
    path = resolved_root / "dataset" / filename
    blocks = pl.read_csv(path)
    return (
        blocks.with_columns(
            pl.col("block_number").cast(pl.Int64),
            pl.col("gas_used").cast(pl.Int64),
            pl.col("gas_limit").cast(pl.Int64),
            pl.col("base_fee_per_gas").cast(pl.Float64),
            pl.col("price_usd").cast(pl.Float64),
            pl.col("block_usage_ratio").cast(pl.Float64),
            pl.col("base_fee_usd_per_gas").cast(pl.Float64),
            pl.col("block_time")
            .str.replace(" UTC", "")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f", strict=True)
            .dt.replace_time_zone("UTC")
            .dt.epoch(time_unit="s")
            .cast(pl.Int64)
            .alias("timestamp"),
        )
        .sort("timestamp")
        .unique(subset=["block_number"], keep="first", maintain_order=True)
        .sort("timestamp")
    )


def summarize_reference_raw_datasets(reference_root: Path) -> list[ReferenceRawDataset]:
    summaries: list[ReferenceRawDataset] = []
    for chain in sorted(_CHAIN_DATASET_FILES):
        blocks = load_reference_raw_blocks(reference_root, chain)
        timestamps = blocks["timestamp"].to_numpy()
        summaries.append(
            ReferenceRawDataset(
                chain=chain,
                csv_path=str(
                    canonical_reference_root(reference_root)
                    / "dataset"
                    / _CHAIN_DATASET_FILES[chain]
                ),
                row_count=int(blocks.height),
                first_timestamp=int(timestamps[0]),
                last_timestamp=int(timestamps[-1]),
            )
        )
    return summaries


def expected_unique_class_count(
    reference_metrics: list[ReferenceMetricRow],
    *,
    chain: str,
    delay_seconds: int,
) -> int | None:
    matches = [
        row.unique_min_block_classes
        for row in reference_metrics
        if (
            row.chain == chain
            and row.delay_seconds == delay_seconds
            and row.unique_min_block_classes is not None
        )
    ]
    if not matches:
        return None
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(
            "Reference unique class counts are inconsistent for "
            f"{chain} {delay_seconds}s: {unique}"
        )
    return unique[0]


def _load_unique_class_counts(notebook_path: Path) -> list[dict[str, float]]:
    if not notebook_path.is_file():
        return []
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            for chunk in _output_text_chunks(output):
                lines.extend(chunk.splitlines())
    matches: list[dict[str, float]] = []
    current_classes: float | None = None
    current_metrics: dict[str, float] = {}
    for line in lines:
        class_match = _NOTEBOOK_METRIC_BLOCK.search(line)
        if class_match is not None:
            current_classes = float(class_match.group("classes"))
            current_metrics = {}
            continue
        for key, pattern in _METRIC_LINE_PATTERNS.items():
            metric_match = pattern.search(line)
            if metric_match is None:
                continue
            current_metrics[key] = float(metric_match.group("value"))
            break
        if current_classes is None:
            continue
        if set(current_metrics) == set(_METRIC_LINE_PATTERNS):
            matches.append(
                {
                    "unique_min_block_classes": current_classes,
                    "mae_block": current_metrics["mae_block"],
                    "mae_block_rounded": current_metrics["mae_block_rounded"],
                    "acc_block_rounded": current_metrics["acc_block_rounded"],
                    "mae_min_fee": current_metrics["mae_min_fee"],
                }
            )
            current_metrics = {}
    return matches


def _output_text_chunks(output: dict[str, object]) -> list[str]:
    if isinstance(output.get("text"), list):
        return ["".join(output["text"])]
    if isinstance(output.get("text"), str):
        return [str(output["text"])]
    data = output.get("data")
    if isinstance(data, dict):
        text_plain = data.get("text/plain")
        if isinstance(text_plain, list):
            return ["".join(text_plain)]
        if isinstance(text_plain, str):
            return [text_plain]
    return []


def _attach_unique_class_counts(
    rows: list[ReferenceMetricRow],
    notebook_rows: list[dict[str, float]],
) -> list[ReferenceMetricRow]:
    assigned: list[ReferenceMetricRow] = []
    remaining_notebook_rows = list(notebook_rows)
    for row in rows:
        match_index = _matching_notebook_row_index(row, remaining_notebook_rows)
        if match_index is None:
            assigned.append(row)
            continue
        match = remaining_notebook_rows.pop(match_index)
        assigned.append(
            ReferenceMetricRow(
                chain=row.chain,
                delay_seconds=row.delay_seconds,
                model_type=row.model_type,
                mae_block=row.mae_block,
                mae_block_rounded=row.mae_block_rounded,
                acc_block_rounded=row.acc_block_rounded,
                mae_min_fee=row.mae_min_fee,
                unique_min_block_classes=int(match["unique_min_block_classes"]),
            )
        )
    return assigned


def _matching_notebook_row_index(
    row: ReferenceMetricRow,
    notebook_rows: list[dict[str, float]],
) -> int | None:
    for index, candidate in enumerate(notebook_rows):
        if not _approx_equal(row.mae_block, candidate["mae_block"]):
            continue
        if not _approx_equal(row.mae_block_rounded, candidate["mae_block_rounded"]):
            continue
        if not _approx_equal(row.acc_block_rounded, candidate["acc_block_rounded"]):
            continue
        if not _approx_equal(row.mae_min_fee, candidate["mae_min_fee"]):
            continue
        return index
    return None


def _approx_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-4, abs_tol=5e-4)
