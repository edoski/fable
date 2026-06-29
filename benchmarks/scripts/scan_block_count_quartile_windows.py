from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

EXPORT_DIR = Path("benchmarks/exports/evaluation_window_scans")
GWEI = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class ScanConfig:
    chain: str
    corpus_id: str
    training_cutoff: int
    output_prefix: str
    block_count: int
    stride_blocks: int
    samples_per_metric_quartile: int


def utc_timestamp(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).astimezone(UTC).timestamp())


def iso_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def config_from_env() -> ScanConfig:
    return ScanConfig(
        chain=os.environ["SPICE_SCAN_CHAIN"],
        corpus_id=os.environ["SPICE_SCAN_CORPUS_ID"],
        training_cutoff=utc_timestamp(os.environ["SPICE_SCAN_TRAINING_CUTOFF"]),
        output_prefix=os.environ["SPICE_SCAN_OUTPUT_PREFIX"],
        block_count=int(os.environ.get("SPICE_SCAN_BLOCK_COUNT", "1200")),
        stride_blocks=int(os.environ.get("SPICE_SCAN_STRIDE_BLOCKS", "1200")),
        samples_per_metric_quartile=int(
            os.environ.get("SPICE_SCAN_SAMPLES_PER_METRIC_QUARTILE", "27")
        ),
    )


def load_blocks(config: ScanConfig) -> pl.DataFrame:
    root = Path("outputs/corpora") / config.chain / config.corpus_id / "blocks"
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    blocks = (
        pl.scan_parquet([str(path) for path in files])
        .select(
            "block_number",
            "timestamp",
            "base_fee_per_gas",
            "gas_used",
            "gas_limit",
        )
        .filter(pl.col("timestamp") >= config.training_cutoff)
        .sort("block_number")
        .collect()
    )
    if blocks.is_empty():
        raise ValueError("training cutoff left no post-cutoff blocks")
    if blocks["block_number"].n_unique() != blocks.height:
        raise ValueError("block_count scan requires unique block_number rows")
    return blocks


def block_windows(config: ScanConfig, blocks: pl.DataFrame) -> list[dict[str, object]]:
    block_numbers = blocks["block_number"].cast(pl.Int64).to_numpy()
    timestamps = blocks["timestamp"].cast(pl.Int64).to_numpy()
    base_fee = blocks["base_fee_per_gas"].cast(pl.Float64).to_numpy() / GWEI
    gas_used = blocks["gas_used"].cast(pl.Float64).to_numpy()
    gas_limit = blocks["gas_limit"].cast(pl.Float64).to_numpy()
    gas_utilization = np.divide(
        gas_used,
        gas_limit,
        out=np.zeros_like(gas_used, dtype=np.float64),
        where=gas_limit > 0.0,
    )

    rows: list[dict[str, object]] = []
    last_start = int(blocks.height) - config.block_count
    for start_index in range(0, last_start + 1, config.stride_blocks):
        end_index = start_index + config.block_count
        start_block = int(block_numbers[start_index])
        last_block = int(block_numbers[end_index - 1])
        end_block_exclusive = last_block + 1
        if end_block_exclusive - start_block != config.block_count:
            continue

        fees = base_fee[start_index:end_index]
        if fees.shape[0] != config.block_count or np.any(fees <= 0.0):
            continue
        log_changes = np.diff(np.log(fees))
        volatility = float(np.std(log_changes, ddof=1))
        start_ts = int(timestamps[start_index])
        end_ts = int(timestamps[end_index - 1]) + 1
        row = {
            "chain": config.chain,
            "corpus_id": config.corpus_id,
            "window_id": f"{config.chain}_block{config.block_count}_{start_block}",
            "start_block": start_block,
            "end_block_exclusive": end_block_exclusive,
            "block_count": config.block_count,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "start_iso": iso_timestamp(start_ts),
            "end_iso": iso_timestamp(end_ts),
            "mean_base_fee_gwei": float(np.mean(fees)),
            "median_base_fee_gwei": float(np.median(fees)),
            "p10_base_fee_gwei": float(np.percentile(fees, 10)),
            "p90_base_fee_gwei": float(np.percentile(fees, 90)),
            "base_fee_volatility": volatility,
            "mean_gas_utilization": float(np.mean(gas_utilization[start_index:end_index])),
        }
        rows.append(row)
    if not rows:
        raise ValueError("block_count scan produced no candidate windows")
    return rows


def assign_quartiles(rows: list[dict[str, object]], metric: str, output_column: str) -> None:
    ordered = sorted(rows, key=lambda row: float(row[metric]))
    n_rows = len(ordered)
    for rank, row in enumerate(ordered):
        quartile_index = min(3, int(rank * 4 / n_rows))
        row[output_column] = f"q{quartile_index + 1}"
        row[f"{metric}_percentile"] = (rank + 0.5) / n_rows


def select_representative_windows(
    rows: list[dict[str, object]],
    *,
    metric: str,
    metric_label: str,
    quartile_column: str,
    samples_per_quartile: int,
    used_start_blocks: set[int],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for quartile in ("q1", "q2", "q3", "q4"):
        group = sorted(
            [row for row in rows if row[quartile_column] == quartile],
            key=lambda row: float(row[metric]),
        )
        if len(group) < samples_per_quartile:
            raise ValueError(
                f"not enough candidate windows for {metric_label} {quartile}: "
                f"{len(group)} < {samples_per_quartile}"
            )
        targets = np.linspace(
            0.5 / samples_per_quartile,
            1.0 - 0.5 / samples_per_quartile,
            samples_per_quartile,
        )
        for selection_index, target in enumerate(targets, start=1):
            target_index = int(round(float(target) * (len(group) - 1)))
            chosen = _nearest_unused(group, target_index, used_start_blocks)
            used_start_blocks.add(int(chosen["start_block"]))
            row = dict(chosen)
            row["source_window_id"] = row["window_id"]
            row["selection_metric"] = metric_label
            row["quartile"] = quartile
            row["selection_index"] = selection_index
            row["window_id"] = (
                f"{row['chain']}_block{row['block_count']}_"
                f"{metric_label}_{quartile}_{selection_index:03d}_"
                f"{row['start_block']}"
            )
            selected.append(row)
    return selected


def _nearest_unused(
    group: list[dict[str, object]],
    target_index: int,
    used_start_blocks: set[int],
) -> dict[str, object]:
    for radius in range(len(group)):
        for candidate_index in (target_index - radius, target_index + radius):
            if candidate_index < 0 or candidate_index >= len(group):
                continue
            candidate = group[candidate_index]
            if int(candidate["start_block"]) not in used_start_blocks:
                return candidate
    raise ValueError("could not select a non-overlapping representative window")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quartile_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for metric_label, metric, quartile_column in (
        ("fee_level", "median_base_fee_gwei", "fee_quartile"),
        ("volatility", "base_fee_volatility", "volatility_quartile"),
    ):
        for quartile in ("q1", "q2", "q3", "q4"):
            group = [row for row in rows if row[quartile_column] == quartile]
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary.append(
                {
                    "selection_metric": metric_label,
                    "quartile": quartile,
                    "candidate_windows": len(group),
                    "min_value": float(values.min()),
                    "median_value": float(np.median(values)),
                    "max_value": float(values.max()),
                }
            )
    return summary


def main() -> None:
    config = config_from_env()
    if config.block_count <= 0 or config.stride_blocks <= 0:
        raise ValueError("block count and stride must be positive")
    blocks = load_blocks(config)
    rows = block_windows(config, blocks)
    assign_quartiles(rows, "median_base_fee_gwei", "fee_quartile")
    assign_quartiles(rows, "base_fee_volatility", "volatility_quartile")

    used_start_blocks: set[int] = set()
    selected = []
    selected.extend(
        select_representative_windows(
            rows,
            metric="median_base_fee_gwei",
            metric_label="fee_level",
            quartile_column="fee_quartile",
            samples_per_quartile=config.samples_per_metric_quartile,
            used_start_blocks=used_start_blocks,
        )
    )
    selected.extend(
        select_representative_windows(
            rows,
            metric="base_fee_volatility",
            metric_label="volatility",
            quartile_column="volatility_quartile",
            samples_per_quartile=config.samples_per_metric_quartile,
            used_start_blocks=used_start_blocks,
        )
    )

    all_path = EXPORT_DIR / f"{config.output_prefix}_windows_all.csv"
    selected_path = EXPORT_DIR / f"{config.output_prefix}_windows_selected_source.csv"
    recommended_path = EXPORT_DIR / f"{config.output_prefix}_windows_recommended.csv"
    summary_path = EXPORT_DIR / f"{config.output_prefix}_window_quartile_summary.csv"
    write_csv(all_path, rows)
    write_csv(selected_path, selected)
    write_csv(recommended_path, selected)
    write_csv(summary_path, quartile_summary(rows))
    print(
        "wrote "
        f"candidates={len(rows)} selected={len(selected)} "
        f"all={all_path} recommended={recommended_path}"
    )


if __name__ == "__main__":
    main()
