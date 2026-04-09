"""Block loading and validation."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from spice_temporal.contracts import BlockRow, EnrichedBlockRow, RawBlockRow
from spice_temporal.records import BlockRecord

REQUIRED_BLOCK_COLUMNS = {
    "block_number",
    "timestamp",
    "base_fee_per_gas",
    "gas_used",
    "gas_limit",
    "chain_id",
}


def load_rows(path: Path) -> list[RawBlockRow]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError("JSON block files must contain a list of rows")
        return cast(list[RawBlockRow], data)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return cast(list[RawBlockRow], [dict(row) for row in reader])
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading parquet files requires pyarrow. Install project dependencies first."
            ) from exc

        table = pq.read_table(path)
        return cast(list[RawBlockRow], table.to_pylist())
    raise ValueError(f"Unsupported block file format: {path.suffix}")


def write_rows(path: Path, rows: Sequence[BlockRow]) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=True, indent=2)
        return
    if suffix == ".csv":
        if not rows:
            raise ValueError("Cannot write an empty CSV without headers")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return
    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Writing parquet files requires pyarrow. Install project dependencies first."
            ) from exc

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return
    raise ValueError(f"Unsupported block file format: {path.suffix}")


def _coerce_block_row(row: EnrichedBlockRow) -> BlockRecord:
    missing = REQUIRED_BLOCK_COLUMNS - row.keys()
    if missing:
        raise ValueError(f"Missing required block columns: {sorted(missing)}")
    return BlockRecord(
        block_number=int(row["block_number"]),
        timestamp=int(row["timestamp"]),
        base_fee_per_gas=int(row["base_fee_per_gas"]),
        gas_used=int(row["gas_used"]),
        gas_limit=int(row["gas_limit"]),
        chain_id=int(row["chain_id"]),
    )


def load_block_records(path: Path) -> list[BlockRecord]:
    rows = load_rows(path)
    return [_coerce_block_row(cast(EnrichedBlockRow, row)) for row in rows]
