"""Block loading and validation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from spice_temporal.contracts import BlockRow, BlockScalar, EnrichedBlockRow, RawBlockRow
from spice_temporal.records import BlockRecord

SUPPORTED_BLOCK_FILE_SUFFIXES = {".json", ".csv", ".parquet"}
REQUIRED_BLOCK_COLUMNS = {
    "block_number",
    "timestamp",
    "base_fee_per_gas",
    "gas_used",
    "gas_limit",
    "chain_id",
}
MISSING_GAS_LIMIT_VALUES = {None, "", 0, "0"}


def _parse_required_block_scalar(row: Mapping[str, object], field: str) -> BlockScalar:
    if field not in row:
        raise ValueError(f"Missing required block column: {field}")
    value = row[field]
    if isinstance(value, bool):
        raise TypeError(f"Invalid value for block column {field}: {value!r}")
    if isinstance(value, int | str):
        return value
    raise TypeError(f"Invalid value for block column {field}: {value!r}")


def _parse_optional_block_scalar(row: Mapping[str, object], field: str) -> BlockScalar | None:
    if field not in row:
        return None
    value = row[field]
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"Invalid value for block column {field}: {value!r}")
    if isinstance(value, int | str):
        return value
    raise TypeError(f"Invalid value for block column {field}: {value!r}")


def parse_raw_block_row(row: object) -> RawBlockRow:
    if not isinstance(row, dict):
        raise TypeError("Block rows must be JSON-like mappings")
    raw_row: RawBlockRow = {
        "block_number": _parse_required_block_scalar(row, "block_number"),
        "timestamp": _parse_required_block_scalar(row, "timestamp"),
        "base_fee_per_gas": _parse_required_block_scalar(row, "base_fee_per_gas"),
        "gas_used": _parse_required_block_scalar(row, "gas_used"),
        "chain_id": _parse_required_block_scalar(row, "chain_id"),
    }
    gas_limit = _parse_optional_block_scalar(row, "gas_limit")
    if gas_limit is not None:
        raw_row["gas_limit"] = gas_limit
    return raw_row


def is_missing_gas_limit(value: object) -> bool:
    return value in MISSING_GAS_LIMIT_VALUES


def has_missing_gas_limit(row: RawBlockRow) -> bool:
    return is_missing_gas_limit(row.get("gas_limit"))


def build_enriched_block_row(row: RawBlockRow, *, gas_limit: BlockScalar) -> EnrichedBlockRow:
    return EnrichedBlockRow(
        block_number=row["block_number"],
        timestamp=row["timestamp"],
        base_fee_per_gas=row["base_fee_per_gas"],
        gas_used=row["gas_used"],
        gas_limit=gas_limit,
        chain_id=row["chain_id"],
    )


def parse_enriched_block_row(row: RawBlockRow) -> EnrichedBlockRow:
    gas_limit = row.get("gas_limit")
    if is_missing_gas_limit(gas_limit):
        raise ValueError("Block dataset must contain gas_limit for every row")
    return build_enriched_block_row(row, gas_limit=cast(BlockScalar, gas_limit))


def _is_hidden_relative_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def iter_block_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_BLOCK_FILE_SUFFIXES:
            raise ValueError(f"Unsupported block file format: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Block dataset path does not exist: {path}")

    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_BLOCK_FILE_SUFFIXES
        and not _is_hidden_relative_path(candidate.relative_to(path))
    ]
    if not files:
        raise ValueError(f"No supported block files found under {path}")
    return sorted(files)


def load_rows(path: Path) -> list[RawBlockRow]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError("JSON block files must contain a list of rows")
        return [parse_raw_block_row(row) for row in data]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [parse_raw_block_row(dict(row)) for row in reader]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading parquet files requires pyarrow. Install project dependencies first."
            ) from exc

        table = pq.read_table(path)
        return [parse_raw_block_row(row) for row in table.to_pylist()]
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


def parse_block_record(row: EnrichedBlockRow) -> BlockRecord:
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


def _validate_block_records(blocks: list[BlockRecord]) -> list[BlockRecord]:
    if not blocks:
        raise ValueError("Block dataset is empty")
    blocks = sorted(blocks, key=lambda block: block.block_number)
    chain_ids = {block.chain_id for block in blocks}
    if len(chain_ids) != 1:
        raise ValueError(
            f"Block dataset must contain exactly one chain_id, got {sorted(chain_ids)}"
        )
    for left, right in zip(blocks, blocks[1:], strict=False):
        if left.block_number == right.block_number:
            raise ValueError(f"Duplicate block_number detected: {left.block_number}")
    return blocks


def load_block_records(path: Path) -> list[BlockRecord]:
    rows: list[RawBlockRow] = []
    for file_path in iter_block_files(path):
        rows.extend(load_rows(file_path))
    blocks = [parse_block_record(parse_enriched_block_row(row)) for row in rows]
    return _validate_block_records(blocks)
