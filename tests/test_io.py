import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from spice_temporal.io import (
    has_missing_gas_limit,
    load_block_records,
    load_rows,
    parse_enriched_block_row,
    write_rows,
)
from spice_temporal.records import BlockRecord


def make_block(index: int, *, chain_id: int = 1) -> BlockRecord:
    return BlockRecord(
        block_number=index,
        timestamp=1_700_000_000 + 12 * index,
        base_fee_per_gas=100 + (index % 5),
        gas_used=15_000_000 + index,
        gas_limit=30_000_000,
        chain_id=chain_id,
    )


class BlockIoTestCase(unittest.TestCase):
    def test_load_rows_validates_supported_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            rows = [
                {
                    "block_number": 1,
                    "timestamp": 1_700_000_012,
                    "base_fee_per_gas": 100,
                    "gas_used": 15_000_001,
                    "chain_id": 1,
                }
            ]
            for suffix in (".json", ".csv", ".parquet"):
                path = dataset_dir / f"blocks{suffix}"
                write_rows(path, rows)
                loaded = load_rows(path)
                self.assertEqual(str(loaded[0]["block_number"]), "1")
                self.assertNotIn("gas_limit", loaded[0])

    def test_load_block_records_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "blocks.json"
            path.write_text(
                json.dumps([asdict(make_block(index)) for index in range(3)]),
                encoding="utf-8",
            )
            blocks = load_block_records(path)
        self.assertEqual([block.block_number for block in blocks], [0, 1, 2])

    def test_load_block_records_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            (dataset_dir / "part-1.json").write_text(
                json.dumps([asdict(make_block(index)) for index in range(3)]),
                encoding="utf-8",
            )
            nested = dataset_dir / "nested"
            nested.mkdir()
            (nested / "part-2.json").write_text(
                json.dumps([asdict(make_block(index)) for index in range(3, 6)]),
                encoding="utf-8",
            )
            hidden = dataset_dir / ".cryo" / "reports"
            hidden.mkdir(parents=True)
            (hidden / "report.json").write_text('{"ignored": true}', encoding="utf-8")
            blocks = load_block_records(dataset_dir)
        self.assertEqual([block.block_number for block in blocks], [0, 1, 2, 3, 4, 5])

    def test_load_block_records_rejects_duplicate_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            (dataset_dir / "part-1.json").write_text(
                json.dumps([asdict(make_block(1)), asdict(make_block(2))]),
                encoding="utf-8",
            )
            (dataset_dir / "part-2.json").write_text(
                json.dumps([asdict(make_block(2)), asdict(make_block(3))]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate block_number"):
                load_block_records(dataset_dir)

    def test_load_block_records_rejects_mixed_chain_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            (dataset_dir / "part-1.json").write_text(
                json.dumps([asdict(make_block(1, chain_id=1))]),
                encoding="utf-8",
            )
            (dataset_dir / "part-2.json").write_text(
                json.dumps([asdict(make_block(2, chain_id=137))]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one chain_id"):
                load_block_records(dataset_dir)

    def test_load_block_records_requires_gas_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "blocks.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "block_number": 1,
                            "timestamp": 1_700_000_012,
                            "base_fee_per_gas": 100,
                            "gas_used": 15_000_001,
                            "chain_id": 1,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must contain gas_limit"):
                load_block_records(path)

    def test_load_rows_rejects_invalid_row_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "blocks.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "Block rows must be JSON-like mappings"):
                load_rows(path)

    def test_missing_gas_limit_values_use_one_validation_rule(self) -> None:
        for missing_value in (None, "", 0, "0"):
            row = {
                "block_number": 1,
                "timestamp": 1_700_000_012,
                "base_fee_per_gas": 100,
                "gas_used": 15_000_001,
                "chain_id": 1,
                "gas_limit": missing_value,
            }
            self.assertTrue(has_missing_gas_limit(row))
            with self.assertRaisesRegex(ValueError, "must contain gas_limit"):
                parse_enriched_block_row(row)


if __name__ == "__main__":
    unittest.main()
