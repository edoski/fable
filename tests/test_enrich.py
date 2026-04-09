import unittest

from spice_temporal.contracts import RawBlockRow
from spice_temporal.enrich import enrich_rows_with_gas_limit


class EnrichTestCase(unittest.TestCase):
    def test_enrich_rows_only_fetches_missing_blocks(self) -> None:
        calls: list[list[int]] = []

        def fake_fetch(block_numbers: list[int]) -> dict[int, int]:
            calls.append(block_numbers)
            return {block_number: 30_000_000 + block_number for block_number in block_numbers}

        rows: list[RawBlockRow] = [
            {
                "block_number": 1,
                "timestamp": 1,
                "base_fee_per_gas": 10,
                "gas_used": 20,
                "chain_id": 1,
                "gas_limit": "",
            },
            {
                "block_number": 2,
                "timestamp": 2,
                "base_fee_per_gas": 10,
                "gas_used": 20,
                "chain_id": 1,
                "gas_limit": 123,
            },
            {
                "block_number": 3,
                "timestamp": 3,
                "base_fee_per_gas": 10,
                "gas_used": 20,
                "chain_id": 1,
            },
        ]
        enriched = enrich_rows_with_gas_limit(
            rows,
            fetch_gas_limits=fake_fetch,
            batch_size=2,
            max_methods_per_second=1000.0,
        )
        self.assertEqual(calls, [[1, 3]])
        self.assertEqual(enriched[0]["gas_limit"], 30_000_001)
        self.assertEqual(enriched[1]["gas_limit"], 123)
        self.assertEqual(enriched[2]["gas_limit"], 30_000_003)


if __name__ == "__main__":
    unittest.main()
