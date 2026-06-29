from __future__ import annotations

from benchmarks.scripts.write_evaluation_suite_from_window_csv import render_unique_suite


def test_block_window_suite_writer_omits_empty_tags() -> None:
    rendered = render_unique_suite(
        "block_suite",
        [
            {
                "window_id": "ethereum_block1200_fee_q1_001_123",
                "start_block": "123",
                "block_count": "1200",
            }
        ],
    )

    assert rendered == (
        "id: block_suite\n"
        "items:\n"
        "  - id: ethereum_block1200_fee_q1_001_123\n"
        "    start_block: 123\n"
        "    block_count: 1200\n"
    )
