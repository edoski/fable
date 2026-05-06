"""Corpus split parquet chunk mechanics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ...acquisition import BlockPullPlan, BlockRange
from ..contract import CanonicalBlockRow, canonicalize_block_frame
from ..io import load_block_frame, write_block_file
from ..validation import validate_contiguous_block_frame
from ._types import CorpusSplitMaterializationSpec


@dataclass(slots=True)
class ParquetBlockPullSink:
    output_dir: Path
    materialization: CorpusSplitMaterializationSpec
    pending_rows: list[CanonicalBlockRow]

    @classmethod
    def create(
        cls,
        output_dir: Path,
        *,
        materialization: CorpusSplitMaterializationSpec,
    ) -> ParquetBlockPullSink:
        return cls(output_dir=output_dir, materialization=materialization, pending_rows=[])

    def completed_prefix_end(self, plan: BlockPullPlan) -> int:
        return completed_prefix_end(
            self.output_dir,
            plan=plan,
            expected_chain_id=self.materialization.expected_chain_id,
            required_columns=self.materialization.required_columns,
        )

    def write_rows(self, rows: list[CanonicalBlockRow]) -> None:
        self.pending_rows.extend(rows)
        while len(self.pending_rows) >= self.materialization.chunk_size:
            write_block_rows_chunk(
                self.output_dir,
                chain_name=self.materialization.chain_name,
                rows=self.pending_rows[: self.materialization.chunk_size],
            )
            self.pending_rows = self.pending_rows[self.materialization.chunk_size :]

    def finish(self) -> None:
        if self.pending_rows:
            write_block_rows_chunk(
                self.output_dir,
                chain_name=self.materialization.chain_name,
                rows=self.pending_rows,
            )
            self.pending_rows = []


def filter_block_range(frame: pl.DataFrame, block_range: BlockRange) -> pl.DataFrame:
    return frame.filter(
        (pl.col("block_number") >= block_range.start)
        & (pl.col("block_number") < block_range.end)
    ).sort("block_number")


def write_block_dataset_dir(
    output_dir: Path,
    *,
    frame: pl.DataFrame,
    chunk_size: int,
    chain_name: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_frame = frame.sort("block_number")
    file_count = 0
    for start_index in range(0, sorted_frame.height, chunk_size):
        chunk = sorted_frame.slice(start_index, min(chunk_size, sorted_frame.height - start_index))
        start_block = int(chunk["block_number"][0])
        end_block = int(chunk["block_number"][-1])
        write_block_file(
            output_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet",
            chunk,
        )
        file_count += 1
    return file_count


def completed_prefix_end(
    output_dir: Path,
    *,
    plan: BlockPullPlan,
    expected_chain_id: int,
    required_columns: frozenset[str],
) -> int:
    if not output_dir.exists():
        return plan.block_range.start
    try:
        frame = load_block_frame(output_dir)
    except ValueError as exc:
        if "No parquet block files found" in str(exc):
            return plan.block_range.start
        raise RuntimeError(
            f"Cannot resume from invalid partial block dataset: {output_dir}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resume from invalid partial block dataset: {output_dir}"
        ) from exc

    validation = validate_contiguous_block_frame(
        frame,
        dataset_path=output_dir,
        expected_chain_id=expected_chain_id,
        required_columns=required_columns,
    )
    if validation.status != "clean":
        raise RuntimeError(f"Cannot resume from invalid partial block dataset: {validation}")
    if validation.first_block_number != plan.block_range.start:
        raise RuntimeError(
            "Cannot resume partial block dataset with a different start block: "
            f"expected {plan.block_range.start}, got {validation.first_block_number}"
        )
    if validation.last_block_number is None:
        return plan.block_range.start
    if validation.last_block_number >= plan.block_range.end:
        return plan.block_range.end
    return validation.last_block_number + 1


def write_block_rows_chunk(
    output_dir: Path,
    *,
    chain_name: str,
    rows: Sequence[CanonicalBlockRow],
) -> Path:
    frame = canonicalize_block_frame(pl.DataFrame(rows))
    start_block = int(frame["block_number"][0])
    end_block = int(frame["block_number"][-1])
    destination = output_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet"
    write_block_file(destination, frame)
    return destination


def combined_frame(*frames: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([frame for frame in frames if frame.height > 0], how="vertical").sort(
        "block_number"
    )
