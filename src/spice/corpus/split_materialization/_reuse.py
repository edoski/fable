"""Corpus split reuse and materialization IO."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from shutil import copy2

import polars as pl

from ...acquisition import BlockRange
from ...core.files import remove_path
from ..io import iter_block_files, load_block_frame
from ._chunks import combined_frame, filter_block_range, write_block_dataset_dir
from ._dataset_state import validate_block_dataset
from ._policy import CorpusSplitOutcome
from ._types import (
    CorpusSplitMaterializationSpec,
    DatasetBuildResult,
    ValidationCallback,
)


def materialize_dataset(
    *,
    mode: str,
    materialization: CorpusSplitMaterializationSpec,
    working_dir: Path,
    validate_result: ValidationCallback,
    frames: Sequence[pl.DataFrame] | None = None,
    outcome: CorpusSplitOutcome,
) -> DatasetBuildResult:
    dataset_dir = working_dir / mode
    if frames is not None:
        remove_path(dataset_dir)
        file_count = write_block_dataset_dir(
            dataset_dir,
            frame=combined_frame(*frames),
            chunk_size=materialization.chunk_size,
            chain_name=materialization.chain_name,
        )
    else:
        file_count = len(iter_block_files(dataset_dir))
    validation = validate_block_dataset(
        dataset_dir,
        expected_chain_id=materialization.expected_chain_id,
    )
    validate_result(validation, dataset_dir)
    return DatasetBuildResult(
        path=dataset_dir,
        validation=validation,
        file_count=file_count,
        promote_dir=dataset_dir,
        outcome=outcome,
    )


def materialize_dataset_from_sources(
    *,
    mode: str,
    materialization: CorpusSplitMaterializationSpec,
    working_dir: Path,
    validate_result: ValidationCallback,
    source_dirs: Sequence[Path] = (),
    source_files: Sequence[Path] = (),
    frames: Sequence[pl.DataFrame] = (),
    outcome: CorpusSplitOutcome,
) -> DatasetBuildResult:
    dataset_dir = working_dir / mode
    remove_path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        for source_file in iter_block_files(source_dir):
            copy2(source_file, dataset_dir / source_file.name)
    for source_file in source_files:
        copy2(source_file, dataset_dir / source_file.name)
    for frame in frames:
        if frame.height > 0:
            write_block_dataset_dir(
                dataset_dir,
                frame=frame,
                chunk_size=materialization.chunk_size,
                chain_name=materialization.chain_name,
            )
    validation = validate_block_dataset(
        dataset_dir,
        expected_chain_id=materialization.expected_chain_id,
    )
    validate_result(validation, dataset_dir)
    return DatasetBuildResult(
        path=dataset_dir,
        validation=validation,
        file_count=len(iter_block_files(dataset_dir)),
        promote_dir=dataset_dir,
        outcome=outcome,
    )


def reusable_block_files_and_edges(
    dataset_dir: Path,
    *,
    block_range: BlockRange,
) -> tuple[list[Path], list[pl.DataFrame]]:
    reusable_files: list[Path] = []
    edge_frames: list[pl.DataFrame] = []
    for block_file in iter_block_files(dataset_dir):
        frame = load_block_frame(block_file)
        start_block = int(frame["block_number"][0])
        end_block = int(frame["block_number"][-1]) + 1
        if end_block <= block_range.start or start_block >= block_range.end:
            continue
        if start_block >= block_range.start and end_block <= block_range.end:
            reusable_files.append(block_file)
            continue
        edge = filter_block_range(frame, block_range)
        if edge.height > 0:
            edge_frames.append(edge)
    return reusable_files, edge_frames
