"""Corpus split acquisition pull effects."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ...acquisition import (
    AcquisitionPullController,
    BlockPullPlan,
    BlockSource,
    TimestampRange,
    pull_block_range,
)
from ..io import load_block_frame
from ._chunks import ParquetBlockPullSink
from ._models import CorpusSplitMaterializationSpec, _SplitPullRange


async def pull_plan_to_frame(
    *,
    block_source: BlockSource,
    plan: BlockPullPlan,
    output_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    controller: AcquisitionPullController,
) -> pl.DataFrame:
    await pull_block_range(
        block_source,
        plan=plan,
        controller=controller,
        sink=ParquetBlockPullSink.create(output_dir, materialization=materialization),
    )
    return load_block_frame(output_dir)


async def pull_plan_to_dir(
    *,
    block_source: BlockSource,
    plan: BlockPullPlan,
    output_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    controller: AcquisitionPullController,
) -> Path:
    await pull_block_range(
        block_source,
        plan=plan,
        controller=controller,
        sink=ParquetBlockPullSink.create(output_dir, materialization=materialization),
    )
    return output_dir


def plan_pull_dir(working_dir: Path, *, label: str, plan: BlockPullPlan) -> Path:
    return (
        working_dir
        / "pulls"
        / f"{label}__{plan.block_range.start}_to_{plan.block_range.end}"
    )


def pull_range_plan(
    block_source: BlockSource,
    pull_range: _SplitPullRange,
    *,
    window: TimestampRange,
) -> BlockPullPlan:
    return block_source.plan_block_range(pull_range.block_range, window=window)


async def pull_plan_range_to_dir(
    *,
    block_source: BlockSource,
    pull_range: _SplitPullRange,
    window: TimestampRange,
    working_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    controller: AcquisitionPullController,
) -> Path:
    plan = pull_range_plan(block_source, pull_range, window=window)
    return await pull_plan_to_dir(
        block_source=block_source,
        plan=plan,
        output_dir=plan_pull_dir(working_dir, label=pull_range.label, plan=plan),
        materialization=materialization,
        controller=controller,
    )
