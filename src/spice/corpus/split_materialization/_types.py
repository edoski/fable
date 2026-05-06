"""Corpus split materialization public and internal types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ...acquisition import BlockPullPlan
from ..validation import BlockDatasetValidationReport
from ._policy import CorpusSplitOutcome


@dataclass(slots=True)
class CorpusSplitMaterializationSpec:
    chain_name: str
    expected_chain_id: int
    chunk_size: int


@dataclass(slots=True)
class ExistingDatasetState:
    path: Path
    validation: BlockDatasetValidationReport
    file_count: int


class CorpusSplitKind(StrEnum):
    HISTORY = "history"
    EVALUATION = "evaluation"


@dataclass(slots=True)
class DatasetBuildResult:
    path: Path
    validation: BlockDatasetValidationReport
    file_count: int
    promote_dir: Path | None
    outcome: CorpusSplitOutcome


@dataclass(frozen=True, slots=True)
class CorpusSplitIntent:
    kind: CorpusSplitKind
    output_dir: Path
    working_dir: Path
    plan: BlockPullPlan


StatusCallback = Callable[[str], None]
ValidationCallback = Callable[[BlockDatasetValidationReport, Path], None]
