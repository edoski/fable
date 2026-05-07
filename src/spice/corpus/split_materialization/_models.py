"""Corpus split materialization public and internal types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ...acquisition import BlockPullPlan, BlockRange
from ..validation import BlockDatasetValidationReport


class CorpusSplitOutcome(StrEnum):
    CREATED = "created"
    REUSED = "reused"
    EXTENDED = "extended"
    REBUILT = "rebuilt"


@dataclass(slots=True)
class CorpusSplitMaterializationSpec:
    chain_name: str
    expected_chain_id: int
    chunk_size: int
    required_columns: frozenset[str]


class CorpusSplitKind(StrEnum):
    HISTORY = "history"
    EVALUATION = "evaluation"


@dataclass(slots=True)
class CorpusSplitMaterializationResult:
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


@dataclass(frozen=True, slots=True)
class _SplitDatasetFacts:
    status: str
    first_block_number: int | None
    last_block_number: int | None


@dataclass(frozen=True, slots=True)
class _SplitDatasetCandidate:
    path: Path
    validation: BlockDatasetValidationReport
    facts: _SplitDatasetFacts
    file_count: int


@dataclass(frozen=True, slots=True)
class _SplitPullRange:
    label: str
    block_range: BlockRange


StatusCallback = Callable[[str], None]
ValidationCallback = Callable[[BlockDatasetValidationReport, Path], None]
