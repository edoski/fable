"""Pure canonical addresses for completed domain objects."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID


def corpus_directory(storage_root: Path, corpus_id: UUID) -> Path:
    return storage_root / "corpora" / str(corpus_id)


def corpus_json_path(storage_root: Path, corpus_id: UUID) -> Path:
    return corpus_directory(storage_root, corpus_id) / "corpus.json"


def corpus_blocks_path(storage_root: Path, corpus_id: UUID) -> Path:
    return corpus_directory(storage_root, corpus_id) / "blocks.parquet"


def study_json_path(storage_root: Path, study_id: UUID) -> Path:
    return storage_root / "studies" / f"{study_id}.json"


def artifact_checkpoint_path(storage_root: Path, artifact_id: UUID) -> Path:
    return storage_root / "artifacts" / f"{artifact_id}.ckpt"


def evaluation_directory(storage_root: Path, evaluation_id: UUID) -> Path:
    return storage_root / "evaluations" / str(evaluation_id)


def evaluation_json_path(storage_root: Path, evaluation_id: UUID) -> Path:
    return evaluation_directory(storage_root, evaluation_id) / "evaluation.json"


def evaluation_observations_path(storage_root: Path, evaluation_id: UUID) -> Path:
    return evaluation_directory(storage_root, evaluation_id) / "observations.parquet"
