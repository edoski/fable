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


def study_directory(storage_root: Path, study_id: UUID) -> Path:
    return storage_root / "studies" / str(study_id)


def study_json_path(storage_root: Path, study_id: UUID) -> Path:
    return study_directory(storage_root, study_id) / "study.json"


def study_trial_directory(storage_root: Path, study_id: UUID, method_index: int) -> Path:
    return study_directory(storage_root, study_id) / "trials" / str(method_index)


def study_trial_checkpoint_path(storage_root: Path, study_id: UUID, method_index: int) -> Path:
    return study_trial_directory(storage_root, study_id, method_index) / "selected.ckpt"


def study_trial_observations_path(storage_root: Path, study_id: UUID, method_index: int) -> Path:
    return study_trial_directory(storage_root, study_id, method_index) / "validation.parquet"


def artifact_directory(storage_root: Path, artifact_id: UUID) -> Path:
    return storage_root / "artifacts" / str(artifact_id)


def artifact_checkpoint_path(storage_root: Path, artifact_id: UUID) -> Path:
    return artifact_directory(storage_root, artifact_id) / "artifact.ckpt"


def artifact_observations_path(storage_root: Path, artifact_id: UUID) -> Path:
    return artifact_directory(storage_root, artifact_id) / "validation.parquet"


def evaluation_directory(storage_root: Path, evaluation_id: UUID) -> Path:
    return storage_root / "evaluations" / str(evaluation_id)


def evaluation_json_path(storage_root: Path, evaluation_id: UUID) -> Path:
    return evaluation_directory(storage_root, evaluation_id) / "evaluation.json"


def evaluation_observations_path(storage_root: Path, evaluation_id: UUID) -> Path:
    return evaluation_directory(storage_root, evaluation_id) / "observations.parquet"
