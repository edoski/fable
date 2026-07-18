"""Storage-owned root transaction boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from .artifact import record_evaluation_state
from .catalog.index import ReindexedCatalogRoot, reindex_catalog_root
from .engine import (
    ARTIFACT_ROOT_KIND,
    STUDY_ROOT_KIND,
    RootKind,
    require_root_kind,
    state_db_path,
)
from .lifecycle import staged_root, validate_root_destination_path
from .workflow_roots import ArtifactRootHandle, StudyRootHandle

EffectT = TypeVar("EffectT")

if TYPE_CHECKING:
    from ..modeling.results import EvaluationRuntimeSummary, LoadedEvaluationSummary


@dataclass(frozen=True, slots=True)
class FullRootCommit(Generic[EffectT]):
    result: EffectT
    reindexed: ReindexedCatalogRoot


@dataclass(frozen=True, slots=True)
class RootMutation(Generic[EffectT]):
    result: EffectT
    reindexed: ReindexedCatalogRoot


def _reindex_root_state(
    storage_root: Path,
    *,
    root_path: Path,
    expected_root_kind: RootKind,
) -> ReindexedCatalogRoot:
    validate_root_destination_path(
        storage_root,
        destination_root=root_path,
        expected_root_kind=expected_root_kind,
    )
    require_root_kind(state_db_path(root_path), expected_root_kind)
    return reindex_catalog_root(storage_root, root_path=root_path)


def _record_mutated_root(
    storage_root: Path,
    *,
    root_path: Path,
    expected_root_kind: RootKind,
    mutation: Callable[[], EffectT],
) -> RootMutation[EffectT]:
    validate_root_destination_path(
        storage_root,
        destination_root=root_path,
        expected_root_kind=expected_root_kind,
    )
    result = mutation()
    reindexed = _reindex_root_state(
        storage_root,
        root_path=root_path,
        expected_root_kind=expected_root_kind,
    )
    return RootMutation(result=result, reindexed=reindexed)


def commit_artifact_root(
    artifact: ArtifactRootHandle,
    *,
    writer: Callable[[Path], EffectT],
) -> FullRootCommit[EffectT]:
    resumable_stage = artifact.root_path.parent / f".{artifact.root_path.name}.staging"
    with staged_root(
        storage_root=artifact.storage_root,
        destination_root=artifact.root_path,
        expected_root_kind=ARTIFACT_ROOT_KIND,
        replace=True,
        purpose="staging",
        prune_stop_at=artifact.root_path.parent.parent,
        staged_root_path_override=resumable_stage,
        reuse_existing=True,
        cleanup_on_failure=False,
    ) as stage:
        result = writer(stage.staged_root)
        reindexed = stage.promote()
    return FullRootCommit(result=result, reindexed=reindexed)


def record_study_root_mutation(
    study: StudyRootHandle,
    *,
    mutation: Callable[[], EffectT],
) -> RootMutation[EffectT]:
    return _record_mutated_root(
        study.storage_root,
        root_path=study.root_path,
        expected_root_kind=STUDY_ROOT_KIND,
        mutation=mutation,
    )


def record_artifact_evaluation_state(
    artifact: ArtifactRootHandle,
    *,
    summary: EvaluationRuntimeSummary,
) -> LoadedEvaluationSummary:
    validate_root_destination_path(
        artifact.storage_root,
        destination_root=artifact.root_path,
        expected_root_kind=ARTIFACT_ROOT_KIND,
    )
    require_root_kind(artifact.state_db_path, ARTIFACT_ROOT_KIND)
    return record_evaluation_state(artifact.state_db_path, summary=summary)
