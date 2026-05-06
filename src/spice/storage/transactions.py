"""Storage-owned root transaction boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from ..core.files import replace_paths_atomic
from .artifact import record_evaluation_state
from .catalog.index import ReindexedCatalogRoot, reindex_catalog_root
from .engine import (
    ARTIFACT_ROOT_KIND,
    DATASET_ROOT_KIND,
    STUDY_ROOT_KIND,
    RootKind,
    require_root_kind,
    state_db_path,
)
from .lifecycle import RootStage, staged_root, validate_root_destination_path
from .workflow_roots import ArtifactRootHandle, CorpusRootHandle, StudyRootHandle

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


@dataclass(frozen=True, slots=True)
class _FullRootTransaction:
    storage_root: Path
    destination_root: Path
    expected_root_kind: RootKind
    replace: bool = True
    purpose: str = "staging"
    prune_stop_at: Path | None = None

    @contextmanager
    def open(self) -> Iterator[RootStage]:
        with staged_root(
            storage_root=self.storage_root,
            destination_root=self.destination_root,
            expected_root_kind=self.expected_root_kind,
            replace=self.replace,
            purpose=self.purpose,
            prune_stop_at=self.prune_stop_at,
        ) as stage:
            yield stage

    def commit(self, writer: Callable[[Path], EffectT]) -> FullRootCommit[EffectT]:
        with self.open() as stage:
            result = writer(stage.staged_root)
            reindexed = stage.promote()
        return FullRootCommit(result=result, reindexed=reindexed)


@dataclass(slots=True)
class _CorpusRootTransaction:
    corpus: CorpusRootHandle
    promotions: list[tuple[Path, Path]] = field(default_factory=list)

    def replace_history(self, source: Path | None) -> None:
        self._add(self.corpus.history_dir, source)

    def replace_evaluation(self, source: Path | None) -> None:
        self._add(self.corpus.evaluation_dir, source)

    def replace_state_db(self, source: Path | None) -> None:
        self._add(self.corpus.state_db_path, source)

    def _add(self, target: Path, source: Path | None) -> None:
        if source is not None:
            self.promotions.append((target, source))

    def commit(self) -> ReindexedCatalogRoot:
        validate_root_destination_path(
            self.corpus.storage_root,
            destination_root=self.corpus.root_path,
            expected_root_kind=DATASET_ROOT_KIND,
        )
        if self.promotions:
            replace_paths_atomic(self.promotions, replace=True)
        return reindex_catalog_root(self.corpus.storage_root, root_path=self.corpus.root_path)


def commit_corpus_acquisition(
    corpus: CorpusRootHandle,
    *,
    history_dir: Path | None,
    evaluation_dir: Path | None,
    state_db: Path,
) -> ReindexedCatalogRoot:
    require_root_kind(state_db, DATASET_ROOT_KIND)
    transaction = _CorpusRootTransaction(corpus=corpus)
    transaction.replace_history(history_dir)
    transaction.replace_evaluation(evaluation_dir)
    transaction.replace_state_db(state_db)
    return transaction.commit()


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


def _artifact_full_root_transaction(
    artifact: ArtifactRootHandle,
    *,
    replace: bool = True,
    purpose: str = "staging",
) -> _FullRootTransaction:
    return _FullRootTransaction(
        storage_root=artifact.storage_root,
        destination_root=artifact.root_path,
        expected_root_kind=ARTIFACT_ROOT_KIND,
        replace=replace,
        purpose=purpose,
        prune_stop_at=artifact.root_path.parent.parent,
    )


def commit_artifact_root(
    artifact: ArtifactRootHandle,
    *,
    writer: Callable[[Path], EffectT],
) -> FullRootCommit[EffectT]:
    return _artifact_full_root_transaction(artifact).commit(writer)


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
