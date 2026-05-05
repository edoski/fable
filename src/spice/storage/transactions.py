"""Storage-owned root transaction boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..core.files import replace_paths_atomic
from .catalog.index import ReindexedCatalogRoot, reindex_catalog_root
from .engine import RootKind
from .lifecycle import (
    RootStage,
    promote_root_stage,
    staged_root,
)


@dataclass(frozen=True, slots=True)
class FullRootTransaction:
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


@dataclass(slots=True)
class PartialRootTransaction:
    storage_root: Path
    root_path: Path
    promotions: list[tuple[Path, Path]] = field(default_factory=list)

    def add(self, target: Path, source: Path | None) -> None:
        if source is not None:
            self.promotions.append((target, source))

    def commit(self) -> ReindexedCatalogRoot:
        if self.promotions:
            replace_paths_atomic(self.promotions, replace=True)
        return reindex_catalog_root(self.storage_root, root_path=self.root_path)


def promote_full_root_stage(
    *,
    storage_root: Path,
    destination_root: Path,
    staged_root_path: Path,
    expected_root_kind: RootKind,
    replace: bool,
) -> ReindexedCatalogRoot:
    return promote_root_stage(
        storage_root=storage_root,
        destination_root=destination_root,
        staged_root=staged_root_path,
        expected_root_kind=expected_root_kind,
        replace=replace,
    )


def reindex_root_state(storage_root: Path, *, root_path: Path) -> ReindexedCatalogRoot:
    return reindex_catalog_root(storage_root, root_path=root_path)
