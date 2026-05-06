# pyright: strict

"""Corpus-root SQLite persistence."""

from __future__ import annotations

from pathlib import Path

from ..core.errors import MissingStateError
from ..corpus.metadata import AcquireRunRecord, DatasetManifest
from .corpus_codecs import ACQUIRE_RUN_CODEC, DATASET_MANIFEST_CODEC
from .engine import (
    DATASET_ROOT_KIND,
    create_state_engine,
    ensure_state_db,
    require_root_kind,
    touch_meta,
)
from .payloads import SequencePayloadStore, SingletonPayloadStore
from .schema import DATASET_TABLES, acquire_runs, dataset_manifest

_DATASET_MANIFEST_STORE = SingletonPayloadStore(
    table=dataset_manifest,
    codec=DATASET_MANIFEST_CODEC,
)
_ACQUIRE_RUN_STORE = SequencePayloadStore(
    table=acquire_runs,
    codec=ACQUIRE_RUN_CODEC,
)


def write_dataset_state(
    db_path: Path,
    *,
    manifest: DatasetManifest,
    acquire_run: AcquireRunRecord,
) -> None:
    """Persist the dataset manifest once and append one acquire-run delta record."""

    ensure_state_db(db_path, root_kind=DATASET_ROOT_KIND, tables=DATASET_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            _DATASET_MANIFEST_STORE.upsert(conn, manifest)
            _ACQUIRE_RUN_STORE.append(conn, acquire_run, recorded_at=_now_timestamp())
            touch_meta(conn, root_kind=DATASET_ROOT_KIND)
    finally:
        engine.dispose()


def load_dataset_manifest(db_path: Path) -> DatasetManifest:
    """Load the canonical dataset manifest that owns corpus provenance."""

    if not db_path.is_file():
        raise MissingStateError(f"Missing dataset manifest: {db_path}")
    require_root_kind(db_path, DATASET_ROOT_KIND)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _DATASET_MANIFEST_STORE.load(conn)
        if manifest is None:
            raise MissingStateError(f"Missing dataset manifest: {db_path}")
        return manifest
    finally:
        engine.dispose()


def list_acquire_runs(db_path: Path) -> list[AcquireRunRecord]:
    """List acquire-run deltas newest first without duplicating manifest provenance."""

    if not db_path.is_file():
        return []
    require_root_kind(db_path, DATASET_ROOT_KIND)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            return _ACQUIRE_RUN_STORE.list(conn, order_by=acquire_runs.c.run_id.desc())
    finally:
        engine.dispose()


def _now_timestamp() -> int:
    from time import time

    return int(time())
