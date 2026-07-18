# pyright: strict

"""Corpus-root SQLite persistence."""

from __future__ import annotations

from pathlib import Path

from ..core.errors import MissingStateError
from ..corpus.metadata import CorpusManifest
from .corpus_codecs import DATASET_MANIFEST_CODEC
from .engine import (
    DATASET_ROOT_KIND,
    create_state_engine,
    require_root_kind,
)
from .payloads import SingletonPayloadStore
from .schema import dataset_manifest

_DATASET_MANIFEST_STORE = SingletonPayloadStore(
    table=dataset_manifest,
    codec=DATASET_MANIFEST_CODEC,
)
def load_corpus_manifest(db_path: Path) -> CorpusManifest:
    """Load the canonical corpus manifest that owns corpus provenance."""

    if not db_path.is_file():
        raise MissingStateError(f"Missing corpus manifest: {db_path}")
    require_root_kind(db_path, DATASET_ROOT_KIND)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _DATASET_MANIFEST_STORE.load(conn)
        if manifest is None:
            raise MissingStateError(f"Missing corpus manifest: {db_path}")
        return manifest
    finally:
        engine.dispose()
