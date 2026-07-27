"""Direct canonical Corpus IO."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import UUID4

from ..addresses import corpus_blocks_path, corpus_json_path
from ..config import CorpusRequest
from ..records import StrictFrozenRecord
from .blocks import BlockFrame
from .contract import Corpus, FinalizedAnchor


class _CorpusDocument(StrictFrozenRecord):
    request: CorpusRequest
    finalized_anchor: FinalizedAnchor


def _load_corpus_document(storage_root: Path, corpus_id: UUID4) -> _CorpusDocument:
    document = _CorpusDocument.model_validate_json(
        corpus_json_path(storage_root, corpus_id).read_text(encoding="utf-8"),
        strict=True,
    )
    if document.request.corpus_id != corpus_id:
        raise ValueError("Corpus request UUID does not match the requested corpus")
    return document


def load_corpus_request(storage_root: Path, corpus_id: UUID4) -> CorpusRequest:
    return _load_corpus_document(storage_root, corpus_id).request


def load_corpus(storage_root: Path, corpus_id: UUID4) -> Corpus:
    document = _load_corpus_document(storage_root, corpus_id)
    return Corpus(
        request=document.request,
        finalized_anchor=document.finalized_anchor,
        blocks=BlockFrame(
            pl.read_parquet(corpus_blocks_path(storage_root, corpus_id)),
            document.request.definition,
        ),
    )
