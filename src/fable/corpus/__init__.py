"""Canonical Corpus loading."""

from .contract import Corpus, FinalizedAnchor
from .io import load_corpus

__all__ = ["Corpus", "FinalizedAnchor", "load_corpus"]
