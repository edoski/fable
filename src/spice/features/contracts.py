"""Compiled feature contracts shared across workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ..semantics import FeatureSemantics
from .core import (
    FeaturePrerequisites,
    ResolvedFeatureTable,
    build_feature_table,
    feature_graph_fingerprint,
    feature_optional_enrichments,
    feature_prerequisites,
    feature_source_columns,
)
from .registry import feature_entry

if TYPE_CHECKING:
    from ..config.models import FeaturesConfig


@dataclass(frozen=True, slots=True)
class CompiledFeatureContract:
    features_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    feature_prerequisites: FeaturePrerequisites
    required_source_columns: frozenset[str] = frozenset()
    optional_source_enrichments: frozenset[str] = frozenset()

    @property
    def semantics(self) -> FeatureSemantics:
        return FeatureSemantics(
            features_id=self.features_id,
            feature_names=self.feature_names,
            feature_graph_fingerprint=self.feature_graph_fingerprint,
            feature_prerequisites=self.feature_prerequisites,
        )

    def build_table(self, blocks: pl.DataFrame) -> ResolvedFeatureTable:
        catalog = feature_entry(self.features_id)
        return build_feature_table(
            blocks,
            features_id=self.features_id,
            catalog=catalog,
            feature_names=self.feature_names,
        )


def compile_feature_contract(*, features: FeaturesConfig) -> CompiledFeatureContract:
    catalog = feature_entry(features.id)
    feature_names = tuple(features.outputs)
    return CompiledFeatureContract(
        features_id=features.id,
        feature_names=feature_names,
        feature_graph_fingerprint=feature_graph_fingerprint(
            features.id,
            feature_names,
            fingerprint_sources=catalog.fingerprint_sources,
        ),
        feature_prerequisites=feature_prerequisites(catalog, feature_names),
        required_source_columns=feature_source_columns(catalog, feature_names),
        optional_source_enrichments=feature_optional_enrichments(catalog, feature_names),
    )
