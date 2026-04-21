"""Compiled feature contracts shared across workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ..semantics import FeatureSemantics
from .core import (
    ResolvedFeatureTable,
    build_feature_table,
    feature_graph_fingerprint,
    feature_prerequisites,
)
from .families.base import FeaturePrerequisites
from .registry import feature_family

if TYPE_CHECKING:
    from ..config.models import FeatureSetConfig


@dataclass(frozen=True, slots=True)
class CompiledFeatureContract:
    feature_set_id: str
    feature_family_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    feature_prerequisites: FeaturePrerequisites

    @property
    def semantics(self) -> FeatureSemantics:
        return FeatureSemantics(
            feature_set_id=self.feature_set_id,
            feature_family_id=self.feature_family_id,
            feature_names=self.feature_names,
            feature_graph_fingerprint=self.feature_graph_fingerprint,
            feature_prerequisites=self.feature_prerequisites,
        )

    def build_table(self, blocks: pl.DataFrame) -> ResolvedFeatureTable:
        family = feature_family(self.feature_family_id)
        return build_feature_table(
            blocks,
            feature_set_id=self.feature_set_id,
            feature_family_id=self.feature_family_id,
            family=family,
            feature_names=self.feature_names,
        )


def compile_feature_contract(*, feature_set: FeatureSetConfig) -> CompiledFeatureContract:
    family = feature_family(feature_set.family.id)
    feature_names = tuple(feature_set.outputs)
    return CompiledFeatureContract(
        feature_set_id=feature_set.id,
        feature_family_id=feature_set.family.id,
        feature_names=feature_names,
        feature_graph_fingerprint=feature_graph_fingerprint(
            feature_set.family.id,
            feature_names,
            fingerprint_sources=family.fingerprint_sources,
        ),
        feature_prerequisites=feature_prerequisites(family, feature_names),
    )
