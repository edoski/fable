"""Compiled feature contracts shared across workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from .engine import (
    FeatureSelection,
    ResolvedFeatureTable,
    build_feature_table,
)
from .families import FeaturePrerequisites, feature_family_spec

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
    def selection(self) -> FeatureSelection:
        return FeatureSelection(
            feature_set_id=self.feature_set_id,
            feature_family_id=self.feature_family_id,
            feature_names=self.feature_names,
        )

    def build_table(self, blocks: pl.DataFrame) -> ResolvedFeatureTable:
        return build_feature_table(blocks, selection=self.selection)


def compile_feature_contract(*, feature_set: FeatureSetConfig) -> CompiledFeatureContract:
    family_spec = feature_family_spec(feature_set.family.id)
    return family_spec.compile_contract(
        feature_set.id,
        feature_set.family,
        tuple(feature_set.outputs),
    )
