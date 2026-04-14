"""Shared feature-family types and helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from hamilton.graph_types import HamiltonNode

    from ..contracts import CompiledFeatureContract


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


class FeatureConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FeatureFamilyConfig(FeatureConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="feature_set.family.id")


class FeaturePrerequisites(FeatureConfigModel):
    history_seconds: int = Field(default=0, ge=0)
    warmup_rows: int = Field(default=0, ge=0)


FEATURE_KIND_TAG = "spice_kind"
FEATURE_KIND_VALUE = "feature"
FEATURE_HISTORY_SECONDS_TAG = "spice_history_seconds"
FEATURE_WARMUP_ROWS_TAG = "spice_warmup_rows"


def _tagged_int(raw_value: str | list[str] | None) -> int:
    if raw_value is None:
        return 0
    return int(raw_value[0] if isinstance(raw_value, list) else raw_value)


def tagged_feature_prerequisites(
    feature_names: tuple[str, ...],
    node_map: dict[str, HamiltonNode],
) -> FeaturePrerequisites:
    return FeaturePrerequisites(
        history_seconds=max(
            _tagged_int(node_map[name].tags.get(FEATURE_HISTORY_SECONDS_TAG))
            for name in feature_names
        ),
        warmup_rows=max(
            _tagged_int(node_map[name].tags.get(FEATURE_WARMUP_ROWS_TAG)) for name in feature_names
        ),
    )


FeatureFamilyConfigT = TypeVar("FeatureFamilyConfigT", bound=FeatureFamilyConfig)


@dataclass(frozen=True, slots=True)
class FeatureFamilySpec(Generic[FeatureFamilyConfigT]):
    id: str
    config_type: type[FeatureFamilyConfigT]
    modules: tuple[object, ...]
    resolve_prerequisites: Callable[
        [tuple[str, ...], dict[str, HamiltonNode]],
        FeaturePrerequisites,
    ]

    def compile_contract(
        self,
        feature_set_id: str,
        family_config: FeatureFamilyConfigT,
        feature_names: tuple[str, ...],
    ) -> CompiledFeatureContract:
        from ..contracts import CompiledFeatureContract
        from ..engine import feature_graph_fingerprint, feature_node_map, make_feature_selection

        if family_config.id != self.id:
            raise ValueError(
                f"Feature family config {family_config.id} does not match spec {self.id}"
            )
        selection = make_feature_selection(
            feature_set_id=feature_set_id,
            feature_family_id=self.id,
            feature_names=feature_names,
        )
        return CompiledFeatureContract(
            feature_set_id=selection.feature_set_id,
            feature_family_id=selection.feature_family_id,
            feature_names=selection.feature_names,
            feature_graph_fingerprint=feature_graph_fingerprint(
                selection.feature_family_id,
                selection.feature_names,
            ),
            feature_prerequisites=self.resolve_prerequisites(
                selection.feature_names,
                feature_node_map(self.id),
            ),
        )
