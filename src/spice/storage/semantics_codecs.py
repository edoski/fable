"""Storage payload codecs for persisted semantic contracts."""

from __future__ import annotations

from pydantic import TypeAdapter

from ..core.errors import StateLayoutError
from ..features import FeaturePrerequisites
from ..metrics import MetricDescriptor
from ..semantics import ArtifactSemantics, StudySemantics
from .payloads import type_adapter_payload, type_adapter_value

_STUDY_SEMANTICS_ADAPTER = TypeAdapter(StudySemantics)
_ARTIFACT_SEMANTICS_ADAPTER = TypeAdapter(ArtifactSemantics)

_ADAPTER_NAMESPACE = {
    "FeaturePrerequisites": FeaturePrerequisites,
    "MetricDescriptor": MetricDescriptor,
}
for _adapter in (
    _STUDY_SEMANTICS_ADAPTER,
    _ARTIFACT_SEMANTICS_ADAPTER,
):
    _adapter.rebuild(_types_namespace=_ADAPTER_NAMESPACE)


def artifact_semantics_payload(semantics: ArtifactSemantics) -> dict[str, object]:
    return type_adapter_payload(
        _ARTIFACT_SEMANTICS_ADAPTER,
        semantics,
        label="semantics payload",
    )


def artifact_semantics_from_payload(payload: dict[str, object]) -> ArtifactSemantics:
    semantics = type_adapter_value(
        _ARTIFACT_SEMANTICS_ADAPTER,
        payload,
        label="semantics payload",
    )
    if payload != artifact_semantics_payload(semantics):
        raise StateLayoutError("Invalid semantics payload payload: payload is not canonical JSON")
    return semantics


def study_semantics_payload(semantics: StudySemantics) -> dict[str, object]:
    return type_adapter_payload(
        _STUDY_SEMANTICS_ADAPTER,
        semantics,
        label="semantics payload",
    )


def study_semantics_from_payload(payload: dict[str, object]) -> StudySemantics:
    semantics = type_adapter_value(
        _STUDY_SEMANTICS_ADAPTER,
        payload,
        label="semantics payload",
    )
    if payload != study_semantics_payload(semantics):
        raise StateLayoutError("Invalid semantics payload payload: payload is not canonical JSON")
    return semantics
