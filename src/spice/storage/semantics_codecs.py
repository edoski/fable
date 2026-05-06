"""Storage payload codecs for persisted semantic contracts."""

from __future__ import annotations

from pydantic import TypeAdapter

from ..core.errors import StateLayoutError
from ..features import FeaturePrerequisites
from ..metrics import MetricDescriptor
from ..semantics import ArtifactSemantics, StudySemantics, TemporalCapabilitySemantics
from .payloads import PayloadCodec, type_adapter_payload, type_adapter_value

_STUDY_SEMANTICS_ADAPTER = TypeAdapter(StudySemantics)
_ARTIFACT_SEMANTICS_ADAPTER = TypeAdapter(ArtifactSemantics)

_ADAPTER_NAMESPACE = {
    "FeaturePrerequisites": FeaturePrerequisites,
    "MetricDescriptor": MetricDescriptor,
    "TemporalCapabilitySemantics": TemporalCapabilitySemantics,
}
for _adapter in (
    _STUDY_SEMANTICS_ADAPTER,
    _ARTIFACT_SEMANTICS_ADAPTER,
):
    _adapter.rebuild(_types_namespace=_ADAPTER_NAMESPACE)


def _encode_artifact_semantics(semantics: ArtifactSemantics) -> dict[str, object]:
    return type_adapter_payload(
        _ARTIFACT_SEMANTICS_ADAPTER,
        semantics,
        label="semantics payload",
    )


def _decode_artifact_semantics(payload: dict[str, object]) -> ArtifactSemantics:
    semantics = type_adapter_value(
        _ARTIFACT_SEMANTICS_ADAPTER,
        payload,
        label="semantics payload",
    )
    if payload != _encode_artifact_semantics(semantics):
        raise StateLayoutError("Invalid semantics payload payload: payload is not canonical JSON")
    return semantics


def _encode_study_semantics(semantics: StudySemantics) -> dict[str, object]:
    return type_adapter_payload(
        _STUDY_SEMANTICS_ADAPTER,
        semantics,
        label="semantics payload",
    )


def _decode_study_semantics(payload: dict[str, object]) -> StudySemantics:
    semantics = type_adapter_value(
        _STUDY_SEMANTICS_ADAPTER,
        payload,
        label="semantics payload",
    )
    if payload != _encode_study_semantics(semantics):
        raise StateLayoutError("Invalid semantics payload payload: payload is not canonical JSON")
    return semantics


ARTIFACT_SEMANTICS_CODEC: PayloadCodec[ArtifactSemantics] = PayloadCodec(
    encode=_encode_artifact_semantics,
    decode=_decode_artifact_semantics,
)
STUDY_SEMANTICS_CODEC: PayloadCodec[StudySemantics] = PayloadCodec(
    encode=_encode_study_semantics,
    decode=_decode_study_semantics,
)
