# pyright: strict

"""Typed acquisition metadata builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from ..config.models import ChainRuntimeSpec
from ..corpus.validation import ValidationStatus

SplitKindValue = Literal["blocks"]
SplitMaterializationOutcomeValue = Literal["created", "reused", "extended", "rebuilt"]


class CorpusMetadataRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CorpusIdentity(CorpusMetadataRecord):
    id: str
    name: str


class ChainMetadata(CorpusMetadataRecord):
    name: str
    runtime: ChainRuntimeSpec

    @field_validator("runtime", mode="before")
    @classmethod
    def _coerce_runtime(_cls, value: object) -> object:
        if isinstance(value, ChainRuntimeSpec):
            return value
        if isinstance(value, Mapping):
            return ChainRuntimeSpec.model_validate(
                dict(cast(Mapping[str, object], value)),
                strict=True,
            )
        return value

    @property
    def chain_id(self) -> int:
        return self.runtime.chain_id


class SplitRequestMetadata(CorpusMetadataRecord):
    start_timestamp: int
    end_timestamp: int
    start_block: int
    end_block: int


class CompactValidationReport(CorpusMetadataRecord):
    status: ValidationStatus
    issues: dict[str, object] | None = None


class SplitCoverageMetadata(CorpusMetadataRecord):
    first_timestamp: int | None
    last_timestamp: int | None
    first_block: int | None
    last_block: int | None
    rows: int


class SplitMaterializationMetadata(CorpusMetadataRecord):
    outcome: SplitMaterializationOutcomeValue
    file_count: int


class CorpusSplitManifest(CorpusMetadataRecord):
    kind: SplitKindValue
    request: SplitRequestMetadata
    coverage: SplitCoverageMetadata
    validation: CompactValidationReport
    materialization: SplitMaterializationMetadata


class CorpusAcquisitionSourceRequirements(CorpusMetadataRecord):
    required_columns: frozenset[str]
    optional_enrichments: frozenset[str]
    temporal_unit: str
    ordering_key: str
    partition_key: str | None

    @field_validator("required_columns", "optional_enrichments", mode="before")
    @classmethod
    def _coerce_string_set(_cls, value: object) -> object:
        if isinstance(value, frozenset):
            raw_items = cast(Iterable[object], value)
        elif isinstance(value, (list, set, tuple)):
            raw_items = cast(Iterable[object], value)
        else:
            return value
        items: list[str] = []
        for item in raw_items:
            if not isinstance(item, str):
                return cast(object, value)
            items.append(item)
        return frozenset(items)

    @field_serializer("required_columns", "optional_enrichments")
    def _serialize_string_set(self, value: frozenset[str]) -> list[str]:
        return sorted(value)


class CorpusManifest(CorpusMetadataRecord):
    corpus: CorpusIdentity
    chain: ChainMetadata
    blocks: CorpusSplitManifest
    source_requirements: CorpusAcquisitionSourceRequirements
