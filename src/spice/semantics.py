# pyright: strict

"""Typed seam semantics and root-level provenance bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .core.reporting import StageMetricDescriptor
    from .features.families.base import FeaturePrerequisites
    from .prediction.base import MetricDescriptor


@dataclass(frozen=True, slots=True)
class FeatureSemantics:
    """Normalized feature provenance derived once from the compiled feature contract."""

    feature_set_id: str
    feature_family_id: str
    feature_names: tuple[str, ...]
    feature_graph_fingerprint: str
    feature_prerequisites: FeaturePrerequisites


@dataclass(frozen=True, slots=True)
class ProblemSemantics:
    """Normalized temporal-compiler provenance for one compiled problem contract."""

    compiler_id: str
    problem_id: str
    lookback_seconds: int
    sample_count: int
    max_delay_seconds: int


@dataclass(frozen=True, slots=True)
class RealizationPolicySemantics:
    """Resolved realization-policy identity for persisted provenance."""

    realization_policy_id: str


@dataclass(frozen=True, slots=True)
class PredictionSemantics:
    """Family-owned prediction semantics that persist with studies and artifacts."""

    prediction_id: str
    prediction_family_id: str
    training_metric_descriptors: tuple[MetricDescriptor, ...]
    progress_metric_descriptors: tuple[StageMetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    supported_workflows: frozenset[str]


@dataclass(frozen=True, slots=True)
class InputNormalizationSemantics:
    """Resolved input-normalization identity for persisted provenance."""

    input_normalization_id: str


@dataclass(frozen=True, slots=True)
class RepresentationSemantics:
    """Resolved model-input representation identity for persisted provenance."""

    representation_id: str


@dataclass(frozen=True, slots=True)
class DatasetBuilderSemantics:
    """Resolved dataset-builder identity for persisted provenance."""

    dataset_builder_id: str


@dataclass(frozen=True, slots=True)
class CorpusSemantics:
    """Canonical corpus provenance shared by dataset manifests and acquire runs."""

    problem: ProblemSemantics
    feature: FeatureSemantics


@dataclass(frozen=True, slots=True)
class StudySemantics:
    """Canonical study provenance bundled from the compiled architectural seams."""

    problem: ProblemSemantics
    realization_policy: RealizationPolicySemantics
    feature: FeatureSemantics
    prediction: PredictionSemantics
    input_normalization: InputNormalizationSemantics
    representation: RepresentationSemantics
    dataset_builder: DatasetBuilderSemantics


@dataclass(frozen=True, slots=True)
class ArtifactSemantics:
    """Canonical artifact provenance bundled from the compiled architectural seams."""

    problem: ProblemSemantics
    realization_policy: RealizationPolicySemantics
    feature: FeatureSemantics
    prediction: PredictionSemantics
    input_normalization: InputNormalizationSemantics
    representation: RepresentationSemantics
    dataset_builder: DatasetBuilderSemantics
    max_candidate_slots: int
