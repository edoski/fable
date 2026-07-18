"""Artifact loading and serving runtime assembly."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from ..config.models import ChainSpec
from ..corpus.metadata import CorpusAcquisitionSourceRequirements
from ..features import CompiledFeatureContract, compile_feature_contract
from ..modeling.artifacts import LoadedTrainingArtifact, load_training_artifact
from ..modeling.dataset_builders import SequenceRuntimeMetadata
from ..modeling.runtime_planning import ModelingRuntimePlan, build_cpu_modeling_runtime_plan
from ..prediction import CompiledPredictionContract, compile_prediction_contract
from ..storage.catalog.index import resolve_artifact_record
from ..storage.selectors import ArtifactSelector
from ..storage.workflow_roots import artifact_root_handle_from_record
from ..temporal.compilers.observed_time_window import ObservedTimeWindowRuntimeMetadata
from ..temporal.contracts import CompiledProblemContract, compile_problem_contract
from .config import SEPOLIA_CHAIN_ID, ServingConfig

_CORE_CORPUS_SOURCE_COLUMNS = frozenset({"block_number", "timestamp", "chain_id"})


@dataclass(frozen=True, slots=True)
class ServingRuntime:
    config: ServingConfig
    artifact: LoadedTrainingArtifact
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract
    prediction_contract: CompiledPredictionContract
    runtime_plan: ModelingRuntimePlan
    source_requirements: CorpusAcquisitionSourceRequirements
    support_block_count: int

    @property
    def chain(self) -> ChainSpec:
        return self.config.chain

    @property
    def artifact_id(self) -> str:
        return self.artifact.manifest.artifact_id

    @property
    def slot_spacing_seconds(self) -> float:
        metadata = self.artifact.manifest.temporal_capability.compiler_runtime_metadata
        if not isinstance(metadata, ObservedTimeWindowRuntimeMetadata):
            raise TypeError("serving requires observed_time_window temporal capability metadata")
        return float(metadata.slot_spacing_seconds)

    @property
    def sequence_length(self) -> int:
        metadata = self.artifact.manifest.sequence_runtime_metadata
        if not isinstance(metadata, SequenceRuntimeMetadata):
            raise TypeError("serving requires sequence runtime metadata")
        return int(metadata.sequence_length)


def load_serving_runtime(config: ServingConfig) -> ServingRuntime:
    if config.chain.runtime.chain_id != SEPOLIA_CHAIN_ID:
        raise ValueError("serving runtime requires Sepolia chain config")

    record = resolve_artifact_record(
        config.storage_root,
        selector=ArtifactSelector(artifact_id=config.artifact_id),
    )
    artifact_root = artifact_root_handle_from_record(config.storage_root, record)
    artifact = load_training_artifact(artifact_root.root_path)
    manifest = artifact.manifest
    artifact_chain_name = config.artifact_chain_name or config.chain.name
    if manifest.chain_name != artifact_chain_name:
        raise ValueError(
            "serving artifact chain does not match configured artifact chain: "
            f"{manifest.chain_name} != {artifact_chain_name}"
        )

    feature_contract = compile_feature_contract(features=manifest.features)
    if feature_contract.feature_graph_fingerprint != manifest.feature_graph_fingerprint:
        raise ValueError("current feature graph does not match serving artifact")
    if feature_contract.feature_prerequisites != manifest.feature_prerequisites:
        raise ValueError("current feature prerequisites do not match serving artifact")

    problem_contract = compile_problem_contract(
        problem=manifest.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    prediction_contract = compile_prediction_contract(
        prediction_id=manifest.prediction.id,
        family_id=manifest.prediction.family_id,
    )
    runtime_plan = build_cpu_modeling_runtime_plan(
        batch_size=config.batch_size,
        deterministic=manifest.training.deterministic,
        seed=manifest.training.seed,
    )
    support_block_count = _support_block_count(
        sequence_length=_sequence_length(manifest.sequence_runtime_metadata),
        feature_warmup_rows=feature_contract.feature_prerequisites.warmup_rows,
        feature_history_seconds=feature_contract.feature_prerequisites.history_seconds,
        problem_history_seconds=problem_contract.required_history_seconds,
        nominal_block_time_seconds=config.chain.runtime.nominal_block_time_seconds,
    )
    required_columns = frozenset(feature_contract.required_source_columns)
    return ServingRuntime(
        config=config,
        artifact=artifact,
        feature_contract=feature_contract,
        problem_contract=problem_contract,
        prediction_contract=prediction_contract,
        runtime_plan=runtime_plan,
        source_requirements=CorpusAcquisitionSourceRequirements(
            required_columns=_CORE_CORPUS_SOURCE_COLUMNS | required_columns,
            optional_enrichments=feature_contract.acquisition_enrichments,
            temporal_unit="block",
            ordering_key="block_number",
            partition_key="chain_id",
        ),
        support_block_count=support_block_count,
    )


def _sequence_length(metadata: object) -> int:
    if not isinstance(metadata, SequenceRuntimeMetadata):
        raise TypeError("serving requires sequence runtime metadata")
    return int(metadata.sequence_length)


def _support_block_count(
    *,
    sequence_length: int,
    feature_warmup_rows: int,
    feature_history_seconds: int,
    problem_history_seconds: int,
    nominal_block_time_seconds: float,
) -> int:
    spacing = max(1, ceil(nominal_block_time_seconds))
    history_rows = ceil((feature_history_seconds + problem_history_seconds) / spacing)
    return max(2, sequence_length + feature_warmup_rows + history_rows + 4)
