from __future__ import annotations

from typing import cast

import pytest

from spice.config import TrainConfig, WorkflowTask
from spice.config.models import ChainRuntimeSpec
from spice.core.errors import StateConflictError
from spice.corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from spice.corpus.metadata import (
    BlockRangeMetadata,
    ChainMetadata,
    CompactValidationReport,
    CorpusSplitManifest,
    CorpusSplitManifests,
    DatasetIdentity,
    DatasetManifest,
    SplitCoverageMetadata,
    SplitMaterializationMetadata,
    SplitRequestMetadata,
    TimestampRangeMetadata,
)
from spice.features import compile_feature_contract
from spice.temporal.contracts import compile_problem_contract


def _manifest(*, history_seconds: int, history_rows: int) -> DatasetManifest:
    history = _split_manifest(
        "history",
        start_timestamp=1000,
        end_timestamp=1000 + history_seconds,
        first_block=1,
        rows=history_rows,
    )
    evaluation = _split_manifest(
        "evaluation",
        start_timestamp=2000,
        end_timestamp=2100,
        first_block=10_000,
        rows=64,
    )
    return DatasetManifest(
        dataset=DatasetIdentity(id="cor_test", name="test"),
        chain=ChainMetadata(
            name="ethereum",
            runtime=ChainRuntimeSpec(
                chain_id=1,
                uses_poa_extra_data=False,
                nominal_block_time_seconds=12.0,
            ),
        ),
        splits=CorpusSplitManifests(history=history, evaluation=evaluation),
    )


def _split_manifest(
    kind: str,
    *,
    start_timestamp: int,
    end_timestamp: int,
    first_block: int,
    rows: int,
) -> CorpusSplitManifest:
    last_block = first_block + rows - 1
    return CorpusSplitManifest(
        kind=kind,
        request=SplitRequestMetadata(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            start_block=first_block,
            end_block=first_block + rows,
        ),
        coverage=SplitCoverageMetadata(
            first_timestamp=start_timestamp,
            last_timestamp=end_timestamp,
            first_block=first_block,
            last_block=last_block,
            rows=rows,
        ),
        validation=CompactValidationReport(
            status="clean",
            rows=rows,
            block_range=BlockRangeMetadata(first=first_block, last=last_block),
            timestamp_range=TimestampRangeMetadata(
                first=start_timestamp,
                last=end_timestamp,
            ),
        ),
        materialization=SplitMaterializationMetadata(outcome="created", file_count=1),
    )


def test_training_corpus_coverage_accepts_compiled_requirement(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            override=model_workflow_override(sample_count=4, lookback_seconds=24),
        ),
    )
    feature_contract = compile_feature_contract(features=config.features)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    requirement = training_coverage_requirement(contract)

    validate_corpus_coverage(
        _manifest(
            history_seconds=requirement.history_seconds,
            history_rows=requirement.history_rows,
        ),
        contract=contract,
        feature_contract=feature_contract,
        requirement=requirement,
    )


def test_training_corpus_coverage_rejects_short_history(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            override=model_workflow_override(sample_count=4, lookback_seconds=24),
        ),
    )
    feature_contract = compile_feature_contract(features=config.features)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    requirement = training_coverage_requirement(contract)

    with pytest.raises(StateConflictError, match="history coverage is insufficient"):
        validate_corpus_coverage(
            _manifest(
                history_seconds=requirement.history_seconds - 1,
                history_rows=requirement.history_rows,
            ),
            contract=contract,
            feature_contract=feature_contract,
            requirement=requirement,
        )
