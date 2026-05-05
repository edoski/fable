from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spice.benchmarks.collection_resolver import (
    benchmark_collection_selection,
    resolve_benchmark_evaluation,
)
from spice.benchmarks.dependency_ledger import BenchmarkDependencyLedger
from spice.benchmarks.models import BenchmarkPlanEntry
from spice.benchmarks.root_ledger import BenchmarkMaterializedRoot, BenchmarkRootLedger
from spice.benchmarks.runs import BenchmarkSubmissionRecord
from spice.benchmarks.selection_ledger import BenchmarkSelectionLedger
from spice.config import EvaluateConfig, StorageSpec, WorkflowTask
from spice.config.groups import load_named_group_payload
from spice.core.errors import SpiceOperatorError
from spice.evaluation.registry import coerce_evaluator_config
from spice.execution.transfer import PulledArtifactRoot
from spice.storage.catalog import CatalogArtifactRecord


def _evaluate_config(tmp_path: Path) -> EvaluateConfig:
    return EvaluateConfig(
        storage=StorageSpec(root=tmp_path / "outputs"),
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        evaluation=coerce_evaluator_config(
            load_named_group_payload("poisson_replay_2h", "evaluation")
        ),
        delay_seconds=36,
    )


def _submission(*, execution_ref: str = "slurm:57549") -> BenchmarkSubmissionRecord:
    return BenchmarkSubmissionRecord(
        run_id="case.evaluate",
        workflow=WorkflowTask.EVALUATE,
        job_id="57549",
        execution_ref=execution_ref,
        git_commit="abc123",
        dependency=None,
        log_path="/tmp/spice-evaluate-57549.out",
    )


def _entry(config: EvaluateConfig | None = None) -> BenchmarkPlanEntry:
    config = _evaluate_config(Path("/tmp/spice-test")) if config is None else config
    return BenchmarkPlanEntry(
        run_id="case.evaluate",
        case_id="case",
        step_id="evaluate",
        workflow=WorkflowTask.EVALUATE,
        dependencies=BenchmarkDependencyLedger(
            local_run_ids=(),
            external_slurm_dependencies=(),
            artifact_from_run_id=None,
        ),
        dimension_labels={},
        selection=BenchmarkSelectionLedger(
            evaluation=config.evaluation.id,
            delay_seconds=config.delay_seconds,
        ),
        root_ledger=BenchmarkRootLedger(
            entries=(
                BenchmarkMaterializedRoot(
                    run_id="case.evaluate",
                    workflow=WorkflowTask.EVALUATE,
                    role="consumed",
                    root_kind="dataset",
                    root_id=config.dataset_id,
                    dataset_id=config.dataset_id,
                ),
                BenchmarkMaterializedRoot(
                    run_id="case.evaluate",
                    workflow=WorkflowTask.EVALUATE,
                    role="consumed",
                    root_kind="artifact",
                    root_id=config.artifact_id,
                    artifact_id=config.artifact_id,
                    dataset_id=config.dataset_id,
                ),
            )
        ),
        config=config,
    )


def _selection(config: EvaluateConfig):
    return benchmark_collection_selection(_entry(config), _submission())


def _summary(
    config,
    *,
    delay_seconds: int | None = None,
    evaluator_id: str | None = None,
    execution_ref: str | None = "slurm:57549",
):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            delay_seconds=config.delay_seconds if delay_seconds is None else delay_seconds,
            evaluator_id=config.evaluation.id if evaluator_id is None else evaluator_id,
            execution_provenance=None
            if execution_ref is None
            else SimpleNamespace(execution_ref=execution_ref),
        )
    )


def _manifest(*, max_delay_seconds: int = 36, artifact_id: str = "artifact-1"):
    return SimpleNamespace(max_delay_seconds=max_delay_seconds, artifact_id=artifact_id)


def _artifact_record(root_path: Path, *, artifact_id: str = "artifact-1") -> CatalogArtifactRecord:
    return CatalogArtifactRecord(
        artifact_id=artifact_id,
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def _pulled_artifact(tmp_path: Path) -> PulledArtifactRoot:
    source_record = _artifact_record(tmp_path / "remote" / "artifact-1")
    local_record = _artifact_record(tmp_path / "local" / "artifact-1")
    return PulledArtifactRoot(
        source_record=source_record,
        local_record=local_record,
        destination_root=local_record.root_path,
        dataset_present=True,
    )


def test_collection_resolver_uses_pulled_artifact_and_loads_matching_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    loaded_paths: list[Path] = []
    training = SimpleNamespace(runtime=SimpleNamespace(test_metrics=None))
    summary = _summary(config)
    pulled = _pulled_artifact(tmp_path)

    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda path: loaded_paths.append(path) or training,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda path: loaded_paths.append(path) or _manifest(),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.list_evaluation_summaries",
        lambda path: loaded_paths.append(path) or [summary],
    )

    resolved = resolve_benchmark_evaluation(_selection(config), pulled=pulled)

    assert resolved is not None
    assert resolved.evaluation is summary
    assert resolved.training is training
    assert loaded_paths == [pulled.local_record.state_db_path] * 3


def test_collection_resolver_returns_none_when_summary_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    pulled = _pulled_artifact(tmp_path)
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda _path: _manifest(),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.list_evaluation_summaries",
        lambda _path: [],
    )

    assert resolve_benchmark_evaluation(_selection(config), pulled=pulled) is None


def test_collection_resolver_rejects_duplicate_matching_summaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    pulled = _pulled_artifact(tmp_path)
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda _path: _manifest(),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.list_evaluation_summaries",
        lambda _path: [_summary(config), _summary(config)],
    )

    with pytest.raises(SpiceOperatorError, match="Multiple evaluation summaries"):
        resolve_benchmark_evaluation(_selection(config), pulled=pulled)


def test_collection_resolver_rejects_missing_execution_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    pulled = _pulled_artifact(tmp_path)
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda _path: _manifest(),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.list_evaluation_summaries",
        lambda _path: [_summary(config, execution_ref=None)],
    )

    with pytest.raises(SpiceOperatorError, match="execution provenance"):
        resolve_benchmark_evaluation(_selection(config), pulled=pulled)


def test_collection_resolver_rejects_stale_execution_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    pulled = _pulled_artifact(tmp_path)
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda _path: _manifest(),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.list_evaluation_summaries",
        lambda _path: [_summary(config, execution_ref="slurm:old")],
    )

    with pytest.raises(SpiceOperatorError, match="expected slurm:57549"):
        resolve_benchmark_evaluation(_selection(config), pulled=pulled)


def test_collection_resolver_matches_default_delay_to_artifact_capability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path).model_copy(update={"delay_seconds": None})
    matching = _summary(config, delay_seconds=72)
    stale = _summary(config, delay_seconds=36)
    pulled = _pulled_artifact(tmp_path)
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda _path: _manifest(max_delay_seconds=72),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.list_evaluation_summaries",
        lambda _path: [stale, matching],
    )

    resolved = resolve_benchmark_evaluation(_selection(config), pulled=pulled)

    assert resolved is not None
    assert resolved.evaluation is matching


def test_collection_selection_rejects_run_id_mismatch(tmp_path: Path) -> None:
    config = _evaluate_config(tmp_path)

    with pytest.raises(SpiceOperatorError, match="run id"):
        benchmark_collection_selection(
            _entry(config),
            _submission().model_copy(update={"run_id": "other.evaluate"}),
        )


def test_collection_selection_rejects_root_ledger_mismatch(tmp_path: Path) -> None:
    config = _evaluate_config(tmp_path)
    entry = _entry(config)
    bad_ledger = BenchmarkRootLedger(
        entries=(
            BenchmarkMaterializedRoot(
                run_id="case.evaluate",
                workflow=WorkflowTask.EVALUATE,
                role="consumed",
                root_kind="artifact",
                root_id="other-artifact",
                artifact_id="other-artifact",
            ),
        )
    )

    with pytest.raises(SpiceOperatorError, match="artifact mismatch"):
        benchmark_collection_selection(
            entry.__class__(
                run_id=entry.run_id,
                case_id=entry.case_id,
                step_id=entry.step_id,
                workflow=entry.workflow,
                dependencies=entry.dependencies,
                dimension_labels=entry.dimension_labels,
                selection=entry.selection,
                root_ledger=bad_ledger,
                config=entry.config,
            ),
            _submission(),
        )


def test_collection_selection_rejects_missing_root_ledger_entries(tmp_path: Path) -> None:
    config = _evaluate_config(tmp_path)
    entry = _entry(config)

    with pytest.raises(SpiceOperatorError, match="missing consumed artifact"):
        benchmark_collection_selection(
            entry.__class__(
                run_id=entry.run_id,
                case_id=entry.case_id,
                step_id=entry.step_id,
                workflow=entry.workflow,
                dependencies=entry.dependencies,
                dimension_labels=entry.dimension_labels,
                selection=entry.selection,
                root_ledger=BenchmarkRootLedger(),
                config=entry.config,
            ),
            _submission(),
        )


def test_collection_resolver_rejects_pulled_artifact_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    pulled = _pulled_artifact(tmp_path)
    pulled = PulledArtifactRoot(
        source_record=pulled.source_record,
        local_record=_artifact_record(tmp_path / "local" / "other", artifact_id="other"),
        destination_root=tmp_path / "local" / "other",
        dataset_present=True,
    )

    with pytest.raises(SpiceOperatorError, match="Pulled artifact"):
        resolve_benchmark_evaluation(_selection(config), pulled=pulled)


def test_collection_resolver_rejects_manifest_artifact_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    pulled = _pulled_artifact(tmp_path)
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_training_summary",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection_resolver.load_artifact_manifest",
        lambda _path: _manifest(artifact_id="other"),
    )

    with pytest.raises(SpiceOperatorError, match="Artifact manifest"):
        resolve_benchmark_evaluation(_selection(config), pulled=pulled)
