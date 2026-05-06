from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from spice.benchmarks.collection import collect_benchmark_run
from spice.benchmarks.plan_materialization import (
    BenchmarkDependencyLedger,
    BenchmarkPlanEntry,
    BenchmarkRootFacts,
    BenchmarkRootLedger,
    BenchmarkRootLedgerEntry,
    BenchmarkSelectionLedger,
)
from spice.benchmarks.result_store import index_counts
from spice.benchmarks.runs import (
    BenchmarkSubmissionRecord,
    append_submission_jsonl,
    collection_snapshot_path,
    create_benchmark_run_dir,
    load_collection_snapshot,
    load_plan_jsonl,
    write_plan_jsonl,
)
from spice.config import EvaluateConfig, StorageSpec, WorkflowTask
from spice.config.groups import load_named_group_payload
from spice.config.models import ArtifactVariant
from spice.core.errors import SelectorResolutionError, SpiceOperatorError
from spice.evaluation.registry import coerce_evaluator_config
from spice.execution.transfer_transaction import TransferredArtifactRoot
from spice.storage.catalog.records import CatalogArtifactRecord
from tests.catalog_helpers import artifact_record


def _evaluate_config(tmp_path: Path) -> EvaluateConfig:
    return EvaluateConfig(
        storage=StorageSpec(root=tmp_path / "outputs"),
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        evaluation=coerce_evaluator_config(
            load_named_group_payload("poisson_replay", "evaluation")
        ),
        delay_seconds=36,
    )


def _write_evaluate_run(run_dir: Path, config) -> None:
    entry = BenchmarkPlanEntry(
        run_id="case.evaluate",
        case_id="case",
        step_id="evaluate",
        workflow=WorkflowTask.EVALUATE,
        dependencies=BenchmarkDependencyLedger(
            local_run_ids=(),
            external_slurm_dependencies=(),
            artifact_from_run_id=None,
        ),
        dimension_labels={"models": "lstm"},
        selection=BenchmarkSelectionLedger(
            evaluation=config.evaluation.id,
            delay_seconds=config.delay_seconds,
        ),
        root_facts=_evaluate_root_facts(
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
        root_ledger=_evaluate_root_ledger(
            "case.evaluate",
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
        config=config,
    )
    write_plan_jsonl(run_dir, [entry])
    append_submission_jsonl(
        run_dir,
        BenchmarkSubmissionRecord(
            run_id=entry.run_id,
            workflow=entry.workflow,
            job_id="57549",
            execution_ref="slurm:57549",
            git_commit="abc123",
            dependency=None,
            log_path="/tmp/spice-evaluate-57549.out",
        ),
    )


def _artifact_record(root_path: Path, artifact_id: str = "artifact-1") -> CatalogArtifactRecord:
    return artifact_record(root_path, artifact_id=artifact_id)


def _evaluate_root_ledger(
    run_id: str,
    *,
    artifact_id: str,
    dataset_id: str,
) -> BenchmarkRootLedger:
    return BenchmarkRootLedger(
        entries=(
            BenchmarkRootLedgerEntry(
                run_id=run_id,
                workflow=WorkflowTask.EVALUATE,
                role="consumed",
                root_kind="dataset",
                root_id=dataset_id,
                dataset_id=dataset_id,
            ),
            BenchmarkRootLedgerEntry(
                run_id=run_id,
                workflow=WorkflowTask.EVALUATE,
                role="consumed",
                root_kind="artifact",
                root_id=artifact_id,
                artifact_id=artifact_id,
                dataset_id=dataset_id,
            ),
        )
    )


def _evaluate_root_facts(
    *,
    artifact_id: str,
    dataset_id: str,
    artifact_source_dataset_id: str | None = None,
) -> BenchmarkRootFacts:
    return BenchmarkRootFacts(
        consumed_dataset_id=dataset_id,
        consumed_artifact_id=artifact_id,
        consumed_artifact_dataset_id=artifact_source_dataset_id or dataset_id,
        artifact_source_dataset_id=artifact_source_dataset_id,
    )


def _loaded_summary(config):
    return SimpleNamespace(
        evaluation_storage_id="poisson_replay-36s-storage",
        recorded_at=1_700_000_000,
        manifest=SimpleNamespace(
            artifact_id="artifact-1",
            dataset_id="dataset-1",
            chain_name="ethereum",
            dataset_name="icdcs_2026",
            features_id="core_fee_dynamics",
            model=SimpleNamespace(id="lstm"),
            problem_id="current_row_nominal",
            prediction_id="icdcs_2026",
            objective=SimpleNamespace(id="evaluation"),
            variant=ArtifactVariant.BASELINE,
            study_id=None,
            study=None,
        ),
        runtime=SimpleNamespace(
            delay_seconds=config.delay_seconds,
            evaluator_id=config.evaluation.id,
            execution_provenance=SimpleNamespace(
                execution_ref="slurm:57549",
                job_id="57549",
                log_path="/tmp/spice-evaluate-57549.out",
                workflow_task="evaluate",
                target="disi_l40",
            ),
            sample_count=123,
            total_events=7,
            n_history_rows=200,
            n_evaluation_rows=100,
            metrics=SimpleNamespace(
                values={
                    "profit_over_baseline": 0.12,
                    "cost_over_optimum": 1.5,
                    "baseline_cost_over_optimum": 1.7,
                }
            ),
            window_metrics={},
            runs=(),
        ),
    )


def test_benchmark_plan_jsonl_round_trips_plan_entry(tmp_path: Path) -> None:
    config = _evaluate_config(tmp_path)
    run_dir = create_benchmark_run_dir(
        "round_trip",
        target="disi_l40",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    entry = BenchmarkPlanEntry(
        run_id="case.evaluate",
        case_id="case",
        step_id="evaluate",
        workflow=WorkflowTask.EVALUATE,
        dependencies=BenchmarkDependencyLedger(
            local_run_ids=("case.train",),
            external_slurm_dependencies=("afterok:42",),
            artifact_from_run_id="case.train",
        ),
        dimension_labels={"features": "core"},
        selection=BenchmarkSelectionLedger(
            surface="current_row_fee_dynamics",
            evaluation=config.evaluation.id,
        ),
        root_facts=_evaluate_root_facts(
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
        root_ledger=_evaluate_root_ledger(
            "case.evaluate",
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
        config=config,
    )

    write_plan_jsonl(run_dir, [entry])
    loaded = load_plan_jsonl(run_dir)

    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.run_id == entry.run_id
    assert restored.dependencies.local_run_ids == ("case.train",)
    assert restored.dependencies.external_slurm_dependencies == ("afterok:42",)
    assert restored.dimension_labels == {"features": "core"}
    assert restored.dependencies.artifact_from_run_id == "case.train"
    assert restored.selection == entry.selection
    assert restored.root_facts == entry.root_facts
    assert restored.root_ledger == entry.root_ledger
    assert restored.config.model_dump(mode="json") == config.model_dump(mode="json")


def test_benchmark_collect_writes_snapshot_and_replaces_index_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    run_dir = create_benchmark_run_dir(
        "collect_case",
        target="disi_l40",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    _write_evaluate_run(run_dir, config)
    resolve_calls: list[object] = []
    pull_calls: list[object] = []
    pulled = TransferredArtifactRoot(
        source_record=_artifact_record(tmp_path / "remote" / "artifact-1"),
        local_record=_artifact_record(tmp_path / "local" / "artifact-1"),
        destination_root=tmp_path / "local" / "artifact-1",
        dataset_present=True,
    )

    def fake_resolve(selection, *, artifact_record):
        assert selection.execution_ref == "slurm:57549"
        resolve_calls.append(artifact_record)
        return SimpleNamespace(
            evaluation=_loaded_summary(config),
            training=SimpleNamespace(
                runtime=SimpleNamespace(
                    test_metrics=SimpleNamespace(
                        values={
                            "total_loss": 0.3,
                            "offset_accuracy": 0.8,
                            "macro_f1": 0.7,
                        }
                    )
                )
            ),
        )

    class FakeTransferTransaction:
        def pull_artifact(self, artifact_id: str):
            pull_calls.append(artifact_id)
            return pulled

    monkeypatch.setattr(
        "spice.benchmarks.collection.open_storage_transfer_transaction",
        lambda _target, *, local_storage_root: FakeTransferTransaction(),
    )
    monkeypatch.setattr("spice.benchmarks.collection.resolve_benchmark_evaluation", fake_resolve)
    index_path = tmp_path / "benchmarks" / "results.sqlite"

    snapshot = collect_benchmark_run(run_dir=run_dir, index_path=index_path)
    duplicate_snapshot = collect_benchmark_run(run_dir=run_dir, index_path=index_path)

    assert snapshot.records[0].git_commit == "abc123"
    assert duplicate_snapshot.records[0].execution_ref == "slurm:57549"
    assert load_collection_snapshot(run_dir).records[0].artifact_id == "artifact-1"
    assert index_counts(index_path) == {
        "runs": 1,
        "observations": 1,
        "metrics": 6,
    }
    assert len(resolve_calls) == 2
    assert len(pull_calls) == 2
    assert pull_calls[0] == config.artifact_id


def test_benchmark_collect_refuses_partial_snapshot_and_index_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    run_dir = create_benchmark_run_dir(
        "missing_case",
        target="disi_l40",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    _write_evaluate_run(run_dir, config)

    class FakeTransferTransaction:
        def pull_artifact(self, artifact_id: str):
            del artifact_id
            raise SelectorResolutionError(kind="artifact", records=[])

    monkeypatch.setattr(
        "spice.benchmarks.collection.open_storage_transfer_transaction",
        lambda _target, *, local_storage_root: FakeTransferTransaction(),
    )
    index_path = tmp_path / "benchmarks" / "results.sqlite"

    with pytest.raises(SpiceOperatorError, match="No artifact matches found"):
        collect_benchmark_run(run_dir=run_dir, index_path=index_path)

    assert not collection_snapshot_path(run_dir).exists()
    assert not index_path.exists()


def test_benchmark_collect_refuses_partial_write_when_summary_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    run_dir = create_benchmark_run_dir(
        "missing_summary_case",
        target="disi_l40",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    _write_evaluate_run(run_dir, config)
    pulled = TransferredArtifactRoot(
        source_record=_artifact_record(tmp_path / "remote" / "artifact-1"),
        local_record=_artifact_record(tmp_path / "local" / "artifact-1"),
        destination_root=tmp_path / "local" / "artifact-1",
        dataset_present=True,
    )

    class FakeTransferTransaction:
        def pull_artifact(self, artifact_id: str):
            del artifact_id
            return pulled

    monkeypatch.setattr(
        "spice.benchmarks.collection.open_storage_transfer_transaction",
        lambda _target, *, local_storage_root: FakeTransferTransaction(),
    )
    monkeypatch.setattr(
        "spice.benchmarks.collection.resolve_benchmark_evaluation",
        lambda *_args, **_kwargs: None,
    )
    index_path = tmp_path / "benchmarks" / "results.sqlite"

    with pytest.raises(SpiceOperatorError, match="Evaluation summary not found"):
        collect_benchmark_run(run_dir=run_dir, index_path=index_path)

    assert not collection_snapshot_path(run_dir).exists()
    assert not index_path.exists()


def test_benchmark_collect_pulls_same_artifact_once_for_multiple_evaluations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _evaluate_config(tmp_path)
    run_dir = create_benchmark_run_dir(
        "cached_pull_case",
        target="disi_l40",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    first = BenchmarkPlanEntry(
        run_id="case.evaluate_a",
        case_id="case",
        step_id="evaluate_a",
        workflow=WorkflowTask.EVALUATE,
        dependencies=BenchmarkDependencyLedger(
            local_run_ids=(),
            external_slurm_dependencies=(),
            artifact_from_run_id=None,
        ),
        dimension_labels={},
        selection=BenchmarkSelectionLedger(),
        root_facts=_evaluate_root_facts(
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
        root_ledger=_evaluate_root_ledger(
            "case.evaluate_a",
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
        config=config,
    )
    second = replace(
        first,
        run_id="case.evaluate_b",
        step_id="evaluate_b",
        root_ledger=_evaluate_root_ledger(
            "case.evaluate_b",
            artifact_id=config.artifact_id,
            dataset_id=config.dataset_id,
        ),
    )
    write_plan_jsonl(run_dir, [first, second])
    for entry in (first, second):
        append_submission_jsonl(
            run_dir,
            BenchmarkSubmissionRecord(
                run_id=entry.run_id,
                workflow=entry.workflow,
                job_id=entry.run_id,
                execution_ref=f"slurm:{entry.run_id}",
                git_commit="abc123",
                dependency=None,
                log_path="/tmp/spice-evaluate.out",
            ),
        )
    pulled = TransferredArtifactRoot(
        source_record=_artifact_record(tmp_path / "remote" / "artifact-1"),
        local_record=_artifact_record(tmp_path / "local" / "artifact-1"),
        destination_root=tmp_path / "local" / "artifact-1",
        dataset_present=True,
    )
    pull_calls: list[str] = []
    submissions: list[str] = []

    def fake_resolve(selection, *, artifact_record: CatalogArtifactRecord):
        assert artifact_record is pulled.local_record
        submissions.append(selection.execution_ref)
        return SimpleNamespace(
            evaluation=_loaded_summary(config),
            training=None,
        )

    class FakeTransferTransaction:
        def __init__(self) -> None:
            self._pulled = False

        def pull_artifact(self, artifact_id: str):
            if not self._pulled:
                pull_calls.append(artifact_id)
                self._pulled = True
            return pulled

    monkeypatch.setattr(
        "spice.benchmarks.collection.open_storage_transfer_transaction",
        lambda _target, *, local_storage_root: FakeTransferTransaction(),
    )
    monkeypatch.setattr("spice.benchmarks.collection.resolve_benchmark_evaluation", fake_resolve)

    collect_benchmark_run(run_dir=run_dir, index_path=tmp_path / "results.sqlite")

    assert len(pull_calls) == 1
    assert submissions == ["slurm:case.evaluate_a", "slurm:case.evaluate_b"]
