# pyright: strict

"""All-or-nothing benchmark collection from remote completed workflows."""

from __future__ import annotations

from pathlib import Path

from ..config.models import WorkflowTask
from ..core.errors import SelectorResolutionError, SpiceOperatorError
from ..execution.transfer_transaction import (
    StorageTransferTransaction,
    open_storage_transfer_transaction,
)
from ..storage.catalog.records import CatalogArtifactRecord
from ..storage.engine import RootKind
from .collection_resolver import (
    BenchmarkCollectionSelection,
    benchmark_collection_selection,
    resolve_benchmark_evaluation,
)
from .result_index import BENCHMARK_RESULT_INDEX_PATH, upsert_benchmark_collection_snapshot
from .result_records import (
    BenchmarkCollectionSnapshot,
    BenchmarkResultRecord,
    build_benchmark_result_record,
)
from .runs import (
    format_datetime,
    load_benchmark_run,
    utc_now,
    write_benchmark_collection_snapshot,
)


def collect_benchmark_run(
    run_dir: Path,
    *,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
) -> BenchmarkCollectionSnapshot:
    run = load_benchmark_run(run_dir)
    metadata = run.metadata
    plan = list(run.plan)
    submissions = run.submissions
    evaluate_entries = [entry for entry in plan if entry.workflow is WorkflowTask.EVALUATE]
    collector_time = utc_now()
    transfer_transactions: dict[Path, StorageTransferTransaction] = {}
    records: list[BenchmarkResultRecord] = []
    for entry in evaluate_entries:
        submission = submissions.get(entry.run_id)
        if submission is None:
            raise SpiceOperatorError(f"Missing submission record for benchmark run {entry.run_id}")
        selection = benchmark_collection_selection(
            entry,
            submission,
            target=metadata.target,
        )
        try:
            transaction = _transfer_transaction_for_selection(
                selection,
                target_name=metadata.target,
                transactions=transfer_transactions,
            )
            pulled = transaction.pull_root(RootKind.ARTIFACT, selection.artifact_id)
            if not isinstance(pulled.destination_record, CatalogArtifactRecord):
                raise SpiceOperatorError("pulled benchmark root is not an artifact")
            state = resolve_benchmark_evaluation(
                selection,
                artifact_record=pulled.destination_record,
            )
        except SelectorResolutionError as exc:
            raise SpiceOperatorError(str(exc)) from exc
        if state is None:
            raise SpiceOperatorError(
                f"Evaluation summary not found for benchmark run {entry.run_id}"
            )
        records.append(
            build_benchmark_result_record(
                entry=entry,
                submission=submission,
                resolved=state,
                collector_time=collector_time,
            )
        )
    snapshot = BenchmarkCollectionSnapshot(
        schema_version=1,
        benchmark=metadata.benchmark,
        run_dir=str(run_dir),
        target=metadata.target,
        run_created_at_utc=metadata.created_at_utc,
        collected_at_utc=format_datetime(collector_time),
        expected_evaluate_count=len(evaluate_entries),
        records=tuple(records),
    )
    write_benchmark_collection_snapshot(run_dir, snapshot)
    upsert_benchmark_collection_snapshot(snapshot, index_path=index_path)
    return snapshot


def _transfer_transaction_for_selection(
    selection: BenchmarkCollectionSelection,
    *,
    target_name: str,
    transactions: dict[Path, StorageTransferTransaction],
) -> StorageTransferTransaction:
    transaction = transactions.get(selection.storage_root)
    if transaction is None:
        transaction = open_storage_transfer_transaction(
            target_name,
            local_storage_root=selection.storage_root,
        )
        transactions[selection.storage_root] = transaction
    return transaction
