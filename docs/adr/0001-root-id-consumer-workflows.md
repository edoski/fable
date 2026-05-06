# ADR 0001: Root-ID Consumer Workflows

## Status

Accepted.

## Context

Training, tuning, evaluation, storage transfer, and benchmark collection previously rebuilt existing root paths from full workflow config identity. That made consuming an existing corpus, study, or artifact depend on reproducing the original config payload.

Evaluation carried the largest cost: it inherited training semantics, recomputed artifact identity from config, and then compared that config back to the artifact manifest.

## Decision

Producer workflows keep Workflow Selection and config-derived destination identity. Existing-root consumers use exact root ids resolved through the catalog first.

- `acquire` produces a corpus root.
- `tune` consumes `dataset_id` and produces `study_id`.
- Baseline `train` consumes `dataset_id` and produces `artifact_id`.
- Tuned `train` consumes `study_id` and produces `artifact_id`.
- `evaluate` consumes `artifact_id` and `dataset_id`.

Evaluation is manifest-first. The artifact manifest supplies trained semantics. The selected corpus manifest supplies chain runtime and evaluation window. The active evaluation config supplies only evaluator, delay, batch size, storage root, and root ids.

Cross-corpus evaluation is allowed only within the same chain. Different-chain evaluation is rejected.

## Consequences

Old surface-shaped evaluation and ambiguous destructive storage operations are removed. Benchmarks must materialize artifact ids through `artifact_from` and use a tuned train step between tune and evaluate.

Existing roots from older layouts can be regenerated instead of migrated.

## Implementation Notes

Existing-root workflow consumers resolve ids into typed root handles before orchestration:

- `storage.workflow_root_materialization` applies exact Storage Selectors for existing roots.
- `storage.workflow_root_materialization` derives produced-root ids and produced root handles.
- `workflow_roots.py` carries workflow-facing root handle models.
- `storage.transactions` carries workflow-facing, handle-shaped root mutation boundaries.
- `acquire` receives its produced corpus root from storage root materialization.
- Baseline `train` resolves `dataset_id`, then derives the artifact root.
- Tuned `train` resolves `study_id`, uses the study's dataset id to resolve the corpus, then derives the artifact root.
- `evaluate` resolves `dataset_id` and `artifact_id` independently; artifact inference validates manifest compatibility.

Benchmarks remain id-shaped at their public interface. Benchmark Plan Materialization derives `study_id`, `artifact_id`, and `dataset_id` before workflows run.
