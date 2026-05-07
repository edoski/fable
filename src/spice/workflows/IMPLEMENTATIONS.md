# Concrete Workflows

Workflows orchestrate complete user operations. They do not own model architectures, feature math, storage schema, or evaluator algorithms. Workflow Preparation performs preflight once by calling Storage Root Materialization for root handles and owning active config, training/tuning coverage validation, and artifact inference setup. `workflows.reporting` owns runner-facing report composition. Runners call owner packages, and Storage Transactions expose handle-shaped commit and mutation boundaries for results.

## Workflow Map

```text
acquire  -> corpus root
train    -> artifact root
tune     -> study root
evaluate -> artifact evaluation state
```

Acquire runs locally. Train, tune, and evaluate are remote CLI workflows; their Python runners remain direct entrypoints for the remote runner and tests.

## Acquire

Acquire downloads block data and writes a corpus root.

```text
resolve config
  -> materialize produced corpus root
  -> prepare acquisition request
  -> report plan
  -> create block source
  -> delegate to Corpus Assembly
  -> close block source
```

Corpus Assembly builds Corpus Capability Planning, materializes history/evaluation splits through Corpus Acquisition Stage, lets planning run bounded refill attempts when compiled valid sample count is short, writes corpus state, and commits the root.

Dry run reports the plan without materializing data.

## Train

Train creates an artifact root.

```text
resolve config and identity
  -> materialize roots
  -> prepare active config, corpus manifest, and training spec
  -> commit artifact root through storage transaction
  -> train model
  -> write manifest/model/state
  -> promote artifact root
  -> reindex catalog
```

The artifact manifest stores exact configs and semantic fingerprints. Tuned train validates study identity before applying best params.

## Tune

Tune creates or resumes a study root.

```text
resolve study identity
  -> materialize roots
  -> prepare corpus manifest and validate coverage
  -> open tuning execution
  -> delegate trial execution to modeling
  -> report trial and best-trial callbacks
  -> reindex study root
```

Trial artifacts are not persisted as final artifacts. The study stores sampled params and best epoch metadata through Tuning Execution.

## Evaluate

Evaluate runs diagnostic scoring for an existing artifact.

```text
resolve artifact
  -> materialize roots
  -> validate semantics/capability/coverage and build inference dataset
  -> score evaluator
  -> record artifact evaluation state through Storage Transactions with execution provenance when remote
```

Evaluate writes into the existing artifact state DB through `storage.transactions.record_artifact_evaluation_state()`. It does not stage or replace the artifact root.

## Remote Submission Boundary

Workflow functions require resolved configs. CLI submission chooses the target and opens the **Execution Session**. Downstream workflow code receives explicit storage roots and does not choose remote targets.

## Failure Modes

| Workflow | Common failures |
| --- | --- |
| acquire | RPC retry exhaustion, insufficient valid samples after refill, invalid corpus validation. |
| train | Missing corpus coverage, CUDA unavailable, artifact destination conflict. |
| tune | Study identity mismatch, requested trial count below existing trial count, missing coverage for max tuned lookback. |
| evaluate | Artifact semantic mismatch, evaluation delay too large, missing corpus coverage. |

## Extension Pattern

A new workflow should orchestrate existing package contracts and commit one clear storage effect. Put reusable algorithms in owner packages, not workflow files.
