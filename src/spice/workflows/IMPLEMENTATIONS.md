# Concrete Workflows

Workflows orchestrate complete user operations. They do not own model architectures, feature math, storage schema, or evaluator algorithms. Workflow Preparation performs preflight once by calling Storage Root Materialization for root handles and using focused internal modules for active config, training/tuning preflight, and inference setup. Runners call owner packages, and Storage Transactions commit results.

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

Corpus Assembly performs capability planning, materializes history/evaluation splits, runs bounded refill attempts when compiled valid sample count is short, writes corpus state, and commits the root.

Dry run reports the plan without materializing data.

## Train

Train creates an artifact root.

```text
resolve config and identity
  -> materialize roots
  -> prepare active config, corpus manifest, and training spec
  -> open full-root transaction
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
  -> prepare corpus manifest and coverage spec
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
  -> upsert evaluation state with execution provenance when remote
```

Evaluate writes into the existing artifact state DB. It does not stage or replace the artifact root.

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
