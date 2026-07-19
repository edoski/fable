# Corpus Architecture

## Purpose

`corpus` owns canonical block data and coverage checks. It is the boundary between raw acquisition output and downstream ML feature construction.

## Corpus Identity

Corpus identity is durable data identity:

```text
chain + corpus name + corpus window -> corpus id
```

It does not include model, prediction, evaluator, or artifact settings.

## Acquisition To Corpus Flow

```text
provider/RPC blocks
  |
  v
canonical block rows
  |
  v
blocks parquet dataset
  |
  v
corpus manifest + acquire run record
  |
  v
corpus root state
```

External uncertainty enters through acquisition: provider failures, missing blocks, latency, and chain-specific payload details. Corpus validation resolves those concerns before feature construction starts.

## Corpus Assembly

Corpus Assembly owns acquisition-to-corpus orchestration. It builds a Corpus Acquisition Planning context, asks it for the configured block-range plan, delegates committed fulfillment to Corpus Acquisition Stage, and returns the committed corpus result.

Corpus Acquisition Planning owns feature/problem contract compilation for acquisition coverage and generic Corpus Acquisition Source Requirements. It consumes `required_source_columns` and `acquisition_enrichments` from the compiled feature contract. Source requirements describe required source columns, optional enrichments, temporal unit, ordering key, and partition key. They are corpus-level requirements, not RPC configuration. Concrete acquisition adapters receive them and either bind them to provider behavior or fail before acquisition starts.

Corpus Acquisition Stage owns acquire staging paths, materialization session lifecycle, planning-to-intent adaptation, blocks fulfillment, state DB staging, partial commit wiring, successful cleanup, and preserve-on-failure behavior. Corpus Split Materialization fulfills the blocks intent and owns staged/committed candidate loading, target assessment, pull execution, chunk writing, source reuse, and validation of the blocks parquet dataset. Validation includes active required source columns, so a corpus with null required source columns cannot be reused or promoted. Its public session interface stays small; private implementation keeps parquet IO, source reuse, acquisition pulls, invariant checks, and materializer control flow local.

The corpus manifest records the requested blocks range, observed coverage, validation status/issues, materialization outcome, chain metadata, and source requirements. Coverage owns observed rows and ranges; validation owns cleanliness and issues.

The acquire workflow supplies a Workflow Config, resolved paths, and a block source constructed with corpus-derived source requirements. It does not know feature-source policy, requirement-to-provider mapping, capability planning, split materialization, refill, staging, or publication ordering.

## Coverage Checks

Training and evaluation workflows ask corpus coverage whether the stored block windows satisfy the temporal problem and feature prerequisites.

```text
corpus manifest
  + feature prerequisites
  + temporal problem coverage requirement
  -> pass/fail before model work starts
```

This prevents expensive training or evaluation from running on a corpus that cannot support the requested examples.

## Boundaries

Corpus code should not know about model families, evaluator adapters, training
losses, or artifact variants. Corpus Capability Planning may compile
feature/problem contracts only to prove raw corpus coverage during acquisition.
