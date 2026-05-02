# Corpus Architecture

## Purpose

`corpus` owns canonical block data and coverage checks. It is the boundary between raw acquisition output and downstream ML feature construction.

## Corpus Identity

Corpus identity is durable data identity:

```text
chain + dataset name + evaluation date -> corpus id
```

It does not include model, prediction, objective, evaluator, or artifact settings. It also does not include the incidental acquisition window. Acquisition windows are provenance for a run; corpus identity is the named canonical dataset context.

## Acquisition To Corpus Flow

```text
provider/RPC blocks
  |
  v
canonical block rows
  |
  v
history and evaluation parquet datasets
  |
  v
dataset manifest + acquire run record
  |
  v
corpus root state
```

External uncertainty enters through acquisition: provider failures, missing blocks, latency, and chain-specific payload details. Corpus validation resolves those concerns before feature construction starts.

## Corpus Assembly

Corpus Assembly owns acquisition-to-corpus orchestration. It builds a Corpus Capability Planning context, asks it for history/evaluation window and refill decisions, delegates staging/fulfillment/commit mechanics to Corpus Acquisition Stage, builds dataset provenance, and returns the committed corpus result.

Corpus Capability Planning owns feature/problem contract compilation for acquisition coverage, initial history sizing, valid temporal capability sample counting, and bounded refill policy. Corpus Acquisition Stage owns acquire staging paths, materialization session lifecycle, history refill sequencing, evaluation fulfillment, state DB staging, partial commit wiring, successful cleanup, and preserve-on-failure behavior. Corpus Split Materialization fulfills Split Intents: its pure fulfillment plan owns staged reuse, cached reuse, extension, rebuild/create, and pull-range decisions; the materialization session owns completed-prefix validation, chunk writing, and validation of history/evaluation parquet datasets. Extension paths reuse whole clean parquet chunks and materialize only missing or edge block ranges.

The acquire workflow supplies a Workflow Config, resolved paths, and a block source. It does not know capability planning, split materialization, refill, staging, or publication ordering.

## Coverage Checks

Training and evaluation workflows ask corpus coverage whether the stored block windows satisfy the temporal problem and feature prerequisites.

```text
dataset manifest
  + feature prerequisites
  + temporal problem coverage requirement
  -> pass/fail before model work starts
```

This prevents expensive training or evaluation from running on a corpus that cannot support the requested examples.

## Boundaries

Corpus code should not know about model families, evaluator engines, objectives, or artifact variants. Corpus Capability Planning may compile feature/problem contracts only to prove raw corpus coverage during acquisition.
