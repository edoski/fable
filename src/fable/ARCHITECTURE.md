# Spice Package Architecture

## Purpose

`spice` is organized around generic ML domains for blockchain fee-decision research. Each domain owns one kind of meaning: raw blocks, features, temporal examples, tensors, model behavior, prediction semantics, evaluation metrics, or persisted state.

The package-level rule is simple:

```text
workflows coordinate domains
domains expose contracts
concrete implementations stay behind owner registries
```

## End-To-End Transformation

```text
block facts
  -> canonical corpus rows
  -> feature columns
  -> temporal problem examples
  -> model-ready tensors
  -> model outputs
  -> decoded prediction results
  -> evaluator metrics
  -> persisted roots and catalog records
```

Every arrow preserves or changes meaning. The architecture exists to keep those meanings explicit. Most ML bugs in this kind of system are boundary bugs: future information leaks into inputs, logits are treated like decisions, or artifacts are saved without the provenance needed to interpret them.

## Domain Responsibilities

```text
config       named YAML + workflow selections -> typed configs
corpus       canonical block data and coverage
features     observable feature columns and prerequisites
temporal     anchors, context rows, candidate windows, execution policies
modeling     tensorization, training, inference, scoring bridge
metrics      shared metric descriptors, metric sets, and window summaries
prediction   targets, losses, training metric production, decoded-result kinds
evaluation   decision-quality scoring over decoded results
benchmarks   benchmark specs, durable plan entries, run state, collection
storage      identities, roots, SQLite state, lifecycle, catalog
execution    explicit remote targets, resolved workflow execution, transfer
workflows    task orchestration
cli          operator-facing command edge
```

## Standard Seam

```text
payload / config model
        |
        v
owner coercer
        |
        v
local spec table
        |
        v
require_spec_config()
        |
        v
compiled contract
        |
        v
workflow usage
```

The local spec table is a small in-repo dispatch table owned by the package whose implementations it selects. This keeps architecture explicit and modular.

## Persistence Boundary

Checked-in YAML configs are validated through package-owned coercers. Runtime artifacts are stored with explicit root-kind metadata and typed payload codecs.

## Module Map

```text
spice/
  acquisition/      raw block acquisition adapters
  benchmarks/       benchmark schema, plan materialization, run state, collection
  cli/              command-line adapter
  conf/             checked-in YAML specs
  config/           config groups, typed loaders, surfaces, resolved workflow snapshots
  core/             shared primitives and errors
  corpus/           canonical block data and validation
  evaluation/       decoded decision scoring
  execution/        remote execution, transfer, and Slurm submission
  features/         observable feature construction
  metrics.py        shared metric descriptors, metric sets, and window summaries
  modeling/         datasets, models, training, inference
  prediction/       target/loss/decode semantics
  storage/          identities, roots, SQLite state, catalog, lifecycle
  temporal/         time-window problem construction
  workflows/        application task orchestration
```

## Beginner Map

Deep learning code usually has five pieces:

```text
data -> examples -> model -> loss -> evaluation
```

Spice expands each piece because blockchain fee decisions are temporal:

```text
data        corpus + features
examples    temporal compilers + execution policies
model       dataset builders + model families
loss        prediction families
evaluation  evaluator contracts + storage summaries
```

This separation lets one model architecture learn different prediction tasks, one
prediction task use different evaluator metrics at evaluation time, and one corpus
support multiple experiments.
