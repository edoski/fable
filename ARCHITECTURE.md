# Architecture Guide

This document describes the current `spice` architecture after the clean-break refactor.
It is intentionally about the code that exists now, not the older shape of the repository.

## Design Goals

- one installable namespace: `spice`
- one config model: strict Pydantic
- one runtime block format: Parquet
- one settings model: `pydantic-settings`
- one RPC transport: `HTTPX`
- one console layer: `Rich` + stdlib `logging`
- one feature/data engine: `Polars` + `NumPy`
- one ML runtime: `PyTorch`

The codebase is organized to keep domain-specific logic custom while replacing infrastructure-heavy custom code with mature packages.

## Top-Level Structure

```text
src/spice/
  api.py
  cli.py
  core/
  acquisition/
  data/
  modeling/
tests/
configs/
```

`src/spice` remains the package root because `spice` is the public import namespace.
The internal modules use relative imports so the code does not repeat the `spice.` prefix everywhere.

## Package Breakdown

### `core`

`core` contains the narrow shared primitives used by the rest of the repo:

- `config.py`: strict Pydantic models for experiment config
- `settings.py`: environment-backed runtime settings via `BaseSettings`
- `constants.py`: shared filenames and evaluation timestamps
- `console.py`: reporter protocol, silent reporter, and Rich-backed CLI reporter

This layer has no chain-specific logic and no model-specific logic.

### `acquisition`

`acquisition` owns everything needed to go from provider settings to validated block datasets:

- `cryo.py`: pull planning plus streamed subprocess execution
- `rpc_providers.py`: provider resolution and secret redaction
- `rpc.py`: narrow batched JSON-RPC client over HTTPX
- `enrich.py`: table-oriented `gas_limit` hydration
- `raw_validation.py`: file-range and timestamp validation for raw pulls
- `provenance.py`: persisted dataset source manifests

Key invariants:

- raw pulls and enrichment are separate stages
- validation is read-only
- manifests are strict Pydantic payloads
- promotion requires an existing raw manifest

### `data`

`data` owns the block-table and supervised-dataset math:

- `io.py`: Parquet-only dataset IO
- `features.py`: Polars-based feature table construction
- `datasets.py`: temporal geometry and array-backed store construction
- `normalization.py`: overlap-aware weighted scaling using `StandardScaler`

Key invariants:

- runtime block datasets are Parquet only
- enriched datasets are canonical model inputs with exactly six `Int64` block columns
- duplicate block numbers are rejected
- mixed chain IDs are rejected
- train-only scaling is computed from overlapped window coverage

### `modeling`

`modeling` owns the ML and simulation runtime:

- `models.py`: LSTM, Transformer, and Transformer-LSTM baselines
- `torch_datasets.py`: lazy sequence slicing into PyTorch batches
- `_runtime.py`: device and loader helpers
- `evaluation.py`: low-level batch metric aggregation
- `training.py`: epoch loop and early stopping
- `pipeline.py`: training/inference dataset preparation
- `inference.py`: batched prediction
- `simulation.py`: evaluation-day temporal simulation
- `artifacts.py`: model persistence
- `reporting.py`: structured train and simulation reports

Key invariants:

- the task is bounded-delay block selection, not full trajectory forecasting
- action `0` means next block
- action `k` means wait `k` extra blocks beyond the next-block baseline
- training, inference, and simulation all consume the same array-backed store contracts

## Workflow

The operational flow is:

1. pull raw blocks with `cryo`
2. validate raw output
3. enrich missing `gas_limit`
4. build feature tables and temporal stores
5. fit a weighted train-only scaler
6. train a baseline model
7. persist artifacts and JSON reports
8. run evaluation-day simulation from the persisted artifact

At the API/CLI boundary this is exposed as:

- `blocks plan`
- `blocks pull`
- `blocks stage`
- `blocks acquire`
- `blocks enrich`
- `blocks validate`
- `blocks promote`
- `train`
- `simulate`

## Why These Packages

The refactor deliberately replaced custom infrastructure code with mature packages where it was a clear simplification:

- manual schema/config code -> `Pydantic v2`
- manual `.env` loading -> `pydantic-settings`
- mixed-format row IO -> `Polars` Parquet pipeline
- `urllib` RPC transport -> `HTTPX`
- custom scaler statistics -> `scikit-learn` `StandardScaler(sample_weight=...)`
- ad hoc console output -> `Rich`

The research-specific logic remains custom:

- temporal action semantics
- dataset geometry
- label construction
- model definitions
- loss and economic metrics
- evaluation-day simulation rules

## Reports And Manifests

Persisted JSON boundary objects are strict Pydantic models:

- raw source manifests
- enriched source manifests
- training artifact manifests
- training run reports
- simulation reports

There is no multi-version schema handling in the codebase. The repository assumes one current schema.

## Tests

The test suite is intentionally aligned to the current architecture only.

It covers:

- config parsing
- provider resolution
- Parquet IO and raw validation
- enrichment and RPC retry behavior
- dataset preparation
- training/simulation workflows
- console reporting behavior
- CLI smoke paths

The suite does not carry transition checks for removed modules or removed formats.

## Extension Rules

If the codebase grows, keep the same boundaries:

- put shared config/settings/reporting primitives in `core`
- keep network and dataset acquisition logic in `acquisition`
- keep table math and dataset assembly in `data`
- keep model/runtime logic in `modeling`
- keep `api.py` as the supported Python façade
- keep `cli.py` as the supported operational surface

New work should fit one of those layers without reintroducing flat top-level module sprawl or parallel code paths.
