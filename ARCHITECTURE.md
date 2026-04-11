# Architecture Guide

This document describes the current `spice` architecture after the clean-break refactor.
It is intentionally about the code that exists now.

## Design Goals

- one runtime configuration system: `Hydra`
- one reproducibility layer: `DVC`
- one run-tracking layer: `MLflow`
- one training runtime: `Lightning`
- one tuning engine: `Optuna`
- one RPC transport stack: `web3.py`
- one dataframe validation stack: `Pandera` + `Polars`
- one ML runtime: `PyTorch`

The codebase stays aggressive about replacing infrastructure code and conservative
about replacing research-specific logic.

## Top-Level Structure

```text
src/spice/
  acquisition/
  conf/
  core/
  data/
  modeling/
  workflows/
tests/
dvc.yaml
params.yaml
```

## Package Breakdown

### `core`

- [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py): structured runtime config dataclasses plus Hydra/OmegaConf coercion and validation
- [constants.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/constants.py): shared filenames and evaluation timestamps
- [console.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/console.py): Rich-backed workflow reporting
- [tracking.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/tracking.py): MLflow setup and structured logging helpers

### `acquisition`

- [cryo.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/cryo.py): cryo command planning and streamed execution
- [provider.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/provider.py): thin provider config to `web3.py` bridge
- [rpc.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/rpc.py): block gas-limit hydration client on top of `web3.py`
- [enrich.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/enrich.py): canonical enrichment
- [raw_validation.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/raw_validation.py): raw parquet pull validation, including Pandera-validated cross-file summary checks

This layer no longer contains snapshot registries or a custom JSON-RPC transport.

### `data`

- [io.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/io.py): block-dataset discovery and parquet IO contract
- [block_schema.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/block_schema.py): canonical block schemas and Pandera dataframe validation
- [features.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/features.py): feature engineering
- [datasets.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/datasets.py): temporal geometry and store construction
- [normalization.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/normalization.py): scaler fitting and transformation

`io.py` stays custom because it is not just a parquet wrapper. It centralizes the
repo’s dataset-path contract:

- only parquet inputs are accepted
- the single hidden metadata namespace is `.spice/metadata.json`
- dataset roots may be a file or directory
- enriched loads always pass through canonical validation

The actual parquet engine is already delegated to `Polars`. Replacing this small
adapter would not reduce real complexity.

### `modeling`

- [models.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/models.py): custom model definitions using mature `torch.nn` layers
- [lightning_module.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/lightning_module.py): Lightning training harness
- [training.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/training.py): trainer assembly and evaluation helpers
- [pipeline.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/pipeline.py): dataset preparation for training and inference
- [inference.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/inference.py): prediction
- [simulation.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/simulation.py): temporal-only Poisson simulation
- [artifacts.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/artifacts.py): model persistence
- [reporting.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/reporting.py): structured reports

What stays custom here:

- temporal dataset geometry
- bounded-delay action semantics
- dual-head model outputs
- loss semantics
- economic metrics
- temporal-only Poisson evaluation logic

### `workflows`

- [acquire.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/acquire.py)
- [train.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/train.py)
- [simulate.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/simulate.py)
- [tune.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/tune.py)

Each workflow exposes:

- a callable `run(...)` function for direct execution and tests
- a Hydra `main(...)` wrapper for installed entrypoints

## Workflow Graph

DVC is the primary orchestration surface:

1. `acquire`
2. `tune`
3. `train`
4. `simulate`

Dataset storage is keyed by explicit dataset windows:

- `artifacts/datasets/<chain>/<dataset_id>/raw/...`
- `artifacts/datasets/<chain>/<dataset_id>/enriched/...`
- `artifacts/datasets/<chain>/<dataset_id>/.spice/metadata.json`

Model storage is keyed by the training dataset window:

- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/...`
- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuning/...`

`dataset.id`, `dataset.evaluation_start_timestamp`,
`dataset.evaluation_end_timestamp`, and `dataset.min_history_anchor_count`
define the active dataset window. `target_anchor_count` remains the
training/tuning sample count.

In the DVC thesis path, `train` explicitly depends on the tuned model-local
`best_params.json` artifact and runs with `tuning.apply_best_params=true`, so
the best Optuna result is consumed intentionally rather than implicitly. Direct
`spice-train` usage still defaults to `tuning.apply_best_params=false`.

Stage definitions live in [dvc.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/dvc.yaml), and DVC-facing run variables live in [params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml).

## Why These Package Choices

Selected:

- `Hydra`: replaces the custom YAML/settings loader with structured composition
- `DVC`: replaces snapshot/provenance orchestration with stage-based reproducibility
- `MLflow`: replaces ad hoc run bookkeeping
- `Lightning` + `TorchMetrics`: replaces the handwritten training loop and standard metric plumbing
- `Optuna`: provides a first-class HPO path
- `web3.py`: replaces the custom RPC transport layer
- `Pandera`: replaces bespoke dataframe validation while still allowing custom table derivation logic

Rejected:

- `sklearn` as a model replacement
- forecast-first time-series frameworks
- `SimPy` for the current temporal-only simulator
- `Prefect`, `Kedro`, and `Great Expectations`

## Tests

The test suite now targets the new architecture only:

- Hydra config composition and validation
- provider construction and RPC adapters
- Pandera-backed validation behavior
- data preparation
- acquisition, training, simulation, and tuning workflow smoke tests
- MLflow local tracking smoke coverage

There are no transition tests for deleted legacy modules.
