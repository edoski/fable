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

- [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py): Pydantic runtime config schema plus Hydra/OmegaConf coercion
- [constants.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/constants.py): shared filenames and evaluation timestamps
- [console.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/console.py): Rich-backed workflow reporting
- [json.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/json.py): shared JSON artifact serialization
- [tracking.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/tracking.py): MLflow setup and structured logging helpers

### `acquisition`

- [provider.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/provider.py): thin provider config to `web3.py` bridge
- [rpc.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/rpc.py): timestamp-window resolution, block-range planning, canonical block-field extraction, and Parquet writing
- [datasets.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/datasets.py): dataset reuse, validation, rebuilding, and history-window expansion
- [metadata.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/metadata.py): dataset metadata loading, validation, and serialization
- [windowing.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/windowing.py): history window sizing, reuse, and backward expansion rules

This layer no longer contains snapshot registries or a custom JSON-RPC transport.

### `data`

- [io.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/io.py): block-dataset discovery and parquet IO contract
- [block_contract.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/block_contract.py): canonical block-field contract, row mapping, and Pandera dataframe validation
- [features.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/features.py): feature engineering
- [datasets.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/datasets.py): temporal geometry and store construction
- [normalization.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/normalization.py): scaler fitting and transformation
- [validation_base.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/validation_base.py): shared validation report primitives

`io.py` stays custom because it is not just a parquet wrapper. It centralizes the
repo’s dataset-path contract:

- only parquet inputs are accepted
- the single hidden metadata namespace is `.spice/metadata.json`
- dataset roots may be a file or directory
- block loads always pass through canonical validation

The actual parquet engine is already delegated to `Polars`. Replacing this small
adapter would not reduce real complexity.

### `modeling`

- [models.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/models.py): custom model definitions using mature `torch.nn` layers
- [datamodule.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/datamodule.py): Lightning dataloaders and class-weight plumbing
- [lightning_module.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/lightning_module.py): Lightning training harness
- [training.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/training.py): trainer assembly and evaluation helpers
- [pipeline.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/pipeline.py): dataset preparation for training and inference
- [execution.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/execution.py): shared persisted training execution used by train and tune
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
- [dvc.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/dvc.py): DVC params loader and stage dispatcher
- [train.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/train.py)
- [simulate.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/simulate.py)
- [tune.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/tune.py)
- [_shared.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/_shared.py): workflow session/runtime helpers, JSON emission, and small shared config utilities

Each workflow exposes:

- a callable `run(...)` function for direct execution and tests
- a Hydra `main(...)` wrapper for installed entrypoints

The workflow layer is intentionally thin:

- `workflows/_shared.py` owns reporter lifecycle, MLflow setup, config logging, and nested run handling
- `workflows/dvc.py` owns the DVC-only `params.yaml` loading path and then dispatches the same workflow functions
- `acquire.py` orchestrates stage order only and delegates pull/window/metadata policy to `acquisition/*`
- `train.py` and `tune.py` both route persisted model/report generation through `modeling/execution.py`

## Workflow Graph

DVC is the primary orchestration surface:

1. `acquire`
2. `tune`
3. `train`
4. `simulate`

Dataset storage is keyed by explicit dataset windows:

- `artifacts/datasets/<chain>/<dataset_id>/history/...`
- `artifacts/datasets/<chain>/<dataset_id>/evaluation/...`
- `artifacts/datasets/<chain>/<dataset_id>/.spice/metadata.json`

Model storage is keyed by the training dataset window:

- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/...`
- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuning/...`

`dataset.id`, `dataset.window.start_date`, `dataset.window.end_date`,
`dataset.temporal.*`, and `dataset.sampling.*` define the active dataset
configuration.

In the DVC thesis path, `train` explicitly depends on the tuned model-local
`best_params.json` artifact. The `spice-dvc train` runner applies that artifact
intentionally. Direct `spice-train` usage still defaults to
`tuning.apply_best_params=false`.

DVC Hydra composition is enabled in [.dvc/config](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/.dvc/config). Stage definitions live in [dvc.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/dvc.yaml), and the composed baseline consumed by the stages lives in [params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml).

## Why These Package Choices

Selected:

- `Hydra`: owns runtime defaults through structured composition
- `DVC`: replaces snapshot/provenance orchestration with stage-based reproducibility
- `MLflow`: replaces ad hoc run bookkeeping
- `Lightning`: replaces the handwritten training loop and dataloader plumbing
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
