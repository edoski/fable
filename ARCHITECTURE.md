# Architecture Guide

## Overview

The codebase has four runtime stages:

1. `acquire`
2. `tune`
3. `train`
4. `simulate`

The same typed config flow powers direct entrypoints and DVC stages.

## Config Flow

Config loading lives in [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py).

Flow:

1. Load [params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml)
2. Read `presets.chain`, `presets.provider`, and `presets.model`
3. Compose the task root from [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf)
4. Apply chain-specific RPC profile overrides
5. Overlay the public params payload
6. Apply CLI overrides
7. Derive runtime paths
8. Validate the final config through Pydantic

Public dataset definition:

- `evaluation.date` defines the fixed one-day evaluation window
- `dataset.sampling.sample_count` defines training and tuning sample count
- `acquisition.history_sample_budget` optionally increases acquired history beyond `sample_count`

## Package Roles

### `core`

- [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py): typed config schema, preset composition, path derivation
- [console.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/console.py): workflow reporting
- [tracking.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/tracking.py): MLflow setup and config logging
- [json.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/json.py): JSON artifact writes
- [files.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/files.py): atomic file and directory promotion helpers

### `acquisition`

- [rpc.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/rpc.py): block planning, RPC pulling, adaptive batching, evaluation-window resolution
- [datasets.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/datasets.py): history/evaluation reuse, prefix extension, rebuilds, and validation
- [metadata.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/metadata.py): typed dataset metadata
- [windowing.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/acquisition/windowing.py): exact history block-count requirement

### `data`

- [block_contract.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/block_contract.py): canonical block schema
- [io.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/io.py): parquet dataset discovery and loading
- [features.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/features.py): feature table construction
- [datasets.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/datasets.py): temporal geometry, trimming, split indices, inference filtering
- [normalization.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/normalization.py): scaler fitting and application
- [validation.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/data/validation.py): contiguous and exact-window dataset validation

### `modeling`

- [models.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/models.py): model construction by family
- [pipeline.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/pipeline.py): training and inference dataset preparation
- [training.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/training.py): trainer execution and metrics
- [execution.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/execution.py): persisted training flow
- [artifacts.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/artifacts.py): model + manifest persistence
- [simulation.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/simulation.py): Poisson arrival simulation over evaluation examples
- [reporting.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/modeling/reporting.py): structured training and simulation reports

### `workflows`

- [acquire.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/acquire.py): acquisition orchestration
- [tune.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/tune.py): Optuna orchestration
- [train.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/train.py): artifact-producing training orchestration
- [simulate.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/simulate.py): evaluation-day simulation orchestration
- [_shared.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/_shared.py): runtime session helpers and training spec construction
- [_tuning.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/_tuning.py): tuning search-space application and typed tuning artifacts
- [dvc.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/workflows/dvc.py): DVC stage loader and dispatcher

## Workflow Semantics

### Acquire

Acquire builds two canonical datasets under the dataset root:

- `history`
- `evaluation`

Semantics:

- evaluation window is exactly one UTC day from `evaluation.date`
- history requirement is derived from temporal geometry and `effective_history_sample_budget`
- if an existing history dataset already covers the required window, it is reused
- if it ends at the correct boundary but starts too late, only the missing prefix is fetched
- if an existing evaluation dataset overlaps the target window, only missing edges are fetched
- successful outputs are promoted atomically

### Tune

Tune reads the canonical history dataset, prepares `sample_count` supervised examples, runs Optuna trials, and writes:

- `study.json`
- `trials.json`
- `best_params.json`

### Train

Train reads the canonical history dataset, trims it to the exact block count needed for `sample_count`, prepares chronological train/validation/test splits, trains the selected model family, and writes:

- `artifact.json`
- `model.pt`
- `train_report.json`

If `artifact.variant=tuned`, train applies parameters from `best_params.json` before building the training spec.

### Simulate

Simulate loads the trained artifact, combines the required trailing history context with the canonical evaluation dataset, runs inference over the evaluation-day examples, and executes repeated Poisson-arrival simulations on those examples.

## Storage Layout

Datasets:

- `artifacts/datasets/<chain>/<dataset_id>/history/...`
- `artifacts/datasets/<chain>/<dataset_id>/evaluation/...`
- `artifacts/datasets/<chain>/<dataset_id>/.spice/metadata.json`

Models:

- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/<variant>/<study_id>/artifact.json`
- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/<variant>/<study_id>/model.pt`
- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/<variant>/<study_id>/train_report.json`

Tuning:

- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuned/<study_id>/tuning/study.json`
- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuned/<study_id>/tuning/trials.json`
- `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuned/<study_id>/tuning/best_params.json`

## DVC Surface

[dvc.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/dvc.yaml) defines four stages:

1. `acquire`
2. `tune`
3. `train`
4. `simulate`

Each stage reads [params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml), scopes its own inputs, and dispatches the same workflow code used by the direct `spice-*` entrypoints.

## Tests

The test suite covers:

- config composition and validation
- acquisition planning and reuse behavior
- data preparation and validation
- tuning, training, and simulation workflow behavior
- artifact and metadata persistence
