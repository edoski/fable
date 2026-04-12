# SPICE

SPICE is a temporal fee-timing pipeline for EVM chains. It acquires canonical block datasets, builds fixed-horizon temporal training data, tunes models, trains artifacts, and runs evaluation-day simulations.

## Stack

- `Hydra` + `OmegaConf` for config composition
- `DVC` for stage orchestration
- `MLflow` for run tracking
- `Lightning` + `PyTorch` for training
- `Optuna` for tuning
- `web3.py` for RPC access
- `Polars` + `Pandera` for block-table validation and dataset IO

## Repository Layout

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

## Setup

```bash
.venv/bin/pip install -e .
```

Provider credentials:

- `direct`: export `ETHEREUM_RPC_URL`, `POLYGON_RPC_URL`, `AVALANCHE_RPC_URL`
- `alchemy`: export `ALCHEMY_API_KEY`

## Main Commands

DVC stages:

```bash
.venv/bin/dvc repro acquire
.venv/bin/dvc repro tune
.venv/bin/dvc repro train
.venv/bin/dvc repro simulate
.venv/bin/dvc repro
```

Direct entrypoints:

```bash
.venv/bin/spice-acquire presets.chain=ethereum presets.provider=publicnode
.venv/bin/spice-tune presets.model=lstm tuning.trial_count=20
.venv/bin/spice-train presets.model=lstm training.device=cpu
.venv/bin/spice-simulate presets.model=lstm training.device=cpu
```

On macOS, DVC stages run through `./bin/spice-awake`, which wraps long runs with `caffeinate` when available.

## Config Surface

[params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml) is the user-edited experiment spec. It selects presets and sets first-order experiment values.

Core fields:

- `presets.chain`
- `presets.provider`
- `presets.model`
- `dataset.id`
- `evaluation.date`
- `dataset.temporal.max_delay_seconds`
- `dataset.temporal.lookback_seconds`
- `dataset.sampling.sample_count`
- `acquisition.history_sample_budget` optional
- `split.*`
- `training.*`
- `model.*` for the selected model family
- `simulation.*`
- `artifact.variant`
- `study.id`

Semantics:

- `evaluation.date` names the fixed final UTC evaluation day
- evaluation window is `[evaluation.date 00:00 UTC, next day 00:00 UTC)`
- history ends at the evaluation start boundary
- `dataset.sampling.sample_count` drives training and tuning sample count
- `acquisition.history_sample_budget` optionally caches more history than training uses; when omitted it falls back to `sample_count`

Hydra preset files under [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf) resolve chain, provider, model-family, runtime, and RPC-profile defaults.

## Data and Artifacts

Output layout:

- history blocks: `artifacts/datasets/<chain>/<dataset_id>/history/...`
- evaluation blocks: `artifacts/datasets/<chain>/<dataset_id>/evaluation/...`
- dataset metadata: `artifacts/datasets/<chain>/<dataset_id>/.spice/metadata.json`
- model artifacts: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/<variant>/<study_id>/...`
- tuning outputs: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuned/<study_id>/tuning/...`

## Acquisition Behavior

Acquire is block-planned.

- It resolves the evaluation start block from `evaluation.date`
- It counts backward by the exact required history block count
- It reuses existing canonical datasets when they already cover the requested window
- It extends existing history by fetching only the missing prefix when more history is needed
- It reuses overlapping evaluation datasets and fetches only missing edges
- It writes metadata after atomic promotion of successful outputs

## Verification

```bash
.venv/bin/ruff check src/spice tests
.venv/bin/pyright
.venv/bin/pytest -q
```
