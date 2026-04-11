# SPICE

Temporal-module baseline for SPICE-style fee-timing experiments.

This repository is intentionally scoped to the temporal module only:

- acquire raw EVM block data
- enrich missing block fields
- build fixed-horizon temporal datasets
- train baseline sequence models
- run evaluation-day temporal simulations
- tune model hyperparameters

It does not implement the broader SPICE spatial/oracle/reputation system.

## Stack

- `Hydra` for runtime configuration and composition
- `DVC` for reproducible stages, artifact tracking, and future remote execution
- `MLflow` for run tracking, params, metrics, and artifacts
- `Lightning` + `TorchMetrics` for training orchestration
- `Optuna` for hyperparameter optimization
- `web3.py` for RPC transport
- `Pandera` + `Polars` for dataset validation and parquet/table work
- `scikit-learn` for scaling
- `NumPy` + `PyTorch` for dataset math, modeling, inference, and simulation

There is no legacy compatibility layer. The repository does not expose `spice.api`,
the old `spice` Typer CLI, snapshot registries, or the old custom YAML/settings
loader.

## Layout

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

Key runtime paths:

- raw datasets: `artifacts/datasets/<chain>/raw/...`
- enriched datasets: `artifacts/datasets/<chain>/enriched/...`
- chain dataset metadata: `artifacts/datasets/<chain>/.spice/metadata.json`
- model artifacts: `artifacts/models/<chain>/<family>/<delay>s/...`
- simulation reports: `artifacts/models/<chain>/<family>/<delay>s/simulation_report.json`
- tuning outputs: `artifacts/models/<chain>/<family>/<delay>s/tuning/...`
- MLflow store: `artifacts/mlruns/`

## Setup

```bash
.venv/bin/pip install -e .
```

If you use the `direct` provider, export chain RPC URLs:

- `ETHEREUM_RPC_URL`
- `POLYGON_RPC_URL`
- `AVALANCHE_RPC_URL`

If you use the `alchemy` provider, export:

- `ALCHEMY_API_KEY`

## Running

Use DVC as the primary surface:

```bash
.venv/bin/dvc repro acquire
.venv/bin/dvc repro train
.venv/bin/dvc repro simulate
.venv/bin/dvc repro tune
```

The baseline DVC variables live in [params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml). Hydra defaults live under [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf).

You can also run the workflow entrypoints directly:

```bash
.venv/bin/spice-acquire chain=ethereum provider=publicnode
.venv/bin/spice-train chain=ethereum model=lstm training.device=cpu
.venv/bin/spice-simulate chain=ethereum model=lstm training.device=cpu
.venv/bin/spice-tune chain=ethereum model=lstm tuning.n_trials=20
```

## Configuration

Hydra config groups live under [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf):

- `chain/`
- `model/`
- `provider/`
- `training/`
- `simulation/`
- `tracking/`
- `tuning/`
- `runtime/`

Runtime validation happens in [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py). That layer enforces the repo’s structural invariants, including transformer head divisibility and provider endpoint availability.

## Verification

```bash
.venv/bin/ruff check src/spice tests
.venv/bin/pytest -q
```
